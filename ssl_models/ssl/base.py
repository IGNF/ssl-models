"""Base SSL model: the multi-modal, multi-temporal token pipeline shared by the
MAE and DINOv3 models.

Tokens are per modality [B, D, L, C], with D = band groups x dates (band-group
major), L = h x w patches and C = embedding dim, or [B, G, D, L, C] with the G band
groups apart (D = dates). They are grouped into sequences [B, L, C] (with the B and
L of the sequences) according to the fusion mode, with `group_mods` /
`ungroup_mods`:
    - "msgfm" / "shared": one sequence per (modality, date), one shared encoder,
    - "monotemp": one sequence per (modality, date), one encoder per modality,
    - "mod": one sequence per modality, one encoder per modality,
    - "group": one sequence per group of modalities, one encoder per group.
In "mod" / "group" modes, the last `inter_depth` blocks are shared and run on
the concatenation of all sequences.

Views of different sizes (e.g. the DINO global and local views) go through the
same pipeline: once ungrouped, the token grids follow from the number of tokens
L = h x w per modality, and the ungrouping matches the length of the grouped
sequences with the numbers of tokens of the views, given at init. In "mod" /
"group" modes, only complete sequences can be ungrouped, not the visible tokens
of masked sequences (MAE).

The CLS tokens (if any) are one per (band group, date) of each modality: they
are tokens [B, D, 1, C] on a 1 x 1 grid, with the same grouping, date encodings
and RoPE as the patch tokens. In the encoder, they are prepended to the patch
tokens of their sequence.

The shared steps are the embedding of the patch and CLS tokens (`embed`), the
position (absolute sin-cos or RoPE) and date encodings (`add_encodings`,
`add_rope`), the drop of whole modalities / dates (`keep_view`), the encoder
stacks on the sequences of registers, CLS tokens and patch tokens (`encode`),
and the probing / finetuning heads (`compute_logits`).
"""

import math
from functools import partial, reduce
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn

from ssl_models.conf.dataset.utils import DatasetConfig, RasterConfig
from ssl_models.layers.embed import Patchify
from ssl_models.layers.head import ClassificationHead, PixelifyHead
from ssl_models.layers.rope import Rope
from ssl_models.layers.utils import (
    encode_dates,
    group_mods,
    posemb_sincos_2d,
    reshape_encoding,
    ungroup_mods,
)


class BaseSSL(nn.Module):
    """Base SSL model: patch and CLS tokens, encodings, encoder, heads.

    The encoder is one ViT stack per model name (built from the `transformer`
    partial), then a shared stack of `inter_depth` blocks; its output is
    normalized by the final norm of its last stacks, and with `inter_norm` the
    outputs of the stacks per model name are also normalized before the shared
    stack. Subclasses build their pretraining forward on top of `encode`.
    """

    def __init__(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        dataset: DatasetConfig,
        fusion_mode: Literal["msgfm", "shared", "monotemp", "mod", "group"],
        embed_dim: int,
        decoder_dim: int | None,
        patch_size: dict[str, int],
        transformer: partial,
        depth: int,
        inter_depth: int = 0,
        inter_norm: bool = False,
        type_head: Literal["linear", "attentive"] = "attentive",
        patch_norm: bool = True,
        pos_enc: Literal["abs", "rope"] = "abs",
        fac_abs_enc: float = 1.0,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        use_cls: bool = False,
        num_registers: int = 0,
        enc_mask_token: bool = False,
        rope_head_dim: int | None = None,
        decoder_rope_head_dim: int | None = None,
        rope_base: float = 100.0,
        rope_shift: float | None = None,
        rope_jitter: float | None = None,
        rope_rescale: float | None = None,
        view_drop_mod: float | None = None,
        view_drop_dates: float | None = None,
        grid_size_local: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.fusion_mode = fusion_mode
        self.embed_dim = embed_dim
        self.pos_enc = pos_enc
        self.fac_date_enc = fac_date_enc
        self.date_dim = date_dim

        # one encoder per model name
        match fusion_mode:
            case "msgfm" | "shared":
                self.name_models = ["shared"]
            case "monotemp" | "mod":
                self.name_models = list(dataset.inputs)
            case "group":
                self.name_models = list(
                    dict.fromkeys(name_group for _, name_group in dataset.groups),
                )

        # encoder stacks
        if inter_depth and fusion_mode not in ("mod", "group"):
            msg = f"Intermediate fusion not compatible with fusion mode {fusion_mode}."
            raise NotImplementedError(msg)
        if not 0 <= inter_depth <= depth:
            msg = f"Invalid inter depth {inter_depth} for depth {depth}."
            raise ValueError(msg)
        self.encoder = nn.ModuleDict(
            {
                name: transformer(
                    depth=depth - inter_depth,
                    final_norm=inter_norm or not inter_depth,
                )
                for name in self.name_models
            },
        )
        self.encoder_inter = (
            transformer(depth=inter_depth, final_norm=True) if inter_depth else None
        )

        # patch embeddings, shared between modalities with the same `name_embed`
        self.patch_embed = nn.ModuleDict()
        self.mod_embed, self.len_bands, self.num_dates, self.grid_size = {}, {}, {}, {}
        for name_mod, mod in dataset.inputs.items():
            num_bands = (
                [mod.bands] if isinstance(mod.bands, int) else list(map(len, mod.bands))
            )
            self.len_bands[name_mod] = len(num_bands)
            self.num_dates[name_mod] = mod.num_dates * len(num_bands)
            if mod.image_size % patch_size[name_mod]:
                msg = (
                    f"Image size {mod.image_size} of {name_mod} is not a multiple "
                    f"of its patch size {patch_size[name_mod]}."
                )
                raise ValueError(msg)
            self.grid_size[name_mod] = mod.image_size // patch_size[name_mod]

            name_embed = mod.name_embed if mod.name_embed else name_mod
            self.mod_embed[name_mod] = name_embed
            if name_embed in self.patch_embed:
                embed = self.patch_embed[name_embed]
                if (
                    embed.num_bands != num_bands
                    or embed.patch_size != patch_size[name_mod]
                ):
                    msg = f"Patchify layers inconsistently defined for {name_embed}"
                    raise ValueError(msg)
            else:
                self.patch_embed[name_embed] = Patchify(
                    mod.bands,
                    embed_dim,
                    patch_size[name_mod],
                    norm=patch_norm,
                )

        self.group = partial(group_mods, fusion_mode=fusion_mode, groups=dataset.groups)
        # tokens per (band group, date) to ungroup: patch tokens of the views (global,
        # and local if any), and CLS tokens
        grid_sizes = [self.grid_size] + ([grid_size_local] if grid_size_local else [])
        num_tokens = [
            {name_mod: grid_size[name_mod] ** 2 for name_mod in grid_size}
            for grid_size in grid_sizes
        ]
        if use_cls:
            num_tokens.append(dict.fromkeys(self.grid_size, 1))
        self.ungroup = partial(
            ungroup_mods,
            fusion_mode=fusion_mode,
            groups=dataset.groups,
            num_dates=self.num_dates,
            num_tokens=num_tokens,
        )

        # position encodings: absolute sin-cos (fixed) or RoPE
        self.grid_pos_enc = dataset.grid_pos_enc or reduce(
            math.lcm,
            self.grid_size.values(),
        )
        match pos_enc:
            case "abs":
                self.register_buffer(
                    "enc_pos_enc",
                    posemb_sincos_2d(
                        h=self.grid_pos_enc,
                        w=self.grid_pos_enc,
                        dim=embed_dim,
                        date_dim=date_dim,
                        fac_abs_enc=fac_abs_enc,
                    ),
                    persistent=False,
                )
                if decoder_dim is not None:
                    self.register_buffer(
                        "dec_pos_enc",
                        posemb_sincos_2d(
                            h=self.grid_pos_enc,
                            w=self.grid_pos_enc,
                            dim=decoder_dim,
                            date_dim=self.date_dim,
                        ),
                        persistent=False,
                    )
                else:
                    self.dec_pos_enc = None
                self.enc_rope = None
                self.dec_rope = None
            case "rope":
                if rope_head_dim is None:
                    msg = "RoPE needs the attention head dim of the encoder."
                    raise ValueError(msg)
                if decoder_dim is not None and decoder_rope_head_dim is None:
                    msg = "RoPE needs the attention head dim of the decoder."
                    raise ValueError(msg)
                self.enc_pos_enc = None
                self.dec_pos_enc = None
                self.enc_rope = Rope(
                    head_dim=rope_head_dim,
                    base=rope_base,
                    shift_coords=rope_shift,
                    jitter_coords=rope_jitter,
                    rescale_coords=rope_rescale,
                )
                # no coordinate augmentations for the decoder (encoder regularization)
                self.dec_rope = (
                    Rope(head_dim=decoder_rope_head_dim, base=rope_base)
                    if decoder_dim is not None
                    else None
                )
            case _:
                msg = f"Invalid position encoding {pos_enc}. Expected 'abs' or 'rope'."
                raise ValueError(msg)

        # CLS tokens, one per modality, repeated over the band groups and dates
        self.use_cls = use_cls
        self.cls_token = nn.ParameterDict(
            {
                name_mod: nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
                for name_mod in (dataset.inputs if use_cls else [])
            },
        )
        # registers, one set per encoder
        self.num_registers = num_registers
        self.register_tokens = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, num_registers, embed_dim))
                for name in (self.name_models if num_registers else [])
            },
        )
        for param in (*self.cls_token.values(), *self.register_tokens.values()):
            nn.init.trunc_normal_(param, std=0.02)

        # drop of whole modalities / dates (`keep_view`), a sequence can then be
        # fully dropped and needs CLS tokens or registers to attend to
        if (view_drop_mod or view_drop_dates) and not (use_cls or num_registers):
            msg = "Dropping modalities / dates needs CLS tokens or registers."
            raise ValueError(msg)
        self.view_drop_mod = view_drop_mod
        self.view_drop_dates = view_drop_dates

        # mask token replacing the masked patch embeddings (`DINOv3.mask`)
        self.enc_mask_token = (
            nn.Parameter(torch.zeros(embed_dim)) if enc_mask_token else None
        )

        # probing / finetuning heads, on the patch tokens or CLS tokens
        self.heads = nn.ModuleDict()
        for name_target, target in dataset.targets.items():
            if isinstance(target, RasterConfig):
                if dataset.ref_input is None:
                    msg = f"Ref input must be provided for raster target {name_target}"
                    raise ValueError(msg)
                target_image_size = round(
                    dataset.crop_meters / target.resolution_meters,
                )
                ref_grid_size = self.grid_size[dataset.ref_input]
                if target_image_size % ref_grid_size:
                    msg = (
                        f"Target image size {target_image_size} "
                        f"is not a multiple of ref input grid {ref_grid_size}"
                    )
                    raise ValueError(msg)
                self.heads[name_target] = PixelifyHead(
                    type_head,
                    embed_dim,
                    target.num_classes,
                    target_image_size // ref_grid_size,
                )
            else:
                self.heads[name_target] = ClassificationHead(
                    type_head,
                    embed_dim,
                    target.num_classes,
                )

    def embed(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
        """Embed patches into tokens [B, D, L, C], and create the CLS tokens
        [B, D, 1, C] (None if no CLS), both grouped."""
        x = {}
        for name_mod in self.dataset.inputs:
            name_embed = self.mod_embed[name_mod]
            x[name_mod] = self.patch_embed[name_embed](batch[name_mod])

        if not self.use_cls:
            return self.group(x), None

        x_cls = {
            name_mod: self.cls_token[name_mod].expand(
                *x[name_mod].shape[:2], 1, x[name_mod].shape[-1]
            )
            for name_mod in self.dataset.inputs
        }
        return self.group(x), self.group(x_cls)

    def add_enc_encodings(
        self,
        x: dict[str, Tensor],
        x_cls: dict[str, Tensor] | None,
        batch: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
        """`add_encodings` with the absolute position encodings of the encoder for
        the patch tokens, without (date encodings only) for the CLS tokens."""
        x = self.add_encodings(x, batch, abs_pos_enc=self.enc_pos_enc)
        if x_cls is not None:
            x_cls = self.add_encodings(x_cls, batch, abs_pos_enc=None)
        return x, x_cls

    def add_dec_encodings(
        self,
        x: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """`add_encodings` with the absolute position encodings of the decoder."""
        return self.add_encodings(x, batch, abs_pos_enc=self.dec_pos_enc)

    def add_encodings(
        self,
        x: dict[str, Tensor],
        batch: dict[str, Tensor],
        abs_pos_enc: Tensor | None,
    ) -> dict[str, Tensor]:
        """Add the absolute position encodings `abs_pos_enc` (if any) and the
        date encodings to the tokens."""
        x = self.ungroup(x)

        for name_mod in x:
            grid_size = round(x[name_mod].shape[2] ** 0.5)
            pos_encoding = (
                reshape_encoding(abs_pos_enc, grid_size)
                if abs_pos_enc is not None
                else 0
            )
            date_encoding = encode_dates(
                dates=batch[f"{name_mod}_dates"],
                ref_date=batch["ref_date"],
                dim=x[name_mod].shape[-1],
                date_dim=self.date_dim,
                fac_date_enc=self.fac_date_enc,
                grid_size=grid_size,
                len_bands=self.len_bands[name_mod],
            )
            x[name_mod] = x[name_mod] + pos_encoding + date_encoding

        return self.group(x)

    def add_enc_rope(
        self,
        x: dict[str, Tensor],
        x_cls: dict[str, Tensor] | None,
    ) -> tuple[dict[str, Tensor] | None, dict[str, Tensor] | None]:
        """`add_rope` with the RoPE of the encoder, for the patch tokens and the CLS
        tokens (at the center of the view), with the same augmentation."""
        if self.enc_rope is None:
            return None, None

        rope_embed = partial(self.enc_rope, augment=self.enc_rope.sample_augment())
        rope = self.add_rope(x, rope_embed=rope_embed)
        rope_cls = (
            self.add_rope(x_cls, rope_embed=rope_embed) if x_cls is not None else None
        )
        return rope, rope_cls

    def add_dec_rope(
        self,
        x: dict[str, Tensor],
    ) -> dict[str, Tensor] | None:
        """`add_rope` with the RoPE of the decoder."""
        return self.add_rope(x, rope_embed=self.dec_rope)

    def add_rope(
        self,
        x: dict[str, Tensor],
        rope_embed: Rope | partial | None,
    ) -> dict[str, Tensor] | None:
        """RoPE (sin, cos) {group: [B, L, 2 * head_dim]} of the tokens from
        `rope_embed` (None if no RoPE)."""
        if rope_embed is None:
            return None

        rope = {}
        for name_mod, x_mod in self.ungroup(x).items():
            rope[name_mod] = rope_embed(
                batch_size=x_mod.shape[0],
                num_dates=self.num_dates[name_mod],
                grid_size=round(x_mod.shape[2] ** 0.5),
                device=x_mod.device,
            )

        return self.group(rope)

    def keep_view(
        self,
        x: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        teacher: bool = False,
    ) -> dict[str, Tensor] | None:
        """Sample the kept tokens of a view, dropping whole modalities / dates (in
        pretraining, not for the teacher).

        Whole modalities are dropped with probability `view_drop_mod` (only if
        there are several modalities), single dates with probability
        `view_drop_dates` (only for multi-temporal modalities). At least one
        (modality, date) is kept per sample: if all were dropped, the sample
        keeps everything.

        Returns:
            {group: [B, L, 1]} bool, True = kept token (attended), or None if no
            drop (not in pretraining, teacher, or no drop configured).

        """
        if (
            ssl_phase != "pretrain"
            or teacher
            or not (self.view_drop_mod or self.view_drop_dates)
        ):
            return None

        x = self.ungroup(x)

        x_ref = next(iter(x.values()))
        B, device = x_ref.shape[0], x_ref.device
        multi_mod = len(self.dataset.inputs) > 1
        keep = {}
        for name_mod, mod in self.dataset.inputs.items():
            D = mod.num_dates
            keep[name_mod] = torch.ones(B, D, dtype=torch.bool, device=device)
            if self.view_drop_mod and multi_mod:
                keep[name_mod] &= torch.rand(B, 1, device=device) >= self.view_drop_mod
            if self.view_drop_dates and D > 1:
                keep[name_mod] &= (
                    torch.rand(B, D, device=device) >= self.view_drop_dates
                )
        any_kept = torch.stack([k.any(dim=1) for k in keep.values()], dim=1).any(dim=1)

        # dates [B, D] -> with band groups [B, G, D, L, 1] -> tokens [B, D, L, 1]
        keep = {
            name_mod: (keep_mod | ~any_kept[:, None])[:, None, :, None, None]
            .expand(-1, self.len_bands[name_mod], -1, x[name_mod].shape[2], -1)
            .flatten(1, 2)
            for name_mod, keep_mod in keep.items()
        }
        return self.group(keep)

    def add_cls(
        self,
        x: dict[str, Tensor],
        x_cls: dict[str, Tensor] | None,
        rope: dict[str, Tensor] | None,
        rope_cls: dict[str, Tensor] | None,
        attn_mask: dict[str, Tensor] | None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor] | None, dict[str, Tensor] | None]:
        """Prepend the CLS tokens (if any, always attended) to each sequence."""
        if x_cls is None:
            return x, rope, attn_mask

        x = {
            name_group: torch.cat([x_cls[name_group], x[name_group]], dim=1)
            for name_group in x
        }
        if rope is not None:
            rope = {
                name_group: torch.cat([rope_cls[name_group], rope[name_group]], dim=1)
                for name_group in x
            }
        if attn_mask is not None:
            attn_mask = {
                name_group: F.pad(
                    attn_mask[name_group], (x_cls[name_group].shape[1], 0), value=True
                )
                for name_group in x
            }
        return x, rope, attn_mask

    def remove_cls(
        self,
        x: dict[str, Tensor],
        x_cls: dict[str, Tensor] | None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
        """Split each sequence into its patch tokens and its CLS tokens (if any, as
        many as the `x_cls` prepended by `add_cls`)."""
        if x_cls is None:
            return x, None

        x_cls = {
            name_group: x[name_group][:, : x_cls[name_group].shape[1]]
            for name_group in x
        }
        x = {
            name_group: x[name_group][:, x_cls[name_group].shape[1] :]
            for name_group in x
        }
        return x, x_cls

    def add_registers(
        self,
        x: dict[str, Tensor],
        rope: dict[str, Tensor] | None,
        attn_mask: dict[str, Tensor] | None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor] | None, dict[str, Tensor] | None]:
        """Prepend the registers (if any, always attended) to each sequence."""
        if not self.num_registers:
            return x, rope, attn_mask

        x = dict(x)
        rope = dict(rope) if rope is not None else None
        attn_mask = dict(attn_mask) if attn_mask is not None else None
        for name_group in x:
            prefix = (
                self.register_tokens[name_group]
                if name_group in self.register_tokens
                else self.register_tokens["shared"]
            )
            prefix = prefix.expand(x[name_group].shape[0], -1, -1).to(
                x[name_group].dtype
            )
            x[name_group] = torch.cat(
                [prefix, x[name_group]],
                dim=1,
            )

            if rope is not None:  # identity rotation for the registers
                sin, cos = rope[name_group].chunk(2, dim=-1)
                rope[name_group] = torch.cat(
                    [
                        F.pad(sin, (0, 0, self.num_registers, 0), value=0.0),
                        F.pad(cos, (0, 0, self.num_registers, 0), value=1.0),
                    ],
                    dim=-1,
                )
            if attn_mask is not None:
                attn_mask[name_group] = F.pad(
                    attn_mask[name_group],
                    (self.num_registers, 0),
                    value=True,
                )

        return x, rope, attn_mask

    def remove_registers(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Remove the registers (if any) of each sequence."""
        return {name_group: x[name_group][:, self.num_registers :] for name_group in x}

    def encode_or_decode(
        self,
        x: dict[str, Tensor],
        rope: dict[str, Tensor] | None,
        attn_mask: dict[str, Tensor] | None,
        model: nn.Module,
    ) -> dict[str, Tensor]:
        """Apply encoder or decoder (with RoPE and attention mask, if any)."""
        for name_group in x:
            model_group = model[name_group] if name_group in model else model["shared"]
            x[name_group] = model_group(
                x[name_group],
                rope=rope[name_group] if rope is not None else None,
                attn_mask=attn_mask[name_group] if attn_mask is not None else None,
            )

        return x

    def encode_or_decode_all(
        self,
        x: dict[str, Tensor],
        rope: dict[str, Tensor] | None,
        attn_mask: dict[str, Tensor] | None,
        model: nn.Module,
    ) -> dict[str, Tensor]:
        """Apply encoder or decoder on all groups (concatenated along tokens)."""
        name_groups = list(x.keys())
        split_groups = tuple(x[name_group].shape[1] for name_group in name_groups)

        x_groups = torch.cat([x[name_group] for name_group in name_groups], dim=1)
        rope = (
            torch.cat([rope[name_group] for name_group in name_groups], dim=1)
            if rope is not None
            else None
        )
        attn_mask = (
            torch.cat([attn_mask[name_group] for name_group in name_groups], dim=1)
            if attn_mask is not None
            else None
        )

        x_groups = model(x_groups, rope=rope, attn_mask=attn_mask)
        x_groups = x_groups.split(split_groups, dim=1)
        for name_group, x_group in zip(name_groups, x_groups):
            x[name_group] = x_group

        return x

    def encode(
        self,
        x: dict[str, Tensor],
        x_cls: dict[str, Tensor] | None,
        rope: dict[str, Tensor] | None,
        rope_cls: dict[str, Tensor] | None,
        attn_mask: dict[str, Tensor] | None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
        """Apply encoder: stacks per model name, then shared stack (if any), on the
        sequences of [registers, CLS tokens, patch tokens].

        Returns the encoded patch tokens, and CLS tokens (if any), grouped.
        """
        x, rope, attn_mask = self.add_cls(x, x_cls, rope, rope_cls, attn_mask)
        x, rope, attn_mask = self.add_registers(x, rope, attn_mask)

        x = self.encode_or_decode(x, rope, attn_mask, model=self.encoder)
        if self.encoder_inter is not None:
            x = self.encode_or_decode_all(x, rope, attn_mask, model=self.encoder_inter)

        x = self.remove_registers(x)
        return self.remove_cls(x, x_cls)

    def compute_logits(
        self,
        x: dict[str, Tensor],
        x_cls: dict[str, Tensor] | None,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
    ) -> dict[str, Tensor]:
        """Logits of the heads from the encoded tokens (grouped): the patch tokens
        for segmentation, the CLS tokens (if any, else the patch tokens) for
        classification."""
        x = self.ungroup(x)
        grid_size = {name_mod: round(x[name_mod].shape[2] ** 0.5) for name_mod in x}

        ref_input = self.dataset.ref_input
        if ref_input is not None:
            x_seg = {}
            for name_mod in x:
                x_seg[name_mod] = rearrange(
                    x[name_mod],
                    "b d (h w) c -> (b d) c h w",
                    h=grid_size[name_mod],
                )
                x_seg[name_mod] = F.interpolate(
                    x_seg[name_mod],
                    (grid_size[ref_input],) * 2,
                    mode="bilinear",
                )
                x_seg[name_mod] = rearrange(
                    x_seg[name_mod],
                    "(b d) c h w -> b d (h w) c",
                    d=x[name_mod].shape[1],
                )
            x_seg = torch.cat(list(x_seg.values()), dim=1)

        # classification from the CLS tokens, or the patch tokens
        x_cls = self.ungroup(x_cls) if x_cls is not None else x
        x_cls = torch.cat([x_cls[name_mod].flatten(1, 2) for name_mod in x_cls], dim=1)

        logits = {}
        for name_target, target in self.dataset.targets.items():
            match target.type_target:
                case "segment" | "change_detect":
                    logits[name_target] = self.heads[name_target](
                        x_seg,
                        ssl_phase=ssl_phase,
                    )
                case "multilabel_classif" | "classif":
                    logits[name_target] = self.heads[name_target](
                        x_cls,
                        ssl_phase=ssl_phase,
                    )
        return logits
