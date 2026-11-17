"""Multi-modal, multi-temporal RoPE ViT backbone for DINOv3-style pretraining.

Tokens follow the same layout as the MAE models: per modality [B, S, L, D], with
S = band groups x dates (band-group major) and L = h x w patches, grouped into
sequences according to the fusion mode with `group_mods` / `ungroup_mods`:
    - "shared": one sequence per (modality, date), one shared encoder,
    - "monotemp": one sequence per (modality, date), one encoder per modality,
    - "mod": one sequence per modality, one encoder per modality,
    - "group": one sequence per group of modalities, one encoder per group.
In "mod" / "group" modes, the last `inter_depth` blocks are shared and run on
the concatenation of all sequences.

Each sequence is prefixed with its own [CLS, registers] tokens. The image-level
embedding is the mean of the CLS tokens of the sequences that have at least one
kept (not dropped) patch. RoPE coordinates are normalized to [-1, 1] within the
view, so that patches of different modalities at the same location share their
position. Dates are encoded additively, as in the MAE models.
"""

from functools import partial
from typing import Literal

import torch
from torch import Tensor, nn

from ssl_models.conf.dataset.utils import DatasetConfig, InputConfig
from ssl_models.layers.embed import Patchify
from ssl_models.layers.utils import encode_dates, group_mods, ungroup_mods
from ssl_models.layers.vit import (
    RopePositionEmbedding,
    Transformer,
    build_norm,
    grid_coords,
    pad_rope,
)

DIM0_MODES = ("shared", "monotemp")  # dates along the batch dim
DIM1_MODES = ("mod", "group")  # dates along the token dim
VIEW_TYPES = ("global", "local")


def dino_patch_size(mod: InputConfig) -> int:
    """DINO patch size of a modality, defaulting to the MAE patch size."""
    return mod.patch_size.dino if mod.patch_size.dino else mod.patch_size.mae


def local_view_size(
    image_size: int,
    patch_size: int,
    local_ratio: float,
    min_patches: int = 2,
) -> int:
    """Local view size: image_size * local_ratio, as a multiple of patch_size."""
    size = round(image_size * local_ratio / patch_size) * patch_size
    return min(max(size, min_patches * patch_size), image_size)


class DINOv3Backbone(nn.Module):
    """Multi-modal RoPE ViT with CLS and register tokens."""

    def __init__(  # noqa: PLR0913
        self,
        dataset: DatasetConfig,
        fusion_mode: Literal["shared", "monotemp", "mod", "group"],
        inter_depth: int,
        embed_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ffn: Literal["mlp", "swiglu"] = "mlp",
        norm: Literal["layernorm", "rmsnorm"] = "layernorm",
        n_registers: int = 4,
        layerscale_init: float | None = 1e-5,
        drop_path: float = 0.3,
        patch_norm: bool = False,
        rope_base: float = 100.0,
        rope_shift: float | None = None,
        rope_jitter: float | None = None,
        rope_rescale: float | None = None,
        local_ratio: float = 112 / 256,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
    ) -> None:
        super().__init__()
        if fusion_mode not in DIM0_MODES + DIM1_MODES:
            msg = (
                f"Invalid fusion mode {fusion_mode} for DINO. "
                f"Expected one of {DIM0_MODES + DIM1_MODES}."
            )
            raise ValueError(msg)
        if inter_depth and fusion_mode not in DIM1_MODES:
            msg = f"Intermediate fusion not compatible with fusion mode {fusion_mode}."
            raise NotImplementedError(msg)
        if not 0 <= inter_depth <= depth:
            msg = f"Invalid inter depth {inter_depth} for depth {depth}."
            raise ValueError(msg)

        self.dataset = dataset
        self.fusion_mode = fusion_mode
        self.embed_dim = embed_dim
        self.n_prefix = 1 + n_registers
        self.n_registers = n_registers
        self.patch_norm = patch_norm
        self.local_ratio = local_ratio
        self.fac_date_enc = fac_date_enc
        self.date_dim = date_dim

        self.mod_embed, self.len_bands, self.num_dates = {}, {}, {}
        self.view_size, self.grid_size = {}, {}
        self.date_encoding, self.ungroup = {}, {}
        self.patch_embed = nn.ModuleDict()
        name_models = self._init_dataset(self.dataset)

        # encoders, with a uniform drop path rate over all blocks (as in DINOv3)
        transformer = partial(
            Transformer,
            dim=embed_dim,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ffn=ffn,
            norm=norm,
            layerscale_init=layerscale_init,
            drop_path=drop_path,
        )
        self.encoder = nn.ModuleDict(
            {name: transformer(depth=depth - inter_depth) for name in name_models},
        )
        self.encoder_inter = transformer(depth=inter_depth) if inter_depth else None

        # prefix tokens, one set per encoder
        self.cls_token = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1, 1, embed_dim)) for name in name_models},
        )
        self.register_tokens = nn.ParameterDict(
            (
                {
                    name: nn.Parameter(torch.zeros(1, n_registers, embed_dim))
                    for name in name_models
                }
                if n_registers
                else {}
            ),
        )
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        for param in (*self.cls_token.values(), *self.register_tokens.values()):
            nn.init.trunc_normal_(param, std=0.02)

        self.norm = build_norm(norm, embed_dim)
        self.rope = RopePositionEmbedding(
            head_dim=embed_dim // heads,
            base=rope_base,
            shift_coords=rope_shift,
            jitter_coords=rope_jitter,
            rescale_coords=rope_rescale,
        )

    def _init_dataset(self, dataset: DatasetConfig) -> list[str]:
        """Set up dataset-dependent structures, return the encoder names."""
        for name_mod, mod in dataset.inputs.items():
            num_bands = (
                [mod.bands] if isinstance(mod.bands, int) else list(map(len, mod.bands))
            )
            self.len_bands[name_mod] = len(num_bands)
            self.num_dates[name_mod] = mod.num_dates * len(num_bands)

            patch_size = dino_patch_size(mod)
            if mod.image_size % patch_size:
                msg = (
                    f"Image size {mod.image_size} of {name_mod} is not a multiple "
                    f"of its patch size {patch_size}."
                )
                raise ValueError(msg)

            name_embed = mod.name_embed if mod.name_embed else name_mod
            self.mod_embed[name_mod] = name_embed
            if name_embed in self.patch_embed:
                embed = self.patch_embed[name_embed]
                if embed.num_bands != num_bands or embed.patch_size != patch_size:
                    msg = f"Patchify layers inconsistently defined for {name_embed}"
                    raise ValueError(msg)
            else:
                self.patch_embed[name_embed] = Patchify(
                    mod.bands,
                    self.embed_dim,
                    patch_size,
                    norm=self.patch_norm,
                )

        self.group = partial(
            group_mods,
            fusion_mode=self.fusion_mode,
            groups=dataset.groups,
        )
        for view in VIEW_TYPES:
            self.view_size[view] = {
                name_mod: (
                    mod.image_size
                    if view == "global"
                    else mod.local_size
                    or local_view_size(
                        mod.image_size,
                        dino_patch_size(mod),
                        self.local_ratio,
                    )
                )
                for name_mod, mod in dataset.inputs.items()
            }
            self.grid_size[view] = {
                name_mod: self.view_size[view][name_mod] // dino_patch_size(mod)
                for name_mod, mod in dataset.inputs.items()
            }
            self.ungroup[view] = partial(
                ungroup_mods,
                fusion_mode=self.fusion_mode,
                groups=dataset.groups,
                num_dates=self.num_dates,
                grid_size=self.grid_size[view],
            )
            self.date_encoding[view] = {
                name_mod: partial(
                    encode_dates,
                    dim=self.embed_dim,
                    date_dim=self.date_dim,
                    fac_date_enc=self.fac_date_enc,
                    grid_size=self.grid_size[view][name_mod],
                    len_bands=self.len_bands[name_mod],
                )
                for name_mod in dataset.inputs
            }

        match self.fusion_mode:
            case "shared":
                return ["shared"]
            case "monotemp" | "mod":
                return list(dataset.inputs)
            case "group":
                return list(
                    dict.fromkeys(name_group for _, name_group in dataset.groups)
                )

    def token_shapes(
        self,
        view: Literal["global", "local"],
    ) -> dict[str, tuple[int, int, int]]:
        """Token shapes {modality: (S, h, w)} of a view."""
        return {
            name_mod: (num_dates, grid, grid)
            for (name_mod, num_dates), grid in zip(
                self.num_dates.items(),
                self.grid_size[view].values(),
            )
        }

    def _model_name(self, name_seq: str) -> str:
        return name_seq if name_seq in self.encoder else "shared"

    def _prefix(self, name_model: str, n: int, dtype: torch.dtype) -> Tensor:
        prefix = [self.cls_token[name_model]]
        if self.n_registers:
            prefix.append(self.register_tokens[name_model])
        return torch.cat(prefix, dim=1).expand(n, -1, -1).to(dtype)

    def forward(  # noqa: PLR0912, PLR0915, C901
        self,
        inputs: dict[str, Tensor],
        view: Literal["global", "local"],
        ibot_mask: dict[str, Tensor] | None = None,
        keep: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Encode a view.

        Args:
            inputs: modalities [N, T, C, H, W] at the view size,
                "{modality}_dates" [N, T, 3] and "ref_date" [N, 1, 3].
            view: "global" or "local", which sets the expected grid sizes.
            ibot_mask: {modality: [N, S, L]} bool, True = replaced by mask token.
            keep: {modality: [N, T]} bool, False = (modality, date) dropped.

        Returns:
            "cls": [N, D] image-level embedding,
            "patch": {modality: [N, S, L, D]} patch tokens,
            "patch_flat": [N, P, D] all patch tokens, modalities concatenated,
            "keep_flat": [N, P] bool, True = patch token not dropped.

        """
        dataset = self.dataset
        grid_size = self.grid_size[view]

        x, keep_tok, coords = {}, {}, {}
        for name_mod, mod in dataset.inputs.items():
            name_embed = self.mod_embed[name_mod]
            tokens = self.patch_embed[name_embed](inputs[name_mod])  # [N, S, L, D]
            N, S, L, _ = tokens.shape  # noqa: N806
            grid = grid_size[name_mod]
            if L != grid**2:
                msg = (
                    f"Got {L} tokens for {name_mod}, expected {grid}x{grid} "
                    f"for a {view} view."
                )
                raise ValueError(msg)

            if ibot_mask is not None:
                tokens = torch.where(
                    ibot_mask[name_mod][..., None],
                    self.mask_token.to(tokens.dtype),
                    tokens,
                )
            date_encoding = self.date_encoding[view][name_mod](
                dates=inputs[f"{name_mod}_dates"],
                ref_date=inputs["ref_date"],
            )
            x[name_mod] = tokens + date_encoding.to(tokens.dtype)

            keep_mod = (
                keep[name_mod]
                if keep is not None
                else torch.ones(
                    N, mod.num_dates, dtype=torch.bool, device=tokens.device
                )
            )
            G = self.len_bands[name_mod]  # noqa: N806
            keep_tok[name_mod] = (
                keep_mod[:, None, :, None]
                .expand(N, G, mod.num_dates, L)
                .reshape(N, S, L)
            )
            coords[name_mod] = grid_coords(grid, grid, device=tokens.device)[
                None,
                None,
            ].expand(1, S, -1, -1)

        group = self.group
        x = group(x)
        keep_seq = {
            name: k[..., 0]
            for name, k in group({m: k[..., None] for m, k in keep_tok.items()}).items()
        }
        coords = {name: c[0] for name, c in group(coords).items()}  # [L_seq, 2]
        names = list(x)

        # one RoPE draw for all sequences, consistent within the view
        lens = [coords[name].shape[0] for name in names]
        sin, cos = self.rope(torch.cat([coords[name] for name in names]))
        rope = {
            name: pad_rope((s, c), self.n_prefix)
            for name, s, c in zip(names, sin.split(lens), cos.split(lens))
        }

        attn_mask = {}
        for name in names:
            n_seq = x[name].shape[0]
            name_model = self._model_name(name)
            x[name] = torch.cat(
                [self._prefix(name_model, n_seq, x[name].dtype), x[name]],
                dim=1,
            )
            attn_mask[name] = (
                torch.cat(
                    [
                        torch.ones(
                            n_seq,
                            self.n_prefix,
                            dtype=torch.bool,
                            device=x[name].device,
                        ),
                        keep_seq[name],
                    ],
                    dim=1,
                )
                if keep is not None
                else None
            )
            x[name] = self.encoder[name_model](x[name], rope[name], attn_mask[name])

        if self.encoder_inter is not None:
            lens_seq = [x[name].shape[1] for name in names]
            rope_all = (
                torch.cat([rope[name][0] for name in names]),
                torch.cat([rope[name][1] for name in names]),
            )
            mask_all = (
                torch.cat([attn_mask[name] for name in names], dim=1)
                if keep is not None
                else None
            )
            x_all = self.encoder_inter(
                torch.cat([x[name] for name in names], dim=1),
                rope_all,
                mask_all,
            )
            x = dict(zip(names, x_all.split(lens_seq, dim=1)))

        # pool CLS tokens of sequences with at least one kept patch
        cls, valid, patch = [], [], {}
        for name in names:
            y = self.norm(x[name])
            cls_seq, valid_seq = y[:, 0], keep_seq[name].any(dim=1)
            if self.fusion_mode in DIM0_MODES:
                cls_seq = cls_seq.unflatten(0, (N, -1))
                valid_seq = valid_seq.unflatten(0, (N, -1))
            else:
                cls_seq, valid_seq = cls_seq[:, None], valid_seq[:, None]
            cls.append(cls_seq)
            valid.append(valid_seq)
            patch[name] = y[:, self.n_prefix :]

        cls = torch.cat(cls, dim=1)
        valid = torch.cat(valid, dim=1).to(cls.dtype)[..., None]
        cls = (cls * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

        patch = self.ungroup[view](patch)
        return {
            "cls": cls,
            "patch": patch,
            "patch_flat": torch.cat(
                [patch[name_mod].flatten(1, 2) for name_mod in dataset.inputs],
                dim=1,
            ),
            "keep_flat": torch.cat(
                [keep_tok[name_mod].flatten(1, 2) for name_mod in dataset.inputs],
                dim=1,
            ),
        }


def dinov3_tiny(**kwargs) -> DINOv3Backbone:  # noqa: ANN003
    """Construct DINOv3 backbone tiny."""
    args = {
        "embed_dim": 192,
        "depth": 12,
        "heads": 3,
        "mlp_ratio": 4.0,
        "ffn": "mlp",
    }
    args.update(kwargs)
    return DINOv3Backbone(**args)


def dinov3_small(**kwargs) -> DINOv3Backbone:  # noqa: ANN003
    """Construct DINOv3 backbone small."""
    args = {
        "embed_dim": 384,
        "depth": 12,
        "heads": 6,
        "mlp_ratio": 4.0,
        "ffn": "mlp",
    }
    args.update(kwargs)
    return DINOv3Backbone(**args)


def dinov3_base(**kwargs) -> DINOv3Backbone:  # noqa: ANN003
    """Construct DINOv3 backbone base."""
    args = {
        "embed_dim": 768,
        "depth": 12,
        "heads": 12,
        "mlp_ratio": 4.0,
        "ffn": "mlp",
    }
    args.update(kwargs)
    return DINOv3Backbone(**args)


def dinov3_large(**kwargs) -> DINOv3Backbone:  # noqa: ANN003
    """Construct DINOv3 backbone large."""
    args = {
        "embed_dim": 1024,
        "depth": 24,
        "heads": 16,
        "mlp_ratio": 4.0,
        "ffn": "mlp",
    }
    args.update(kwargs)
    return DINOv3Backbone(**args)


def dinov3_huge_plus(**kwargs) -> DINOv3Backbone:  # noqa: ANN003
    """Construct DINOv3 backbone huge+."""
    args = {
        "embed_dim": 1280,
        "depth": 32,
        "heads": 20,
        "mlp_ratio": 4.0,
        "ffn": "swiglu",
    }
    args.update(kwargs)
    return DINOv3Backbone(**args)
