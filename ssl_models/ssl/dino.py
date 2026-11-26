"""DINOv3 module: multi-modal, multi-temporal ViT for DINOv3-style pretraining.

Tokens, fusion modes, encodings and encoder stacks are the shared ones of
`ssl_models.ssl.base.BaseSSL`. Each sequence has registers, and one CLS token per
(band group, date) of its modalities. The image-level embedding is the attentive
pooling of the CLS tokens of the kept (not dropped) modalities / dates.

As for MAE, the module samples its masks (iBOT masks, dropped modalities /
dates) and applies its pretraining heads (DINO on the image-level embeddings,
iBOT on the masked patch tokens): `forward` returns the logits of the losses.
"""

from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.conf.mask import DINOv3MaskConfig
from ssl_models.layers.head import AttentiveReduce, DINOHead
from ssl_models.layers.vit import Transformer
from ssl_models.ssl.base import BaseSSL
from ssl_models.train.augment import local_view_size


class DINOv3(BaseSSL):
    """DINOv3: multi-modal ViT with CLS and register tokens (RoPE by default).

    The student samples the masks of its global views, and returns them for the
    teacher, which encodes the unmasked global views and applies the iBOT head
    at the same positions. The views are encoded one by one, by the calls of
    the Lightning module.
    """

    def __init__(  # noqa: PLR0913
        self,
        dataset: DatasetConfig,
        mask: DINOv3MaskConfig,
        fusion_mode: Literal["shared", "monotemp", "mod", "group"],
        inter_depth: int,
        embed_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ffn: Literal["mlp", "swiglu"] = "mlp",
        norm: Literal["layernorm", "rmsnorm"] = "layernorm",
        num_registers: int = 4,
        layerscale_init: float | None = 1e-5,
        drop_path: float = 0.3,
        patch_norm: bool = False,
        inter_norm: bool = False,
        type_head: Literal["linear", "attentive"] = "attentive",
        pos_enc: Literal["abs", "rope"] = "rope",
        rope_base: float = 100.0,
        rope_shift: float | None = None,
        rope_jitter: float | None = None,
        rope_rescale: float | None = None,
        local_ratio: float = 112 / 256,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        dino_prototypes: int = 65536,
        ibot_prototypes: int = 65536,
        head_hidden_dim: int = 2048,
        head_bottleneck_dim: int = 256,
        head_layers: int = 3,
    ) -> None:
        if fusion_mode == "msgfm":
            msg = "Fusion mode msgfm not supported by DINO."
            raise ValueError(msg)
        # encoder stacks, with a uniform drop path rate over all blocks (as in
        # DINOv3)
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
        super().__init__(
            dataset=dataset,
            fusion_mode=fusion_mode,
            embed_dim=embed_dim,
            decoder_dim=None,
            patch_size={
                name_mod: mod.patch_size.dino
                for name_mod, mod in dataset.inputs.items()
            },
            transformer=transformer,
            depth=depth,
            inter_depth=inter_depth,
            inter_norm=inter_norm,
            type_head=type_head,
            patch_norm=patch_norm,
            pos_enc=pos_enc,
            fac_date_enc=fac_date_enc,
            date_dim=date_dim,
            use_cls=True,
            num_registers=num_registers,
            enc_mask_token=True,
            rope_head_dim=embed_dim // heads,
            rope_base=rope_base,
            rope_shift=rope_shift,
            rope_jitter=rope_jitter,
            rope_rescale=rope_rescale,
            view_drop_mod=mask.view_drop_mod,
            view_drop_dates=mask.view_drop_dates,
            grid_size_local={
                name_mod: (
                    mod.local_size
                    or local_view_size(mod.image_size, mod.patch_size.dino, local_ratio)
                )
                // mod.patch_size.dino
                for name_mod, mod in dataset.inputs.items()
            },
        )

        # iBOT masking parameters
        self.ibot_mask_prob = mask.ibot_mask_prob
        self.ibot_mask_ratio = mask.ibot_mask_ratio
        self.ibot_blob_grid = mask.ibot_blob_grid
        self.ibot_mask_shared = mask.ibot_mask_shared

        # image-level embedding, attentive pooling of the CLS tokens
        self.cls_reduce = AttentiveReduce(embed_dim)

        # DINO / iBOT pretraining heads
        head = partial(
            DINOHead,
            in_dim=embed_dim,
            hidden_dim=head_hidden_dim,
            bottleneck_dim=head_bottleneck_dim,
            num_layers=head_layers,
        )
        self.dino_head = head(num_prototypes=dino_prototypes)
        self.ibot_head = head(num_prototypes=ibot_prototypes)

    @staticmethod
    def select_view(
        batch: dict[str, Tensor],
        view: Literal["global", "local"],
    ) -> dict[str, Tensor]:
        """Entries of the view: the batch itself for global views (at its keys),
        its "{key}_local" entries renamed "{key}" for local views."""
        if view == "local":
            batch = {
                k.removesuffix("_local"): v
                for k, v in batch.items()
                if k.endswith("_local")
            }
        return batch

    def mask(
        self,
        x: dict[str, Tensor],
        mask_rec: dict[str, Tensor] | None,
        keep: dict[str, Tensor] | None,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        view: Literal["global", "local"],
        teacher: bool,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor] | None, dict[str, Tensor] | None]:
        """Replace sampled iBOT-masked tokens by the mask token, for the global
        views of the student in pretraining.

        Returns the tokens, the iBOT masks {group: [B, L, 1]} bool, grouped as the
        tokens (True = masked and kept): those sampled for the student, else
        `mask_rec` as is (the masks of the student for the teacher, else None), and
        the attention mask {group: [B, L]} (from the kept tokens `keep`).
        """
        attn_mask = (
            {name_group: keep[name_group][..., 0] for name_group in keep}
            if keep is not None
            else None
        )
        if ssl_phase != "pretrain" or teacher or view != "global":
            return x, mask_rec, attn_mask

        mask_rec = self.ibot_masks(x)
        x = {
            name_group: torch.where(
                mask_rec[name_group],
                self.enc_mask_token.to(x[name_group].dtype),
                x[name_group],
            )
            for name_group in x
        }
        if keep is not None:
            mask_rec = {name: mask_rec[name] & keep[name] for name in mask_rec}
        return x, mask_rec, attn_mask

    def ibot_masks(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Sample the iBOT masks of the tokens x, as blobs aligned in space across
        modalities.

        A fraction `ibot_mask_prob` of samples is masked, with a ratio drawn
        uniformly in `ibot_mask_ratio` (the same ratio for all modalities and dates
        of a sample). Masks are obtained by thresholding smooth noise: a coarse
        `ibot_blob_grid` x `ibot_blob_grid` noise, bilinearly upsampled to each token
        grid. Since all grids cover the same extent, the noise shared by all
        sequences of a sample gives geographically aligned masks across modalities
        and dates. `ibot_mask_shared` in [0, 1] mixes it with independent noise per
        (band group, date).

        Returns:
            {group: [B, L, 1]} bool masks, grouped as the tokens, True = masked.

        """
        x = self.ungroup(x)
        x_ref = next(iter(x.values()))
        B, device = x_ref.shape[0], x_ref.device
        blob_grid, shared = self.ibot_blob_grid, self.ibot_mask_shared

        active = torch.rand(B, device=device) < self.ibot_mask_prob
        ratio = torch.empty(B, device=device).uniform_(*self.ibot_mask_ratio)
        ratio = ratio * active
        shared_noise = torch.rand(B, 1, blob_grid, blob_grid, device=device)

        masks = {}
        for name_mod in x:
            _, D, L, _ = x[name_mod].shape  # noqa: N806
            grid = (round(L**0.5),) * 2

            noise_shared = F.interpolate(shared_noise, grid, mode="bilinear")
            noise_indep = torch.rand(
                B * D,
                1,
                blob_grid,
                blob_grid,
                device=device,
            )
            noise_indep = F.interpolate(noise_indep, grid, mode="bilinear")
            noise = shared * noise_shared.reshape(B, 1, L) + (
                1 - shared
            ) * noise_indep.reshape(B, D, L)
            noise = noise + 1e-3 * torch.rand_like(noise)  # break ties

            num_masked = (ratio * L).round().long()
            ranks = noise.argsort(dim=-1, descending=True).argsort(dim=-1)
            masks[name_mod] = (ranks < num_masked[:, None, None])[..., None]
        return self.group(masks)

    def reduce_cls(
        self,
        x_cls: dict[str, Tensor],
        keep: dict[str, Tensor] | None,
    ) -> Tensor:
        """Image-level embeddings [B, C]: attentive pooling of the CLS tokens of the
        kept (band group, date) (see `keep_view`, all of them if None)."""
        x_cls = self.ungroup(x_cls)  # [B, D, 1, C]

        mask = None
        if keep is not None:
            keep = self.ungroup(keep)  # [B, D, L, 1]

            mask = [keep[name_mod].any(dim=(2, 3)) for name_mod in keep]
            mask = torch.cat(mask, dim=1)

        x_cls = torch.cat([x_cls[name_mod].flatten(1, 2) for name_mod in x_cls], dim=1)
        return self.cls_reduce(x_cls, mask)

    def compute_ibot(
        self,
        x: dict[str, Tensor],
        mask_rec: dict[str, Tensor] | None,
        view: Literal["global", "local"] = "global",
    ) -> tuple[Tensor | None, Tensor | None]:
        """iBOT logits of the masked tokens, and their weights in the iBOT loss.

        The tokens are taken by image, over all modalities, and weighted by
        1 / number of masked tokens of their image: each image has the same total
        weight, whatever its number of masked tokens.

        Args:
            x: encoded patch tokens (grouped).
            mask_rec: iBOT masks, grouped as the tokens (see `mask`), None for
                local views.
            view: "global" or "local", the views encoded (iBOT on global views
                only).

        Returns:
            [M, K] iBOT logits of the M masked tokens, by image, and their [M]
            weights (None, None for local views).

        """
        if view != "global":
            return None, None

        x, mask_rec = self.ungroup(x), self.ungroup(mask_rec)

        name_mods = list(x)
        x = torch.cat([x[name_mod].flatten(1, 2) for name_mod in name_mods], dim=1)
        mask_rec = torch.cat(
            [mask_rec[name_mod].flatten(1, 2) for name_mod in name_mods], dim=1
        )[..., 0]
        num_masked = mask_rec.sum(dim=1)
        ibot_weights = (1 / num_masked.clamp(min=1)).repeat_interleave(num_masked)
        ibot = self.ibot_head(x[mask_rec])

        return ibot, ibot_weights

    def forward(
        self,
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        view: Literal["global", "local"] = "global",
        teacher: bool = False,
        mask_rec: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor] | None]:
        """Forward pass for DINOv3 module, on one view.

        In pretraining, the student drops modalities / dates, and masks its global
        views with iBOT masks; the teacher sees unmasked global views, and applies
        the iBOT head at the masked positions of the student (`mask_rec`).

        Args:
            batch: global views at the keys (modalities [B, D, C, H, W],
                "{modality}_dates" [B, D, 3], "ref_date" [B, 1, 3]), local views
                (if any) likewise at "{key}_local".
            ssl_phase: "pretrain", "probe" or "finetune".
            view: "global" or "local", the views encoded.
            teacher: whether the module is the teacher.
            mask_rec: for the teacher, the iBOT masks of the student (grouped, see
                `mask`).

        Returns:
            In probing / finetuning, "logits": the logits of the heads. In
            pretraining:
            "dino": [B, K] DINO logits,
            "cls": [B, C] image-level embeddings,
            "ibot": [M, K] iBOT logits of the masked kept tokens, by image
                (global views),
            "ibot_weights": [M] weights of these tokens in the iBOT loss (global
                views),
            "mask_rec": {group: [B, L, 1]} bool iBOT masks, grouped as the tokens,
                True = masked kept token (global views).

        """
        batch = self.select_view(batch, view=view)

        x, x_cls = self.embed(batch)
        keep = self.keep_view(x, ssl_phase=ssl_phase, teacher=teacher)
        x, mask_rec, attn_mask = self.mask(
            x, mask_rec, keep, ssl_phase=ssl_phase, view=view, teacher=teacher
        )
        x, x_cls = self.add_enc_encodings(x, x_cls, batch)
        rope, rope_cls = self.add_enc_rope(x, x_cls)

        x_enc, x_cls = self.encode(x, x_cls, rope, rope_cls, attn_mask)

        if ssl_phase == "pretrain":
            cls = self.reduce_cls(x_cls, keep)
            dino = self.dino_head(cls)
            ibot, ibot_weights = self.compute_ibot(x_enc, mask_rec, view=view)

            return {
                "dino": dino,
                "ibot": ibot,
                "ibot_weights": ibot_weights,
                "cls": cls,
                "mask_rec": mask_rec,
            }

        elif ssl_phase in ("probe", "finetune"):
            logits = self.compute_logits(x_enc, x_cls, ssl_phase=ssl_phase)

            return {
                "logits": logits,
            }


def dinov3_tiny(**kwargs) -> DINOv3:  # noqa: ANN003
    """Construct DINOv3 backbone tiny."""
    args = {
        "embed_dim": 192,
        "depth": 12,
        "heads": 3,
        "mlp_ratio": 4.0,
        "ffn": "mlp",
    }
    args.update(kwargs)
    return DINOv3(**args)


def dinov3_small(**kwargs) -> DINOv3:  # noqa: ANN003
    """Construct DINOv3 backbone small."""
    args = {
        "embed_dim": 384,
        "depth": 12,
        "heads": 6,
        "mlp_ratio": 4.0,
        "ffn": "mlp",
    }
    args.update(kwargs)
    return DINOv3(**args)


def dinov3_base(**kwargs) -> DINOv3:  # noqa: ANN003
    """Construct DINOv3 backbone base."""
    args = {
        "embed_dim": 768,
        "depth": 12,
        "heads": 12,
        "mlp_ratio": 4.0,
        "ffn": "mlp",
    }
    args.update(kwargs)
    return DINOv3(**args)


def dinov3_large(**kwargs) -> DINOv3:  # noqa: ANN003
    """Construct DINOv3 backbone large."""
    args = {
        "embed_dim": 1024,
        "depth": 24,
        "heads": 16,
        "mlp_ratio": 4.0,
        "ffn": "mlp",
    }
    args.update(kwargs)
    return DINOv3(**args)


def dinov3_huge_plus(**kwargs) -> DINOv3:  # noqa: ANN003
    """Construct DINOv3 backbone huge+."""
    args = {
        "embed_dim": 1280,
        "depth": 32,
        "heads": 20,
        "mlp_ratio": 4.0,
        "ffn": "swiglu",
    }
    args.update(kwargs)
    return DINOv3(**args)
