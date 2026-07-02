"""DinoV3 baseline module.

Implementation inspired by `dinov2.py` but using `torch.hub.load` as weight
source, with optional loading of a local `.pth` checkpoint.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torchvision.transforms import v2

from ssl_models.baselines.base import BaseModule
from ssl_models.conf.datasets import DatasetsConfig

RGB_BANDS = 3


class Dinov3Baseline(BaseModule):
    """Baseline model based on DinoV3-like hub checkpoints."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        backbone_type: Literal["web_base", "web_large", "sat_large"],
        freeze: bool = False,
        checkpoint_path: str | None = None,
        repo_dir: str = "dinov3",
        hub_model: str | None = None,
        type_head: Literal["linear", "attentive", "m2f"] = "attentive",
        interpolate: Literal["nearest", "bilinear", "bicubic"] = "nearest",
        fusion_mode: Literal["shared", "monotemp"] = "shared",
        add_date_enc: bool = True,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        keep_norm: bool = True,
        resize: bool = True,
        m2f_hidden_dim: int = 256,
        interaction_indexes: list[int] | None = None,
        adapter_with_cp: bool = False,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Create the DinoV3 encoder model using torch.hub.

        Parameters
        ----------
        datasets: DatasetsConfig
            The dataset config used in the probing/finetuning phase.
        backbone_type: str
            Backbone type to use. To choose in "web_base", "web_large", "sat_large".
        freeze: bool
            Whether to freeze the backbone.
        checkpoint_path: str
            Path to optional local checkpoint to load after hub model.
        weight_source: str
            Either "hub" (default) or "random".
        hub_repo: str
            Hub repository name (default "facebookresearch/dinov2").
        hub_model: str
            Optional explicit hub model name. If not provided, inferred from
            backbone_size.
        hub_pretrained: bool
            Whether to load pretrained from hub.
        type_head: str
            Segmentation head to use. Either "linear" or "attentive" or "m2f".
        interpolate: str
            Interpolation used in the image resizing before patchification.
        fusion_mode: str
            Fusion strategy. Either "shared" or "monotemp".
        add_date_enc: bool
            Whether to add date encodings to the embeddings.
        fac_date_enc: float
            Factor used to compute the date encodings.
        date_dim: int
            Dimension of the date embeddings.
        keep_norm: bool
            Whether to keep the final layernorm layer.
        adapter_with_cp: bool
            Whether to enable activation checkpointing inside DINOv3_Adapter.
            Disabled by default because reentrant checkpointing can trigger
            DDP "marked ready twice" errors when adapter blocks are trainable.
        kwargs:
            Arguments to pass to `BaseModule` constructor.
        """
        self.type_head = type_head
        self.interpolate = interpolate
        self.fusion_mode = fusion_mode
        self.checkpoint_path = checkpoint_path
        self.backbone_type = backbone_type
        self.repo_dir = repo_dir
        self.hub_model = hub_model
        self.keep_norm = keep_norm
        self.add_date_enc = add_date_enc
        self.fac_date_enc = fac_date_enc
        self.date_dim = date_dim
        self.resize = resize
        self.m2f_hidden_dim = m2f_hidden_dim
        self.interaction_indexes = interaction_indexes
        self.adapter_with_cp = adapter_with_cp

        self.patch_size = 16

        self.embed_dim_by_type = {
            "web_base": 768,
            "web_large": 1024,
            "sat_large": 1024,
        }

        self.depth_by_type = {
            "web_base": 12,
            "web_large": 24,
            "sat_large": 24,
        }

        self.embed_dim = self.embed_dim_by_type[self.backbone_type]
        self.depth = self.depth_by_type[self.backbone_type]

        if hasattr(datasets, "dataset"):
            datasets_baseline = datasets
        elif hasattr(datasets, "datasets"):
            if len(datasets.datasets) != 1:
                msg = (
                    "Dinov3Baseline expects a single dataset config. "
                    f"Got {len(datasets.datasets)} datasets: "
                    f"{list(datasets.datasets.keys())}."
                )
                raise ValueError(msg)
            single_dataset = next(iter(datasets.datasets.values()))
            datasets_baseline = SimpleNamespace(dataset=single_dataset)
        else:
            msg = "Invalid datasets config: expected 'dataset' or 'datasets' attribute."
            raise ValueError(msg)

        super().__init__(
            datasets_baseline,
            self.fusion_mode,
            self.patch_size,
            self.embed_dim,
            self.type_head,
            self.interpolate,
            self.add_date_enc,
            self.fac_date_enc,
            self.date_dim,
            self.keep_norm,
            **kwargs,
        )

        self._build_backbone(backbone_type)

        if freeze:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

        self.rgb = {
            name_mod: self.len_bands[name_mod] <= RGB_BANDS
            for name_mod in self.len_bands
        }

        if self.resize:
            self.transform = self._make_transform()
        else:
            self.transform = None

    def _build_backbone(
        self,
        backbone_type: Literal["web_base", "web_large", "sat_large"],
    ) -> None:
        repo_path = self._resolve_repo_path()
        self._resolved_repo_path = repo_path

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        # The compiled MultiScaleDeformableAttention C extension (.so) lives
        # inside the ops sub-package and must be on sys.path to be importable.
        ops_path = (
            repo_path / "dinov3" / "eval" / "segmentation" / "models" / "utils" / "ops"
        )
        if str(ops_path) not in sys.path:
            sys.path.insert(0, str(ops_path))
            print("Added MultiScaleDeformableAttention ops path to sys.path:", ops_path)

        if not any(ops_path.glob("MultiScaleDeformableAttention*.so")):
            msg = (
                "MultiScaleDeformableAttention extension not found in "
                f"{ops_path}. Compile it before training Mask2Former."
            )
            raise RuntimeError(msg)

        try:
            import MultiScaleDeformableAttention  # noqa: PLC0415,F401
            print("Successfully imported MultiScaleDeformableAttention.")
        except Exception as exc:  # noqa: BLE001
            msg = (
                "Found MultiScaleDeformableAttention .so but failed to import it. "
                 f"ops_path={ops_path}. Cause: {type(exc).__name__}: {exc}"
            )
            raise RuntimeError(msg) from exc

        backbone = torch.hub.load(
            str(repo_path),
            self._resolve_hub_model_name(backbone_type),
            source="local",
            weights=self.checkpoint_path,
        )
        if self.type_head == "m2f":
            self._build_m2f_decoder(backbone)
        else:
            self.model = backbone

    def _resolve_repo_path(self) -> Path:
        """Resolve DinoV3 local repo path in a way that is robust to DDP cwd."""
        # Expected repo is the folder containing hubconf.py and the dinov3 package.
        base_dir = Path(__file__).resolve().parents[2]
        configured = Path(self.repo_dir).expanduser()

        candidates = [
            configured,
            Path.cwd() / configured,
            base_dir / configured,
            base_dir.parent / "DinoV3" / "dinov3",
        ]

        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if (resolved / "hubconf.py").exists() and (resolved / "dinov3").is_dir():
                return resolved

        msg = (
            "Could not resolve DinoV3 repo path. "
            f"repo_dir={self.repo_dir!r}. "
            "Expected a folder containing hubconf.py and a dinov3/ package."
        )
        raise FileNotFoundError(msg)

    def _resolve_interaction_indexes(self, backbone: torch.nn.Module) -> list[int]:
        """Return block indices at which the adapter cross-attends with backbone."""
        if self.interaction_indexes is not None:
            return self.interaction_indexes
        n_blocks = getattr(backbone, "n_blocks", 12)
        if n_blocks == 24:  # ViT-L — indices from the paper
            return [4, 11, 17, 23]
        return [i * (n_blocks // 4) - 1 for i in range(1, 5)]

    def _build_m2f_decoder(self, backbone: torch.nn.Module) -> None:
        """Build DINOv3_Adapter + Mask2FormerHead from the local DINOv3 repo.

        The backbone is frozen inside DINOv3_Adapter. The adapter (SPM,
        interactions) and the Mask2FormerHead remain trainable.

        Note: DINOv3_Adapter uses SyncBatchNorm; multi-GPU DDP is required.
        For single-GPU use, convert norms manually before training.
        """
        from dinov3.eval.segmentation.models.backbone.dinov3_adapter import (  # noqa: PLC0415
            DINOv3_Adapter,
        )
        from dinov3.eval.segmentation.models.heads.mask2former_head import (  # noqa: PLC0415
            Mask2FormerHead,
        )

        interaction_indexes = self._resolve_interaction_indexes(backbone)
        self.model = DINOv3_Adapter(
            backbone,
            interaction_indexes=interaction_indexes,
            # Reentrant checkpointing in the adapter conflicts with DDP when
            # interaction blocks are trainable.
            with_cp=self.adapter_with_cp,
        )

        embed_dim = backbone.embed_dim
        patch_size = backbone.patch_size

        self.m2f_target_name = next(iter(self.dataset.targets))
        num_classes = self.dataset.targets[self.m2f_target_name].num_classes

        self.m2f_head = Mask2FormerHead(
            input_shape={
                "1": [embed_dim, patch_size * 4, patch_size * 4, 4],
                "2": [embed_dim, patch_size * 2, patch_size * 2, 4],
                "3": [embed_dim, patch_size, patch_size, 4],
                "4": [embed_dim, patch_size // 2, patch_size // 2, 4],
            },
            hidden_dim=self.m2f_hidden_dim,
            num_classes=num_classes,
            ignore_value=255,
        )

    def _m2f_forward(
        self,
        batch: dict[str, Tensor],
        group_batch: dict[str, Tensor],
    ) -> tuple:
        """Forward pass using DINOv3_Adapter + Mask2FormerHead.

        The adapter produces 4-level pyramid features; Mask2Former returns
        per-query class logits and masks that are fused into dense semantic
        logits compatible with the existing cross-entropy training pipeline.

        The no-object class (last class in pred_logits) is discarded before
        the panoptic fusion. Dense predictions are returned as probabilities
        in (B, 1, C, H, W); the existing F.cross_entropy loss is applied on
        these values. For proper Mask2Former training, replace the loss with
        the Hungarian-matching loss from the original implementation.
        """
        ref_input = self.dataset.ref_input or next(iter(self.dataset.inputs))
        x = group_batch[ref_input]  # (B, C, H, W) after group op

        # SpatialPriorModule.stem is Conv2d(3, ...) — slice to RGB if needed
        if x.shape[1] > RGB_BANDS:
            x = x[:, :RGB_BANDS]

        h_in, w_in = x.shape[-2], x.shape[-1]

        # Adapter forward: backbone frozen, adapter trainable
        features = self.model(x)  # {"1": f1, "2": f2, "3": f3, "4": f4}

        # Mask2Former forward
        output = self.m2f_head(features)
        pred_logits = output["pred_logits"]  # (B, Q, num_classes+1)
        pred_masks_lowres = output["pred_masks"]  # (B, Q, H_small, W_small)

        # Upsample masks to input spatial resolution
        pred_masks = F.interpolate(
            pred_masks_lowres,
            size=(h_in, w_in),
            mode=self.interpolate,
            align_corners=False,
        )

        # Panoptic fusion → dense semantic probabilities (B, C, H, W)
        mask_cls = pred_logits.softmax(-1)[..., :-1]  # (B, Q, C) — no-obj dropped
        mask_pred = pred_masks.sigmoid()               # (B, Q, H, W)
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)

        # Reshape to (B, 1, C, H, W) for pipeline compatibility.
        # Raw M2F outputs are exposed under a dedicated key so an optional
        # Hungarian criterion can be computed in the trainer.
        return batch, None, None, None, {
            self.m2f_target_name: semseg.unsqueeze(1),
            f"{self.m2f_target_name}__m2f_raw": {
                "pred_logits": pred_logits,
                "pred_masks": pred_masks_lowres,
                "aux_outputs": output.get("aux_outputs", []),
            },
        }

    def _resolve_hub_model_name(self, backbone_type: str) -> str:
        """Resolve hub model name from backbone type or explicit hub_model param."""
        if self.hub_model is not None:
            return self.hub_model

        model_by_type = {
            "web_base": "dinov3_vitb16",
            "web_large": "dinov3_vitl16",
            "sat_large": "dinov3_vitl16",
        }
        return model_by_type[backbone_type]

    def _transfer_patch_embedding_to_more_channels(
        self,
        state_dict: dict[str, Tensor],
        num_channels: int,
        std: float,
    ) -> dict[str, Tensor]:
        weight_key = "embeddings.patch_embeddings.projection.weight"

        with torch.no_grad():
            new_state_dict = state_dict.copy()
            old_weights = new_state_dict[weight_key]
            bs, _, hs, ws = old_weights.shape

            means = torch.zeros((bs, num_channels - RGB_BANDS, hs, ws))
            stds = torch.ones((bs, num_channels - RGB_BANDS, hs, ws)) * std

            pad_tensor = torch.normal(means, stds)
            new_weights = torch.cat((old_weights, pad_tensor), dim=1)
            new_state_dict[weight_key] = new_weights

        return new_state_dict

    def freeze_backbone(
        self,
        ssl_phase: Literal["probe", "finetune"] = "finetune",
    ) -> None:
        """Freeze the backbone's parameters.

        Parameters
        ----------
        ssl_phase: str
            SSL phase (probe, finetune). Only probing and finetuning are needed.
        """
        if "backbone" in self.model.__dict__:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = False

    def unfreeze_backbone(
        self,
        ssl_phase: Literal["probe", "finetune"] = "finetune",
    ) -> None:
        """Unfreeze the backbone's parameters.

        Parameters
        ----------
        ssl_phase: str
            SSL phase (probe, finetune). Only probing and finetuning are needed.
        """
        if "backbone" in self.model.__dict__:
            for param in self.model.backbone.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def _make_transform(self, resize_size: int = 256):
        to_tensor = v2.ToImage()
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        to_float = v2.ToDtype(torch.float32, scale=True)

        if "sat" in self.backbone_type:
            normalize = v2.Normalize(
                mean=(0.430, 0.411, 0.296),
                std=(0.213, 0.156, 0.143),
            )
        else:
            normalize = v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )

        return v2.Compose([to_tensor, resize, to_float, normalize])

    def grouped_parameters(
        self,
        **kwargs,  # noqa: ANN003
    ) -> list[dict]:
        """Skip implem."""

    def forward(
        self,
        batch: dict[str, Tensor],
        name_dataset: str,
        ssl_phase: Literal["probe", "finetune"],
    ) -> tuple[
        dict[str, Tensor] | None,
        None,
        None,
        dict[str, Tensor] | None,
    ]:
        """DinoV3 forward pass.

        Parameters
        ----------
        batch: dict
            Dictionary containing the data.
        ssl_phase: str
            SSL phase (probe, finetune).
        """
        if self.transform is not None:
            batch = self.resize_and_rescale(batch)

        group_batch = self.group(batch)

        if self.type_head == "m2f":
            return self._m2f_forward(batch, group_batch)

        x_enc, dates = {}, {}
        for name_mod in self.dataset.inputs:
            x_mod = self.model(
                group_batch[name_mod],
                is_training=True
            )["x_norm_patchtokens"]

            x_enc[name_mod] = x_mod
            dates[name_mod] = batch[f"{name_mod}_dates"]

        if self.add_date_enc:
            ref_date = batch["ref_date"]
            x_enc = self._add_date_encodings(x_enc, dates, ref_date)

        logits = self.compute_logits(x_enc)
        return batch, None, None, x_enc, logits

