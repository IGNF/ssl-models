"""DinoV3 baseline module.

Implementation inspired by `dinov2.py` but using `torch.hub.load` as weight
source, with optional loading of a local `.pth` checkpoint.
"""

from types import SimpleNamespace
from typing import Literal

import torch
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
        type_head: Literal["linear", "attentive"] = "attentive",
        interpolate: Literal["nearest", "bilinear", "bicubic"] = "nearest",
        fusion_mode: Literal["shared", "monotemp"] = "shared",
        add_date_enc: bool = True,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        keep_norm: bool = True,
        resize: bool = True,
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
            Segmentation head to use. Either "linear" or "attentive".
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
        self.model = torch.hub.load(
            self.repo_dir,
            self._resolve_hub_model_name(backbone_type),
            source="local",
            weights=self.checkpoint_path
        )

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
        for param in self.model.parameters():
            param.requires_grad = False

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

