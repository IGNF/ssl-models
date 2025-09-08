from typing import Literal

import torch
from terratorch import BACKBONE_REGISTRY
from torch import Tensor
from torch.nn import Module

from conf.datasets import DatasetsConfig
from ssl_models.baselines.base import BaseModule


class PrithviBaseline(BaseModule):
    """Baseline model based on Prithvi."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        unpool_dim: int,
        freeze: bool = False,
        version: str = "v2",
        backbone_size: str = "large",
        type_head: Literal["linear", "attentive"] = "linear",
        multimodal: Literal["mod"] = "mod",
        pretrained_path: str | None = None,
        add_date_enc: bool = False,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        keep_norm: bool = True,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Create the Prithvi encoder model.

        Parameters
        ----------
        datasets: DatasetsConfig
            The dataset config used in the probing/finetuning phase.
        backbone_size: str
            Defines the backbone to use. To choose in "small", "base", "large", "huge".
        unpool_dim: int
            Parameter given to the pixelify head.
        freeze: bool
            To freeze or not to freeze the DinoV2 backbone.
        type_head: str
           Segmentation head to use. Either "linear" (default) or "attentive".
        multimodal: str
           Multimodal strategy. Either "monotemp" (default) or "croma-intergroup".
        pretrained_path: str
           Path to the location of the pretrained weights.
        add_date_enc: bool
           Whether to add the date encodings to the embeddings.
        fac_date_enc: float
           Factor used to compute the date encodings.
        date_dim: int
           Dimension of the date embeddings.
        keep_norm: bool
           Choose to keep the final layernorm layer.
        kwargs:
           Arguments to pass to `BaseModel` constructor.

        """
        self.datasets = datasets
        self.version = version
        self.backbone_size = backbone_size
        self.freeze = freeze
        self.type_head = type_head
        self.multimodal = multimodal

        self.unpool_dim = unpool_dim
        self.pretrained_path = pretrained_path
        self.add_date_enc = add_date_enc
        self.fac_date_enc = fac_date_enc
        self.date_dim = date_dim
        self.keep_norm = keep_norm

        if self.backbone_size == "base":
            self.encoder_dim = 768
            self.encoder_depth = 12
            self.num_heads = 16
            self.patch_size = 16
        else:
            # large by default
            self.encoder_dim = 1024
            self.encoder_depth = 24
            self.num_heads = 16
            self.patch_size = 16

        super().__init__(
            datasets,
            self.patch_size,
            self.encoder_dim,
            self.unpool_dim,
            self.type_head,
            self.multimodal,
            self.add_date_enc,
            self.fac_date_enc,
            self.date_dim,
            self.keep_norm,
            **kwargs,
        )

        if not self._check_model_version_and_size():
            raise ValueError(
                f"Invalid model version {self.version} and size {self.backbone_size}. "
                "Supported versions are: 'v1' for base and 'v2' for large."
            )

        self._build_backbone()

        if self.freeze:
            self.freeze_backbone()

    def _build_backbone(self) -> Module:
        model_dict = {
            ("base","v1", False): "terratorch_prithvi_eo_v1_100",
            ("large","v2", False): "terratorch_prithvi_eo_v2_300",
            ("large","v2", True): "terratorch_prithvi_eo_v2_300_tl",
        }

        self.encoder =  BACKBONE_REGISTRY.build(
                            model_dict[self.backbone_size, self.version, self.add_date_enc],
                            pretrained=True,
                            ckpt_path=self.pretrained_path,
                            num_frames=self.num_dates["s2"],
                        ).cuda()

    def _check_model_version_and_size(self) -> bool:
        model_assoc = [
            ("large", "v2", False),
            ("large", "v2", True),
            ("base", "v1", False)
        ]

        return (self.backbone_size, self.version, self.add_date_enc) in model_assoc

    def freeze_backbone(self) -> None:
        """Freeze the backbone's parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def grouped_parameters(
        self,
        base_lr: float,
        rate_decay: float = 0.75,
    ) -> list[dict]:
        """Return the grouped parameters dictionnary with layerwise learning rate.

        Parameters
        ----------
        base_lr: float
            Base learning rate.
        rate_decay: float
            Input's key.

        Returns
        -------
        List: A list of dictionnaries, containing "params" and "lr" keys, in which
        parameters are grouped.

        """
        pass

    def format_dates(self, dates: Tensor) -> Tensor:
        year = dates[:, :, 0, 0, 0]
        doy = dates[:, :, 1, 0, 0] / 365.25

        return torch.stack([year, doy], dim=-1)

    def _add_date_encodings(self, x: Tensor, temporal_coords: Tensor) -> Tensor:
        """Add date encodings to the embeddings."""
        if self.encoder.temporal_encoding and temporal_coords is not None:
            num_tokens_per_frame = x.shape[1] // self.encoder.num_frames

            temporal_encoding = self.encoder.temporal_embed_enc(
                temporal_coords, num_tokens_per_frame
            )

            x = x + temporal_encoding

        return x

    def forward(
        self,
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Prithvi v1/v2 forward pass.

        Parameters
        ----------
        batch: dict
            Dictionary containing the data.
        ssl_phase: str
            SSL phase (pretrain, probe, finetune). Only probing and finetuning
            are needed for baselines.
        stage: str
            Learning stage: train, val, test.

        Returns
        -------
        tuple: Prithvi's outputs.

        """
        s2_input = batch["s2"].transpose(1,2) # (B, D, C, H, W) -> (B, C, D, H, W)

        if self.add_date_enc:
            s2_dates = batch["s2_dates"]
            formatted_dates = self.format_dates(s2_dates)

            features_prithvi = self.encoder(x=s2_input,
                                            temporal_coords=formatted_dates)
        else:
            features_prithvi = self.encoder(x=s2_input)

        output_prithvi = features_prithvi[-1][:, 1:, :] # remove cls token

        x = {}

        x["s2"] = output_prithvi

        loss_pred, log_input, log_pred, log_target = self.compute_loss_pred(
            x,
            batch,
            ssl_phase,
            stage,
        )
        self.metrics[f"loss_pred_{stage}"].update(loss_pred)
        return None, log_input, log_pred, log_target, loss_pred
