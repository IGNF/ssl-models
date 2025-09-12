"""DINOv2 PyTorch Module.

The DINO-v2 framework: https://arxiv.org/abs/2304.07193
"""

from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import LayerNorm, ModuleDict
from torch.nn import functional as F  # noqa: N812
from transformers import AutoConfig, Dinov2Config
from transformers.models.dinov2.modeling_dinov2 import (Dinov2Embeddings,
                                                        Dinov2Encoder)

from conf.datasets import DatasetsConfig
from ssl_models.baselines.utils import (filter_dict,
                                        get_imagenat_dinov2_state_dict)

RGB_BANDS = 3


class DINOv2(nn.Module):
    """DINOv2 module."""

    def __init__(
        self,
        dataset: DatasetsConfig,
        backbone_size: Literal["small", "base", "large", "huge"] = "base",
        keep_norm: bool = True,
    ) -> None:
        """Create the DINO-v2 model."""
        super().__init__()
        self.dataset = dataset
        self.keep_norm = keep_norm
        self.config_dict = AutoConfig.from_pretrained(
            f"facebook/dinov2-{backbone_size}",
        ).to_dict()

        self._build_backbone(backbone_size)
        self.freeze_backbone()

    def _build_patch_embed(
        self,
        state_dict: dict[str, Tensor],
    ) -> ModuleDict:
        model_config = Dinov2Config(**self.config_dict)
        patch_embed = Dinov2Embeddings(model_config)
        patch_embed.load_state_dict(filter_dict(state_dict, "embeddings"), strict=False)

        return patch_embed

    def _build_encoder(
        self,
        state_dict: dict[str, Tensor],
    ) -> ModuleDict:
        model_config = Dinov2Config(**self.config_dict)
        encoder = Dinov2Encoder(model_config)
        encoder.load_state_dict(filter_dict(state_dict, "encoder"), strict=False)

        return encoder

    def _build_layer_norm(
        self,
        state_dict: dict[str, Tensor],
    ) -> LayerNorm:
        layernorm = LayerNorm(
            self.config_dict["hidden_size"],
            eps=self.config_dict["layer_norm_eps"],
        )
        layernorm.load_state_dict(filter_dict(state_dict, "layernorm"), strict=False)

        return layernorm

    def _build_backbone(
        self,
        backbone_size: Literal["small", "base", "large", "huge"],
    ) -> None:
        # state dict
        state_dict = get_imagenat_dinov2_state_dict(backbone_size)
        # patch embedding layer
        self.patch_embed = self._build_patch_embed(state_dict=state_dict)
        # encoder
        self.encoder = self._build_encoder(state_dict=state_dict)
        # layer norm
        if self.keep_norm:
            self.layernorm = self._build_layer_norm(state_dict=state_dict)

    def freeze_backbone(
        self,
    ) -> None:
        """Freeze the backbone's parameters."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        for param in self.encoder.parameters():
            param.requires_grad = False

        if self.keep_norm:
            for param in self.layernorm.parameters():
                param.requires_grad = False

    def forward(
        self,
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Forward pass."""
        x_enc = {}
        for name_mod in self.dataset.inputs:
            x_enc[name_mod] = []
            for idx in range(batch[name_mod].shape[1]):
                x_mod = batch[name_mod][:, idx, :RGB_BANDS]
                x_mod = F.interpolate(x_mod, size=(224,) * 2, mode="nearest")
                x_mod = self.patch_embed(x_mod)
                x_mod = self.encoder(x_mod)["last_hidden_state"]
                if self.keep_norm:
                    x_mod = self.layernorm(x_mod)
                x_mod = x_mod[:, 0, :]
                x_enc[name_mod].append(x_mod)
            x_enc[name_mod] = torch.cat(x_enc[name_mod], dim=-1)
        return x_enc
