from torchgeo.models import DOFABase16_Weights, DOFALarge16_Weights
import torch
from abc import ABC, abstractmethod
from torch.nn import Identity, Module

class DOFASegmentor(Module, ABC):
    def __init__(
        self,
        image_size: int,
        in_channels: int,
        backbone_size: str,
        num_classes: int,
        freeze: bool = False,
        seg_head: str = 'naive',
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.image_size = image_size
        self.in_channels = in_channels
        self.backbone_size = backbone_size
        self.num_classes = num_classes
        self.freeze = freeze
        
        self.patch_size = 16

        self.args_by_size = {#'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
                             'base':  {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
                             'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
                             #'huge':  {'embed_dim': 1280, 'depth': 32, 'num_heads': 16}
                            }
        self.weights_by_size = {'base': DOFABase16_Weights.DOFA_MAE,
                                'large': DOFALarge16_Weights.DOFA_MAE}

        assert backbone_size in self.args_by_size.keys(), 'backbone_size should be in {}'.format(self.args_by_size.keys())

        self.dofa = DOFA(img_size = self.image_size, patch_size = self.patch_size, global_pool=False, task='segmentation', **self.args_by_size[self.backbone_size])
        self.dofa.load_state_dict(self.weights_by_size[self.backbone_size].get_state_dict(progress=True), strict=False)
        self.dim = self.args_by_size[backbone_size]['embed_dim']
        self.dofa.head = Identity()
        self.wavelengths = [0.64, 0.56, 0.48, 0.81][:self.in_channels]
        
        self.segmentation_head = self.get_segmentation_head(seg_head)
        
        self.aux_config = {}

    def get_segmentation_head(self, seg_head):
        """
        Load the chosen segmentation head.

        Parameters
        ----------
        seg_head: str
            To chose among 'naive', 'progressive', 'upernet'.

        Returns
        -------
        tuple: The segmentation head (nn.Module) instance.
        
        """

        if seg_head == 'naive':
            return NaiveHead(self.patch_size, self.dim, self.image_size, self.num_classes)
        elif seg_head == 'upernet':
            assert self.backbone_size == 'large', "Upernet segmentation head needs ViT large backbone."
            return UperNetSegmentationHead(self.patch_size, self.dim, self.image_size, self.num_classes, backbone_type='dinov2_large')
        else:
            raise NotImplementedError
    
    def freeze_backbone(self):
        """
        Freeze the backbone's parameters (self.dofa).

        Parameters
        ----------
        Returns
        -------
        """
        for param in self.dofa.parameters():
            param.requires_grad = False
    
    def grouped_parameters(self, base_lr, rate_decay=0.75):
        """
        Return the grouped parameters dictionnary with layerwise learning rate.
        
        Parameters
        ----------
        base_lr: float
            Base learning rate.
        rate_decay: float
            Input's key.

        Returns
        -------
        List: A list of dictionnaries, containing 'params' and 'lr' keys, in which parameters are grouped.

        """
        num_layers = len(self.dofa.blocks)
        lr_map = dict([('dofa.blocks.{}'.format(i), base_lr * rate_decay ** (num_layers - i)) for i in range(num_layers)])
        
        lr_map['dofa.norm'] = base_lr * rate_decay ** (num_layers+1)
        lr_map['dofa.patch_embed'] = base_lr * rate_decay ** (num_layers+1)
 
        lr_map['segmentation_head'] = base_lr
        
        params_dict = {}
        params_dict['dofa.norm'] = self.dofa.norm.parameters()
        params_dict['dofa.patch_embed'] = self.dofa.patch_embed.parameters()
        params_dict['segmentation_head'] = self.segmentation_head.parameters()

        for i in range(num_layers):
            params_dict['dofa.blocks.{}'.format(i)] = self.dofa.blocks[i].parameters()
            
        print('Layerwise learning rates:')
        grouped_parameters = []
        for k, lr in lr_map.items():
            grouped_parameters.append({'params': params_dict[k], 'lr': lr, 'name': k})
            print('Layer "{}": {}'.format(k, lr))

        return grouped_parameters

    def forward(self, batch, key):
        """
        DOFA forward pass.

        Parameters
        ----------
        batch: dict
            Dictionary containing the data.
        key: str
            Input's key.

        Returns
        -------
        tuple: DOFA's outputs.
        """
        # Forward pass through the dinov2 encoder, get encoder's feature maps.
        input = batch[key][:, :self.in_channels, :, :]

        output_dofa = self.dofa(input, self.wavelengths)
        output = self.segmentation_head(output_dofa)
        
        return output, []
        
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dynamic One-For-All (DOFA) models."""

from functools import partial
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from timm.models.vision_transformer import Block
from torch import Tensor

def position_embedding(embed_dim: int, pos: Tensor) -> Tensor:
    """Compute the 1D sine/cosine position embedding.

    Args:
        embed_dim: Output dimension D for each position. Must be even.
        pos: A list of positions to be encoded, of size (M,).

    Returns:
        Position embeddings of size (M, D).

    Raises:
        AssertionError: If *embed_dim* is not even.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class TransformerWeightGenerator(nn.Module):
    """Dynamic weight generator for DOFA."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        """Initialize a new TransformerWeightGenerator instance.

        Args:
            input_dim: Input dimensions.
            output_dim: Output dimensions.
            embed_dim: Embedding dimensions.
            num_heads: Number of heads.
            num_layers: Number of layers.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation='gelu',
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is
        # too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the model.

        Args:
            x: Input mini-batch of size (seq_len, batch, input_dim).

        Returns:
            Weight and bias.
        """
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        # Using the last output to generate bias
        bias = self.fc_bias(transformer_output[-1])
        return weights, bias


class FCResLayer(nn.Module):
    """Fully-connected residual layer."""

    def __init__(self, linear_size: int = 128) -> None:
        """Initialize a new FCResLayer instance.

        Args:
            linear_size: Size of linear layer.
        """
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input mini-batch.

        Returns:
            Output of the model.
        """
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out: Tensor = x + y
        return out


class DOFAEmbedding(nn.Module):
    """Dynamic One-For-All (DOFA) embedding."""

    def __init__(
        self, dynamic_embed_dim: int, kernel_size: int = 3, embed_dim: int = 1024
    ) -> None:
        """Initialize a new DOFAEmbedding instance.

        Args:
            dynamic_embed_dim: Dimensions of dynamic weight generator.
            kernel_size: Kernel size of the depth-wise convolution.
            embed_dim: Embedding dimensions.
        """
        super().__init__()
        self.dynamic_embed_dim = dynamic_embed_dim
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(
            dynamic_embed_dim, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(dynamic_embed_dim)

        self._init_weights()

    def _init_weight(self, m: object) -> None:
        """Initialize weights of a single layer.

        Args:
            m: A single layer.
        """
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self) -> None:
        """Initialize weights of all layers."""
        self.weight_generator.apply(self._init_weight)
        self.fclayer.apply(self._init_weight)

    def forward(self, x: Tensor, wavelengths: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the model.

        Args:
            x: Input mini-batch.
            wavelengths: Wavelengths of each spectral band (μm).

        Return:
            Output mini-batch and wavelengths.
        """
        inplanes = wavelengths.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        waves = position_embedding(self.dynamic_embed_dim, wavelengths * 1000)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)  # 3x3x3

        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            x, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


class DOFA(nn.Module):
    """Dynamic One-For-All (DOFA) model.

    Reference implementation:

    * https://github.com/zhu-xlab/DOFA

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2403.15356

    .. versionadded:: 0.6
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        drop_rate: float = 0.0,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        dynamic_embed_dim: int = 128,
        num_classes: int = 45,
        global_pool: bool = True,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # type: ignore[assignment]
        task: str = 'segmentation'
    ) -> None:
        """Initialize a new DOFA instance.

        Args:
            img_size: Input image size.
            patch_size: Patch size.
            drop_rate: Head dropout rate.
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            dynamic_embed_dim: Dimensions of dynamic weight generator.
            num_classes: Number of classes for classification head.
            global_pool: Whether or not to perform global pooling.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim.
            norm_layer: Normalization layer.
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dynamic_embed_dim = dynamic_embed_dim
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.mlp_ratio = mlp_ratio
        self.task = task

        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = DOFAEmbedding(
            dynamic_embed_dim=128, kernel_size=16, embed_dim=embed_dim
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # ---------------------------------------------------------------------------
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )


    def forward_features(self, x: Tensor, wavelengths: list[float]) -> Tensor:
        """Forward pass of the feature embedding layer.

        Args:
            x: Input mini-batch.
            wavelengths: Wavelengths of each spectral band (μm).

        Returns:
            Output mini-batch.
        """
        # embed patches
        wavelist = torch.tensor(wavelengths, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)

        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome: Tensor = self.fc_norm(x)
        elif self.task == 'segmentation':
            x = self.norm(x)
            outcome = x[:, 1:, :]
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome


    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        """Forward pass of the attention head.

        Args:
            x: Input mini-batch.
            pre_logits: Whether or not to return the layer before logits are computed.

        Returns:
            Output mini-batch.
        """
        x = self.head_drop(x)
        x = x if pre_logits else self.head(x)
        return x

    def forward(self, x: Tensor, wavelengths: list[float]) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input mini-batch.
            wavelengths: Wavelengths of each spectral band (μm).

        Returns:
            Output mini-batch.
        """
        x = self.forward_features(x, wavelengths)
        x = self.forward_head(x)
        return x