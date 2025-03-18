from torchgeo.models import DOFABase16_Weights, DOFALarge16_Weights
import torch
from torch import Tensor
from abc import ABC
from torch.nn import ModuleDict, Parameter, ParameterDict, Identity
from ssl_models.baselines.base import BaseModule
from typing import Literal, List
from ssl_models.baselines.utils import filter_dict
from timm.models.vision_transformer import Block

class DOFABaseline(BaseModule, ABC):
    def __init__(
        self,
        datasets,
        backbone_size: str,
        unpool_dim: int,
        pretrained_path: str = None,
        freeze: bool = False,
        type_head: List[str] = "linear",
        multimodal: str = "shared",
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        keep_norm: bool = True,
        **kwargs,
    ) -> None:

        self.backbone_size = backbone_size
        self.freeze = freeze
        
        self.patch_size = 16

        self.args_by_size = {
                             "base":  {"embed_dim": 768, "depth": 12, "num_heads": 12},
                             "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
                            }
        self.weights_by_size = {"base": DOFABase16_Weights.DOFA_MAE,
                                "large": DOFALarge16_Weights.DOFA_MAE}

        assert backbone_size in self.args_by_size.keys(), "backbone_size should be in {}".format(self.args_by_size.keys())
        
        if pretrained_path is not None:
            self.weights_dict = torch.load(pretrained_path)
        else:
            self.weights_dict = self.weights_by_size[self.backbone_size].get_state_dict(progress=True)

        #params
        self.depth = self.args_by_size[backbone_size]["depth"]
        self.embed_dim = self.args_by_size[backbone_size]["embed_dim"]
        self.num_heads = self.args_by_size[backbone_size]["num_heads"]
        self.mlp_ratio = 4.0
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.type_head = type_head
        self.multimodal = multimodal
        self.unpool_dim = unpool_dim
        self.fac_date_enc = fac_date_enc
        self.date_dim = date_dim
        self.keep_norm = keep_norm
        
        super().__init__(datasets, self.patch_size, self.embed_dim, self.unpool_dim, self.type_head, self.multimodal, self.fac_date_enc, self.date_dim, self.keep_norm, **kwargs)

        #layers
        self.wavelengths = {"aerial":[0.64, 0.56, 0.48, 0.81], 
                            "spot":[0.66, 0.56, 0.48], 
                            "s2":[0.665, 0.560, 0.490, 0.842, 0.705, 0.740, 0.783, 0.865, 1.610, 2.190],
                            "s1":[5.405, 5.405]}
        self._build_backbone()

    def _build_backbone(self):
        #layers
        self.patch_embed, self.pos_embed = self._build_patch_and_pos_embed(self.embed_dim, self.weights_dict)
        self.cls_token = self._build_cls_token(self.embed_dim, self.weights_dict)

        self.encoder = self._build_encoder(self.depth, self.embed_dim, self.num_heads, self.mlp_ratio, self.norm_layer, self.weights_dict)
        
        if self.keep_norm:
            self.layernorm = self._build_layernorm(self.embed_dim, self.norm_layer, self.weights_dict)
        else:
            self.layernorm = Identity()

    def _build_patch_and_pos_embed(self, embed_dim, state_dict=None):
        patch_embed = ModuleDict()
        pos_embed = ParameterDict()
    
        for name_mod in self.dataset.inputs:
            patch_embed[name_mod] = DOFAEmbedding(
                dynamic_embed_dim=128, kernel_size=16, embed_dim=embed_dim
            )

            num_patches = (self.dataset.inputs[name_mod].image_size // self.patch_size) ** 2 

            pos_embed[name_mod] = Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

            if state_dict is not None:
                self._transfer(state_dict, patch_embed[name_mod], "patch_embed")

        return patch_embed, pos_embed
    
    def _build_cls_token(self, embed_dim, state_dict):
        cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if state_dict is not None:
            cls_token.data = state_dict["cls_token"]
        
        return cls_token 
    
    def _build_encoder(self, depth, embed_dim, num_heads, mlp_ratio, norm_layer, state_dict=None):
        encoder = ModuleDict()
        
        model_names = self.dataset.inputs if self.multimodal != "shared" else ["shared"]

        for name_mod in model_names:
            encoder_mod = nn.ModuleList([Block(embed_dim,num_heads,mlp_ratio, qkv_bias=True,norm_layer=norm_layer) for i in range(depth)])
            if state_dict is not None:
                self._transfer(state_dict, encoder_mod, "blocks")

            encoder[name_mod] = encoder_mod

        return encoder
        
    def _build_layernorm(self, embed_dim, norm_layer, state_dict=None):
        layernorm = norm_layer(embed_dim)

        if state_dict is not None:
            self._transfer(state_dict, layernorm, "norm")

        return layernorm

    def _transfer(self, state_dict, module, headname):
        module.load_state_dict(filter_dict(state_dict, headname), strict=False)

    def _freeze_backbone(self):
        """
        Freeze the backbone's parameters (self.dofa).

        Parameters
        ----------
        Returns
        -------
        """
        for name_mod in self.patch_embed.keys():
            for param in self.patch_embed[name_mod].parameters():
                param.requires_grad = False
        
        self.cls_token.requires_grad = False
        
        for name_mod in self.encoder.keys():
            for param in self.encoder[name_mod].parameters():
                param.requires_grad = False
        
        for param in self.layernorm.parameters():
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
        lr_map = {}

        for name_mod in self.patch_embed:
            lr_map["patch_embed.{}".format(name_mod)] = base_lr * rate_decay ** (self.depth+1)

        for name_mod in self.encoder:
            for i in range(self.depth):
                lr_map["encoder.{}.{}".format(name_mod, i)] = base_lr * rate_decay ** (self.depth-i)
        
        lr_map["heads"] = base_lr
        
        params_dict = {}
        for name_mod in self.patch_embed:
            params_dict["patch_embed.{}".format(name_mod)] = self.patch_embed[name_mod].parameters()
        
        for name_mod in self.encoder:
            for i in range(self.depth):
                params_dict["encoder.{}.{}".format(name_mod, i)] = self.encoder[name_mod][i].parameters()
        params_dict["heads"] = self.heads.parameters()
        
        if self.keep_norm:
            lr_map["layernorm"] = base_lr
            params_dict["layernorm"] = self.layernorm.parameters()

        print('Layerwise learning rates:')
        grouped_parameters = []
        for k, lr in lr_map.items():
            grouped_parameters.append({"params": params_dict[k], "lr": lr, "name": k})
            print('Layer "{}": {}'.format(k, lr))

        return grouped_parameters
    
    def forward(self, batch: dict[str, Tensor],         
                ssl_phase: Literal["pretrain", "probe", "finetune"],
                stage: Literal["train", "val", "test"],
    ):
        # append cls token
        input = self.group(batch)
        x, dates = {}, {}

        for name_mod in self.dataset.inputs:
            input_mod = input[name_mod]
            waves_mod = torch.tensor(self.wavelengths[name_mod], device=input_mod.device).float()

            x_tokenized, _ = self.patch_embed[name_mod](input_mod, waves_mod)
            x_tokenized = x_tokenized + self.pos_embed[name_mod][:, 1:, :]
            cls_token = self.cls_token + self.pos_embed[name_mod][:, :1, :]
            cls_tokens = cls_token.expand(x_tokenized.shape[0], -1, -1)
            x_enc = torch.cat((cls_tokens, x_tokenized), dim=1)
            
            encoder_mod = "shared" if self.multimodal == "shared" else name_mod
            for block in self.encoder[encoder_mod]:
                x_enc = block(x_enc)
                            
            x_enc = self.layernorm(x_enc) #??

            x[name_mod] = x_enc[:, 1:]
            dates[name_mod] = batch[f"{name_mod}_dates"]

        ref_date = batch["ref_date"]
        x = self._add_date_encodings(x, dates, ref_date)

        loss_pred, log_input, log_pred, log_target = self.compute_loss_pred(
            x,
            batch,
            ssl_phase,
            stage,
        )
        self.metrics[f"loss_pred_{stage}"].update(loss_pred)
        return None, log_input, log_pred, log_target, loss_pred

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
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

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
            activation="gelu",
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
        task: str = "segmentation"
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
        elif self.task == "segmentation":
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