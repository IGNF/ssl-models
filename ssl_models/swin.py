"""Code from https://github.com/Sense-X/MixMIM/tree/master.

# References:
# Swin: https://github.com/microsoft/Swin-Transformer
# timm: https://github.com/rwightman/pytorch-image-models
"""

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.swin_transformer import window_partition, window_reverse
from timm.models.vision_transformer import DropPath, Mlp
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class WindowAttentionSwinV2(nn.Module):
    """Mask window attention of SwinV2."""

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        pretrained_window_size: tuple[int, int],
        nb_mix_images: int,
    ) -> None:
        super().__init__()
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.nb_mix_images = nb_mix_images

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))),
        )

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1),
            self.window_size[0],
            dtype=torch.float32,
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1),
            self.window_size[1],
            dtype=torch.float32,
        )
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / np.log2(8)
        )

        self.register_buffer(
            "relative_coords_table",
            relative_coords_table,
            persistent=False,
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1,
            2,
            0,
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        b_, n, c = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                ),
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(b_, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # attention
        logit_scale = torch.clamp(self.logit_scale, max=np.log(1.0 / 0.01)).exp()
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        q = q * logit_scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(
            -1,
            self.num_heads,
        )
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2,
            0,
            1,
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.reshape(b_, 1, 1, n)
            mask = torch.sum(
                torch.stack(
                    [
                        (mask == i).long() * (mask == i).long().transpose(2, 3)
                        for i in range(self.nb_mix_images)
                    ],
                ),
                dim=0,
            )
            mask = 1 - mask
            if mask.dtype == torch.float16:
                attn = attn - 65500 * mask
            else:
                attn = attn - 1e30 * mask
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return self.proj_drop(x)


class WindowAttention(nn.Module):
    """Mask window attention of SwinV1."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        nb_mix_images: int,
    ) -> None:
        super().__init__()
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.nb_mix_images = nb_mix_images

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads),
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1,
            2,
            0,
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B_, N, C = x.shape  # noqa: N806
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2,
            0,
            1,
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.reshape(B_, 1, 1, N)
            mask = torch.sum(
                torch.stack(
                    [
                        (mask == i).long() * (mask == i).long().transpose(2, 3)
                        for i in range(self.nb_mix_images)
                    ],
                ),
                dim=0,
            )

            mask = 1 - mask
            if mask.dtype == torch.float16:
                attn = attn - 65500 * mask
            else:
                attn = attn - 1e30 * mask
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class MixMAEBlock(nn.Module):
    """MixMAE block."""

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        act_layer: nn.GELU,
        norm_layer: nn.LayerNorm,
        pretrained_window_size: int,
        swin_version: int,
        nb_mix_images: int,
        v2_attn: bool,
    ) -> None:
        super().__init__()
        self.swin_version = swin_version
        self.input_resolution = input_resolution
        self.window_size = window_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution,\
            # we don't partition windows
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)

        if swin_version == 2 or v2_attn:  # noqa: PLR2004
            self.attn = WindowAttentionSwinV2(
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                pretrained_window_size=to_2tuple(pretrained_window_size),
                nb_mix_images=nb_mix_images,
            )
        else:
            self.attn = WindowAttention(
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                nb_mix_images=nb_mix_images,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        H, W = self.input_resolution  # noqa: N806
        B, _, C = x.shape  # noqa: N806

        shortcut = x
        if self.swin_version == 1:
            x = self.norm1(x)
        x = x.view(B, H, W, C)

        # partition windows
        x_windows = window_partition(
            x,
            self.window_size,
        )  # nW*B, window_size, window_size, c

        x_windows = x_windows.view(
            -1,
            self.window_size * self.window_size,
            C,
        )  # nW*B, window_size*window_size, c
        if attn_mask is not None:
            attn_mask = attn_mask.view(B, H, W, 1)
            attn_mask = window_partition(attn_mask, self.window_size)
            attn_mask = attn_mask.view(
                -1,
                self.window_size * self.window_size,
                1,
            )

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows,
            mask=attn_mask,
        )  # nW*B, window_size*window_size, c

        # merge windows
        attn_windows = attn_windows.view(
            -1,
            self.window_size,
            self.window_size,
            C,
        )
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' c

        x = x.view(B, H * W, C)

        if self.swin_version == 1:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        elif self.swin_version == 2:  # noqa: PLR2004
            x = shortcut + self.drop_path(self.norm1(x))
            # FFN
            x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class MixMAELayer(nn.Module):
    """MixMae layer."""

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        norm_layer: nn.LayerNorm,
        downsample: None,
        use_checkpoint: bool,
        pretrained_window_size: int,
        swin_version: int,
        nb_mix_images: int,
        v2_attn: bool,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                MixMAEBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size,
                    swin_version=swin_version,
                    nb_mix_images=nb_mix_images,
                    v2_attn=v2_attn,
                ),
            )
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask=attn_mask)
        if self.downsample is not None:
            H, W = self.input_resolution  # noqa: N806
            B, _, C = x.shape  # noqa: N806
            x = x.view(B, H, W, C)

            x = self.downsample(x)

            x = x.view(B, H * W // 4, -1)

        return x

    def init_respostnorm(self) -> None:
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)
