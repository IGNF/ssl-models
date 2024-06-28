"""Code from https://github.com/Sense-X/MixMIM/tree/master.

# References:
# Swin: https://github.com/microsoft/Swin-Transformer
# timm: https://github.com/rwightman/pytorch-image-models
# MAE:  https://github.com/facebookresearch/mae
# --------------------------------------------------------
"""

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.swin_transformer import (
    PatchMerging,
    window_partition,
    window_reverse,
)
from timm.models.vision_transformer import Block, DropPath, Mlp, PatchEmbed
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ssl_models.utils import get_2d_sincos_pos_embed, posemb_sincos_2d


class WindowAttentionSwinV2(nn.Module):
    """Mask window attention of SwinV2."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        pretrained_window_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))),
            requires_grad=True,
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

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :],
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
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)

        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(torch.tensor(1.0 / 0.01, device=self.logit_scale.device)),
        ).exp()
        attn = attn * logit_scale

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
            mask_new = mask * mask.transpose(2, 3) + (1 - mask) * (1 - mask).transpose(
                2,
                3,
            )
            mask_new = 1 - mask_new
            if mask_new.dtype == torch.float16:
                attn = attn - 65500 * mask_new
            else:
                attn = attn - 1e30 * mask_new
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
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

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
            mask_new = mask * mask.transpose(2, 3) + (1 - mask) * (1 - mask).transpose(
                2,
                3,
            )
            mask_new = 1 - mask_new
            if mask_new.dtype == torch.float16:
                attn = attn - 65500 * mask_new
            else:
                attn = attn - 1e30 * mask_new
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class MixMAEBlock(nn.Module):
    """MixMAE block."""

    def __init__(
        self,
        dim: int,
        input_resolution,
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
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.swin_version = swin_version
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution,\
            # we don't partition windows
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        if self.swin_version == 2:  # noqa: PLR2004
            self.attn = WindowAttentionSwinV2(
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                pretrained_window_size=to_2tuple(pretrained_window_size),
            )
        else:
            self.attn = WindowAttention(
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
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
            if attn_mask.shape[0] == 1:
                attn_mask = attn_mask.repeat(B, 1, 1)  # B, N, 1
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
        input_resolution,
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
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.swin_version = swin_version

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
                    swin_version=self.swin_version,
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

            bd, hd, wd, cd = x.shape
            x = x.view(bd, hd * wd, cd)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim},\
            input_resolution={self.input_resolution},\
            depth={self.depth}"

    def _init_respostnorm(self) -> None:
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class MixMAE(nn.Module):
    """Mixed ans Masked Auto Encoder (MixMAE)."""

    def __init__(  # noqa: PLR0913
        self,
        decoder_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        mlp_ratio: int,
        norm_pix_loss: bool,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        depths: tuple[int, int, int, int],
        num_heads: tuple[int, int, int, int],
        window_size: tuple[int, int, int, int],
        qkv_bias: bool,
        patch_norm: bool,
        drop_rate: float,
        drop_path_rate: float,
        attn_drop_rate: float,
        norm_layer: nn.LayerNorm,
        use_checkpoint: bool,
        range_mask_ratio: float,
        pretrained_window_sizes: tuple[int, int, int, int],
        mask_batch: bool,
        swin_version: int,
    ) -> None:
        super().__init__()
        # decoder args
        self.decoder_dim = decoder_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads

        # encoder args
        self.encoder_stride = 8 * patch_size
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.use_checkpoint = use_checkpoint
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.swin_version = swin_version

        # reconstruction args
        self.norm_pix_loss = norm_pix_loss
        self.range_mask_ratio = range_mask_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            flatten=True,
        )
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size

        self.dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(
                MixMAELayer(
                    dim=int(self.embed_dim * 2**i_layer),
                    input_resolution=(
                        self.patch_grid[0] // (2**i_layer),
                        self.patch_grid[1] // (2**i_layer),
                    ),
                    depth=self.depths[i_layer],
                    num_heads=self.num_heads[i_layer],
                    window_size=self.window_size[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=self.dpr[
                        sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=(
                        PatchMerging if (i_layer < self.num_layers - 1) else None
                    ),
                    use_checkpoint=self.use_checkpoint,
                    pretrained_window_size=pretrained_window_sizes[i_layer],
                    swin_version=self.swin_version,
                ),
            )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(self.num_features)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

        num_patches = self.patch_embed.num_patches
        out_num_patches = (self.img_size // self.encoder_stride) ** 2
        self.out_num_patches = out_num_patches
        self.absolute_pos_embed = (
            nn.Parameter(
                torch.zeros(1, num_patches, self.embed_dim),
                requires_grad=False,
            ),
        )
        self.decoder_pos_embed = (
            nn.Parameter(
                torch.zeros(1, out_num_patches, decoder_dim),
                requires_grad=False,
            ),
        )

        self.decoder_embed = nn.Linear(self.num_features, decoder_dim)
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ],
        )

        self.decoder_norm = norm_layer(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.encoder_stride**2 * 3)

        self.initialize_weights()
        if self.swin_version == 2:  # noqa: PLR2004
            for bly in self.layers:
                bly._init_respostnorm()

        self.mask_batch = mask_batch

    def initialize_weights(self) -> None:
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(
        #     self.absolute_pos_embed.shape[-1], # embed_dim
        #     int(self.patch_embed.num_patches**0.5), # grid size
        #     cls_token=False,
        # )
        pos_embed = posemb_sincos_2d(
            int(self.patch_embed.num_patches**0.5),
            int(self.patch_embed.num_patches**0.5),
            self.absolute_pos_embed.shape[-1],
        )

        self.absolute_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
        )

        # decoder_pos_embed = get_2d_sincos_pos_embed(
        #     self.decoder_pos_embed.shape[-1],
        #     int(self.out_num_patches**0.5),
        #     cls_token=False,
        # )
        decoder_pos_embed = posemb_sincos_2d(
            int(self.out_num_patches**0.5),
            int(self.out_num_patches**0.5),
            self.decoder_pos_embed.shape[-1],
        )

        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0),
        )

    def random_masking(
        self,
        x: Tensor,
        mask_batch: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        b, _, h, w = x["image"].shape
        nb_masks_batch = 1
        if not mask_batch:
            nb_masks_batch = b // 4

        mask_list = []
        mask_s1_list = []
        mask_s2_list = []
        mask_s3_list = []
        mask_mix_image_list = []

        out_h = h // self.encoder_stride
        out_w = w // self.encoder_stride
        s3_h, s3_w = out_h * 2, out_w * 2
        s2_h, s2_w = out_h * 4, out_w * 4
        s1_h, s1_w = out_h * 8, out_w * 8

        seq_l = out_h * out_w

        # use a shared mask for a batch images
        mask_mix_images_init = torch.cat(
            (
                torch.zeros(int(seq_l / 4)),
                torch.ones(int(seq_l / 4)),
                torch.ones(int(seq_l / 4)) * 2,
                torch.ones(int(seq_l / 4)) * 3,
            ),
        )

        for _ in range(nb_masks_batch):
            mask_idx = torch.randperm(seq_l)
            mask_mix_images = torch.reshape(mask_mix_images_init[mask_idx], (1, seq_l))
            mask_mix_images = mask_mix_images.to(device=x["image"].device)

            mask = mask_mix_images.clone()
            mask[mask != 0] = 1

            mask = mask.reshape(1, 1, out_h, out_w)
            mask_mix_images = mask_mix_images.reshape(1, 1, out_h, out_w)
            mask_s1 = torch.nn.functional.interpolate(
                mask,
                size=(s1_h, s1_w),
                mode="nearest",
            )
            mask_s2 = torch.nn.functional.interpolate(
                mask,
                size=(s2_h, s2_w),
                mode="nearest",
            )
            mask_s3 = torch.nn.functional.interpolate(
                mask,
                size=(s3_h, s3_w),
                mode="nearest",
            )
            mask_mix_images = torch.nn.functional.interpolate(
                mask_mix_images,
                size=(s1_h, s1_w),
                mode="nearest",
            )

            mask_list.append(mask)
            mask_s1_list.append(mask_s1)
            mask_s2_list.append(mask_s2)
            mask_s3_list.append(mask_s3)
            mask_mix_image_list.append(mask_mix_images)

        mask = torch.cat(tuple(mask_list), dim=0)
        mask_s1 = torch.cat(tuple(mask_s1_list), dim=0)
        mask_s2 = torch.cat(tuple(mask_s2_list), dim=0)
        mask_s3 = torch.cat(tuple(mask_s3_list), dim=0)
        mask_mix_images = torch.cat(tuple(mask_mix_image_list), dim=0)

        mask = mask.reshape(nb_masks_batch, out_h * out_w, 1).contiguous()
        mask_s1 = mask_s1.reshape(nb_masks_batch, s1_h * s1_w, 1).contiguous()
        mask_s2 = mask_s2.reshape(nb_masks_batch, s2_h * s2_w, 1).contiguous()
        mask_s3 = mask_s3.reshape(nb_masks_batch, s3_h * s3_w, 1).contiguous()
        mask_mix_images = mask_mix_images.reshape(
            nb_masks_batch,
            s1_h * s1_w,
            1,
        ).contiguous()

        return mask_s1, mask_s2, mask_s3, mask, mask_mix_images

    def patchify(self, imgs: Tensor) -> Tensor:
        """Patchify the input.

        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.encoder_stride
        if imgs.shape[2] != imgs.shape[3]:
            msg = (
                f"Invalid image shape {imgs.shape}."
                f" Expected same size for H and W.",
            )
            raise ValueError(msg)
        if imgs.shape[2] % p != 0:
            msg = (
                f"Invalid patch size {imgs.shape}."
                f" Expected divisor of image_size {imgs.shape[2]}",
            )
            raise ValueError(msg)

        H = W = imgs.shape[2] // p  # noqa: N806
        x = imgs.reshape(shape=(imgs.shape[0], 3, H, p, W, p))
        x = rearrange(
            x,
            "n c h p w q -> n h w p q c",
            n=imgs.shape[0],
            c=3,
            h=H,
            w=W,
            p=p,
            q=p,
        )

        return x.reshape(shape=(imgs.shape[0], H * W, p**2 * 3))

    def mixing_images(self, x: Tensor, mask_mix_images: Tensor) -> Tensor:
        """Mix 4 images to create an input.

        x: (B, L, c)
        mask_mix_images: (1, H*patch_size**2 , 1)
        mix_imgs: (B//4, L, c)
        """
        imgs_a, imgs_b, imgs_c, imgs_d = torch.tensor_split(x, 4, dim=0)

        mask_a = mask_mix_images.clone()
        mask_a[mask_mix_images != 0] = 1

        mask_b = mask_mix_images.clone()
        mask_b[mask_mix_images == 1] = 0
        mask_b[mask_mix_images != 1] = 1

        mask_c = mask_mix_images.clone()
        mask_c[mask_mix_images == 2] = 0  # noqa: PLR2004
        mask_c[mask_mix_images != 2] = 1  # noqa: PLR2004

        mask_d = mask_mix_images.clone()
        mask_d[mask_mix_images == 3] = 0  # noqa: PLR2004
        mask_d[mask_mix_images != 3] = 1  # noqa: PLR2004

        return (
            imgs_a * (1.0 - mask_a)
            + imgs_b * (1.0 - mask_b)
            + imgs_c * (1.0 - mask_c)
            + imgs_d * (1.0 - mask_d)
        )

    def forward_encoder(
        self,
        x: Tensor,
        mask_s1: Tensor,
        mask_s2: Tensor,
        mask_s3: Tensor,
        mask_s4: Tensor,
        mask_mix_images: Tensor,
    ) -> Tensor:
        x = self.patch_embed(x)

        x = self.mixing_images(x, mask_mix_images)

        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                x = layer(x, attn_mask=mask_s1)
            elif idx == 1:
                x = layer(x, attn_mask=mask_s2)
            elif idx == 2:  # noqa: PLR2004
                x = layer(x, attn_mask=mask_s3)
            elif idx == 3:  # noqa: PLR2004
                x = layer(x, attn_mask=mask_s4)

        return self.norm(x)

    def forward_decoder(self, x: Tensor, mask: Tensor) -> Tensor:
        # embed tokens
        x = self.decoder_embed(x)
        B, L, _ = x.shape  # noqa: N806

        mask_tokens = self.mask_token.expand(B, L, -1)
        x = x * (1 - mask) + mask_tokens * mask

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for _, blk in enumerate(self.decoder_blocks):
            x = blk(x)
        x = self.decoder_norm(x)

        return self.decoder_pred(x)

    def forward_loss(self, x: Tensor, x_rec: Tensor, mask: Tensor) -> Tensor:
        # initial batch size is divided by 4
        new_batch_size = int(x.shape[0] / 4)
        x = x[:new_batch_size, :, :, :]
        target = self.patchify(x)

        loss_rec = (x_rec - target) ** 2
        loss_rec_masked = torch.masked_select(loss_rec, mask.bool())

        return loss_rec_masked.mean()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mask_s1, mask_s2, mask_s3, mask_s4, mask_mix_images = self.random_masking(
            x,
            self.mask_batch,
        )
        z = self.forward_encoder(
            x["image"],
            mask_s1,
            mask_s2,
            mask_s3,
            mask_s4,
            mask_mix_images,
        )
        x_rec = self.forward_decoder(z, mask_s4)

        loss = self.forward_loss(x["image"], x_rec, mask_s4)

        # masked
        b_mask, mask_l, _ = mask_s1.shape
        mask_w = mask_h = int(mask_l**0.5)
        mask_input = mask_s1.view(b_mask, 1, mask_h, mask_w)
        B, C, H, W = x["image"].shape  # noqa: N806
        mask_input = torch.nn.functional.interpolate(
            mask_input,
            size=(H, W),
            mode="nearest",
        )
        masked = x["image"][: mask_input.shape[0], :, :, :] * (1 - mask_input)
        masked = torch.where(masked != 0, masked, 1)

        # pred
        _, mask_l, _ = mask_s4.shape
        mask_w = mask_h = int(mask_l**0.5)
        mask_output = mask_s4.view(b_mask, 1, mask_h, mask_w)
        mask_output = torch.nn.functional.interpolate(
            mask_output,
            size=(H, W),
            mode="nearest",
        )

        _, _, x_rec_hw3 = x_rec.shape
        x_rec_hw = x_rec_hw3 / C
        x_rec_h = x_rec_w = int(x_rec_hw**0.5)

        x1_rec = x_rec

        x1_rec = rearrange(
            x1_rec,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=H // x_rec_h,
            p1=x_rec_h,
            p2=x_rec_w,
        )

        targets = x["image"][: B // 4, :, :, :]
        pred = x1_rec * mask_output + targets * (1 - mask_output)
        return loss, masked, pred


def mixmae_tiny(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE tiny."""
    default_args = {
        "img_size": kwargs["image_size"],
        "patch_size": 4,
        "in_chans": 3,
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": [16, 16, 16, 8],
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "use_checkpoint": False,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "norm_pix_loss": False,
        "mask_batch": True,
        "swin_version": 2,
    }
    default_args.update(**kwargs)
    return MixMAE(**default_args)


def mixmae_small(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE small."""
    default_args = {
        "img_size": kwargs["image_size"],
        "patch_size": 4,
        "in_chans": 3,
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": [16, 16, 16, 8],
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "use_checkpoint": False,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "norm_pix_loss": False,
        "mask_batch": True,
        "swin_version": 2,
    }
    default_args.update(**kwargs)
    return MixMAE(**default_args)


def mixmae_base(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE base."""
    default_args = {
        "img_size": kwargs["image_size"],
        "patch_size": 4,
        "in_chans": 3,
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "window_size": [16, 16, 16, 8],
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "use_checkpoint": False,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "norm_pix_loss": False,
        "mask_batch": True,
        "swin_version": 2,
    }
    default_args.update(**kwargs)
    return MixMAE(**default_args)


def mixmae_large(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE large."""
    default_args = {
        "img_size": 224,
        "patch_size": 4,
        "in_chans": 3,
        "embed_dim": 192,
        "depths": [2, 2, 18, 2],
        "num_heads": [6, 12, 24, 48],
        "window_size": [14, 14, 14, 7],
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "use_checkpoint": False,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "norm_pix_loss": False,
        "mask_batch": True,
        "swin_version": 2,
    }
    default_args.update(**kwargs)
    return MixMAE(**default_args)


def mixmae_huge(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE huge."""
    default_args = {
        "img_size": 224,
        "patch_size": 4,
        "in_chans": 3,
        "embed_dim": 352,
        "depths": [2, 2, 18, 2],
        "num_heads": [11, 22, 44, 88],
        "window_size": [14, 14, 14, 7],
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "use_checkpoint": False,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "norm_pix_loss": False,
        "mask_batch": True,
        "swin_version": 2,
    }
    default_args.update(**kwargs)
    return MixMAE(**default_args)
