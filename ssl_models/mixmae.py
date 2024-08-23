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
from timm.models.swin_transformer import PatchMerging, window_partition, window_reverse
from timm.models.vision_transformer import Block, DropPath, Mlp, PatchEmbed
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ssl_models.head import AttentiveHead, LinearHead
from ssl_models.odeon_data import OdeonData
from ssl_models.utils import posemb_sincos_2d


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
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.nb_mix_images = nb_mix_images

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
            mask = rearrange(
                mask,
                "bn h w c -> bn c h w",
            )
            if mask.dtype == torch.float16:
                attn = attn + 65500 * mask
            else:
                attn = attn + 1e30 * mask
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
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
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
            mask = rearrange(
                mask,
                "bn h w c -> bn c h w",
            )
            if mask.dtype == torch.float16:
                attn = attn + 65500 * mask
            else:
                attn = attn + 1e30 * mask
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
        shift_size: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.swin_version = swin_version
        self.nb_mix_images = nb_mix_images
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution,\
            # we don't partition windows
            self.window_size = min(self.input_resolution)
            self.shift_size = 0

        if not 0 <= self.shift_size < self.window_size:
            msg = f"shift_size must be in 0-window_size. Not {shift_size}"
            raise ValueError(msg)

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
                nb_mix_images=self.nb_mix_images,
            )
        else:
            self.attn = WindowAttention(
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                nb_mix_images=self.nb_mix_images,
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

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            attn_mask_roll = torch.roll(
                attn_mask.view(-1, H, W),
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            ).reshape(-1, H, W, 1)

            attn_mask = rearrange(
                attn_mask,
                "b (h w) c -> b h w c",
                h=H,
                w=W,
            )

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((B, H, W, 1), device=x.device)  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = (
                        cnt * self.nb_mix_images + attn_mask_roll[:, h, w, :]
                    )
                    cnt += 1

            mask_windows = window_partition(
                img_mask,
                self.window_size,
            )  # nW*B, window_size, window_size, 1
            mask_windows = mask_windows.view(
                -1,
                self.window_size * self.window_size,
            )  # nW*B, sindow_size*window_size
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0,
                0.0,
            )
            attn_mask = attn_mask.unsqueeze(3)
        else:
            if attn_mask.shape[0] == 1:
                attn_mask = attn_mask.repeat(B, 1, 1)
            attn_mask = attn_mask.view(B, H, W, 1)
            attn_mask = window_partition(attn_mask, self.window_size)
            attn_mask = attn_mask.view(
                -1,
                self.window_size * self.window_size,
                1,
            )
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0,
                0.0,
            )

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

        x = window_reverse(
            attn_windows,
            self.window_size,
            H,
            W,
        )  # B H' W' c

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.reshape(B, H * W, C)

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
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.swin_version = swin_version
        self.nb_mix_images = nb_mix_images

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
                    nb_mix_images=self.nb_mix_images,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
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

    def extra_repr(self) -> str:
        return f"dim={self.dim},\
            input_resolution={self.input_resolution},\
            depth={self.depth}"

    def init_respostnorm(self) -> None:
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
        image_size: int,
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
        use_checkpoint: bool,
        mask_batch: bool,
        swin_version: int,
        nb_mix_images: int,
        attn_drop_rate: float = 0.0,
        norm_layer: nn.LayerNorm = nn.LayerNorm,
        range_mask_ratio: float = 0.0,
        pretrained_window_sizes: tuple[int, int, int, int] = [0, 0, 0, 0],
        type_head: str = "attentive",
        **kwargs,  # noqa: ARG002, ANN003
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
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.swin_version = swin_version
        self.nb_mix_images = nb_mix_images

        # reconstruction args
        self.norm_pix_loss = norm_pix_loss
        self.range_mask_ratio = range_mask_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=image_size,
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
                    nb_mix_images=self.nb_mix_images,
                ),
            )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(self.num_features)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

        num_patches = self.patch_embed.num_patches
        out_num_patches = (self.image_size // self.encoder_stride) ** 2
        self.out_num_patches = out_num_patches

        self.register_buffer(
            name="absolute_pos_embed",
            tensor=torch.zeros(1, num_patches, self.embed_dim),
            persistent=False,
        )

        self.register_buffer(
            name="decoder_pos_embed",
            tensor=torch.zeros(1, out_num_patches, decoder_dim),
            persistent=False,
        )

        # Freeze the weights of position & band encoding
        self.absolute_pos_embed = self.absolute_pos_embed.requires_grad_(
            requires_grad=False,
        )
        self.decoder_pos_embed = self.decoder_pos_embed.requires_grad_(
            requires_grad=False,
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
                bly.init_respostnorm()

        self.mask_batch = mask_batch

        self.num_classes = OdeonData.NUM_CLASSES
        self.head = nn.ModuleDict(
            [
                [
                    dataset,
                    AttentiveHead(
                        self.embed_dim * 2 ** (self.num_layers - 1),
                        num_classes,
                    )
                    if type_head == "attentive"
                    else LinearHead(
                        self.embed_dim * 2 ** (self.num_layers - 1),
                        num_classes,
                    ),
                ]
                for dataset, num_classes in self.num_classes.items()
            ],
        )

    def initialize_weights(self) -> None:
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = posemb_sincos_2d(
            int(self.patch_embed.num_patches**0.5),
            int(self.patch_embed.num_patches**0.5),
            self.absolute_pos_embed.shape[-1],
        )

        self.absolute_pos_embed.data.copy_(
            pos_embed.float().unsqueeze(0),
        )

        decoder_pos_embed = posemb_sincos_2d(
            int(self.out_num_patches**0.5),
            int(self.out_num_patches**0.5),
            self.decoder_pos_embed.shape[-1],
        )

        self.decoder_pos_embed.data.copy_(
            decoder_pos_embed.float().unsqueeze(0),
        )

    def random_masking(
        self,
        x: Tensor,
        phase: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        B, _, H, W = x["image"].shape  # noqa: N806
        nb_masks_batch = 1
        if not self.mask_batch and phase == "pretrain":
            nb_masks_batch = B // self.nb_mix_images

        mask_list = []
        mask_s1_list = []
        mask_s2_list = []
        mask_s3_list = []

        out_h = H // self.encoder_stride
        out_w = W // self.encoder_stride
        s3_h, s3_w = out_h * 2, out_w * 2
        s2_h, s2_w = out_h * 4, out_w * 4
        s1_h, s1_w = out_h * 8, out_w * 8

        seq_l = out_h * out_w

        if phase == "pretrain":
            mask_init = torch.cat(
                [
                    torch.full(
                        [seq_l // self.nb_mix_images],
                        i,
                        dtype=torch.float32,
                        device=x["image"].device,
                    )
                    for i in range(self.nb_mix_images)
                ],
            )
        else:
            mask_init = torch.zeros(seq_l, device=x["image"].device)

        for i in range(nb_masks_batch):  # noqa: B007
            mask_idx = torch.randperm(seq_l)
            mask = torch.reshape(mask_init[mask_idx], (1, seq_l))

            mask = mask.reshape(1, 1, out_h, out_w)
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
            mask_list.append(mask)
            mask_s1_list.append(mask_s1)
            mask_s2_list.append(mask_s2)
            mask_s3_list.append(mask_s3)

        mask = torch.cat(tuple(mask_list), dim=0)
        mask_s1 = torch.cat(tuple(mask_s1_list), dim=0)
        mask_s2 = torch.cat(tuple(mask_s2_list), dim=0)
        mask_s3 = torch.cat(tuple(mask_s3_list), dim=0)

        mask = mask.reshape(nb_masks_batch, out_h * out_w, 1).contiguous()
        mask_s1 = mask_s1.reshape(nb_masks_batch, s1_h * s1_w, 1).contiguous()
        mask_s2 = mask_s2.reshape(nb_masks_batch, s2_h * s2_w, 1).contiguous()
        mask_s3 = mask_s3.reshape(nb_masks_batch, s3_h * s3_w, 1).contiguous()

        return mask_s1, mask_s2, mask_s3, mask

    def patchify(self, imgs: Tensor) -> Tensor:
        """Patchify the input.

        imgs: (N, 3, H, W)
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

        return rearrange(
            imgs,
            "b c (h p) (w q) -> b (h w) (p q c)",
            p=self.encoder_stride,
            q=self.encoder_stride,
        )

    def mix_images(
        self,
        x: Tensor,
        mask_mix_images: Tensor,
    ) -> Tensor:
        """Mix n images to create an input.

        x: (B, L, c)
        mask_mix_images: (1, H*patch_size**2 , 1)
        """
        imgs_split = list(torch.tensor_split(x, self.nb_mix_images, dim=0))
        imgs_mix_part = []

        imgs_mix_part = [
            imgs * torch.where(mask_mix_images == i, 1, 0)
            for i, imgs in enumerate(imgs_split)
        ]

        return torch.stack(imgs_mix_part, dim=0).sum(dim=0)

    def random_unmasking(self, mask: Tensor, B: int) -> Tensor:  # noqa: N803
        part_idx = torch.Tensor(list(range(self.nb_mix_images))).repeat_interleave(
            B // self.nb_mix_images,
            dim=0,
        )
        mask = mask.repeat((self.nb_mix_images, 1, 1))
        mask = torch.stack(
            [
                (mask_elem == idx_elem).float()
                for (mask_elem, idx_elem) in zip(mask, part_idx)
            ],
        )
        return 1 - mask

    def unmix_images(self, mask_s4: Tensor, z: Tensor) -> Tensor:
        z = z.repeat((self.nb_mix_images, 1, 1))
        mask_tokens = self.mask_token.expand(1, mask_s4.shape[1], -1)
        return z * (1 - mask_s4) + mask_tokens * mask_s4

    def forward_encoder(
        self,
        x: Tensor,
        mask_s1: Tensor,
        mask_s2: Tensor,
        mask_s3: Tensor,
        mask_s4: Tensor,
        phase: str,
    ) -> Tensor:
        x = self.patch_embed(x)

        if phase == "pretrain":
            x = self.mix_images(x, mask_s1)

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

    def forward_decoder(self, x: Tensor, mask: Tensor, ssl_phase: Tensor) -> Tensor:
        # embed tokens
        x = self.decoder_embed(x)
        _, L, _ = x.shape  # noqa: N806

        if ssl_phase == "pretrain":
            x = self.unmix_images(mask, x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for _, blk in enumerate(self.decoder_blocks):
            x = blk(x)
        x = self.decoder_norm(x)

        return self.decoder_pred(x)

    def forward_loss(
        self,
        x: Tensor,
        x_rec: Tensor,
        mask: Tensor,
    ) -> Tensor:
        target = self.patchify(x)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss_rec = (target - x_rec) ** 2

        loss_rec_masked = torch.masked_select(loss_rec, mask.bool())

        return loss_rec_masked.mean()

    def forward(self, x: Tensor, ssl_phase: str) -> tuple[Tensor, Tensor, Tensor]:
        mask_s1, mask_s2, mask_s3, mask_s4 = self.random_masking(
            x,
            ssl_phase,
        )
        B, C, H, W = x["image"].shape  # noqa: N806

        z = self.forward_encoder(
            x["image"],
            mask_s1,
            mask_s2,
            mask_s3,
            mask_s4,
            ssl_phase,
        )

        if ssl_phase == "pretrain":
            mask_s1 = self.random_unmasking(mask_s1, B)
            mask_s4 = self.random_unmasking(mask_s4, B)

        x_rec = self.forward_decoder(z, mask_s4, ssl_phase)

        loss = self.forward_loss(x["image"], x_rec, mask_s4)

        # masked
        mask_input = mask_s1.view(
            mask_s1.shape[0],
            1,
            H // self.encoder_stride * 8,
            W // self.encoder_stride * 8,
        )
        mask_input = torch.nn.functional.interpolate(
            mask_input,
            size=(H, W),
            mode="nearest",
        )

        masked = torch.where(mask_input == 0, x["image"], 1)

        # pred
        if ssl_phase == "pretrain":
            mask_output = mask_s4.view(
                B,
                1,
                H // self.encoder_stride,
                W // self.encoder_stride,
            )
        else:
            mask_output = mask_s4.view(
                1,
                1,
                H // self.encoder_stride,
                W // self.encoder_stride,
            )

        mask_output = torch.nn.functional.interpolate(
            mask_output,
            size=(H, W),
            mode="nearest",
        )

        x_rec = rearrange(
            x_rec,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=H // int((x_rec.shape[2] / C) ** 0.5),
            p1=int((x_rec.shape[2] / C) ** 0.5),
            p2=int((x_rec.shape[2] / C) ** 0.5),
        )

        pred = x_rec * mask_output + x["image"] * (1 - mask_output)

        # head logits
        head_logits = {
            dataset: self.head[dataset](z, ssl_phase) for dataset in self.num_classes
        }
        return loss, masked, pred, head_logits


def mixmae_tiny(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE tiny."""
    args = {
        "image_size": 256,
        "patch_size": 4,
        "in_chans": 3,
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": [8, 8, 8, 8],
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "use_checkpoint": False,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
    }
    args.update(kwargs)
    return MixMAE(**args)


def mixmae_small(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE small."""
    args = {
        "image_size": kwargs["image_size"],
        "patch_size": 4,
        "in_chans": 3,
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": [8, 8, 8, 8],
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "use_checkpoint": False,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
    }
    args.update(kwargs)
    return MixMAE(**args)


def mixmae_base(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE base."""
    args = {
        "image_size": kwargs["image_size"],
        "patch_size": 4,
        "in_chans": 3,
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "window_size": [8, 8, 8, 8],
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "use_checkpoint": False,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
    }
    args.update(kwargs)
    return MixMAE(**args)


def mixmae_large(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE large."""
    args = {
        "image_size": 224,
        "patch_size": 4,
        "in_chans": 3,
        "embed_dim": 192,
        "depths": [2, 2, 18, 2],
        "num_heads": [6, 12, 24, 48],
        "window_size": [8, 8, 8, 8],
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "use_checkpoint": False,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
    }
    args.update(kwargs)
    return MixMAE(**args)


def mixmae_huge(**kwargs) -> MixMAE:  # noqa: ANN003
    """Construct MixMAE huge."""
    args = {
        "image_size": 224,
        "patch_size": 4,
        "in_chans": 3,
        "embed_dim": 352,
        "depths": [2, 2, 18, 2],
        "num_heads": [11, 22, 44, 88],
        "window_size": [8, 8, 8, 8],
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "patch_norm": True,
        "use_checkpoint": False,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
    }
    args.update(kwargs)
    return MixMAE(**args)
