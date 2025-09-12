# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import Literal

import numpy as np

import timm.models.vision_transformer
import torch
from torch import Tensor, nn

from conf.datasets import DatasetsConfig
from ssl_models.baselines.base import BaseModule
from ssl_models.baselines.utils import interpolate_pos_encoding

class SatMAEBaseline(BaseModule):
    """Baseline model based on SatMAE."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        unpool_dim: int,
        backbone_size: str = "base",
        freeze: bool = False,
        pretrained_path: str | None = None,
        type_head: Literal["linear", "attentive"] = "linear",
        multimodal: Literal["mod"] = "mod",
        add_date_enc: bool = False,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        keep_norm: bool = True,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Create the SatMAE encoder model.

        Parameters
        ----------
        backbone_size: str
            Defines the backbone to use. To choose in "base", "large".
        freeze: bool
            To freeze or not to freeze the backbone.
        global_pool: bool
            Whether to use global pooling (default is False, i.e., use CLS token).
        pretrained_path: str
           Path to the location of the pretrained weights.

        """
        self.datasets = datasets
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

        self.patch_size = 16
        self.encoder_dim = 768 if backbone_size == "base" else 1024

        super().__init__(
            self.datasets,
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
        self.image_size = datasets.dataset.inputs["s2"].image_size
        mean_pretrain = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
        std_pretrain  = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
        self.mean = torch.tensor(mean_pretrain).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std_pretrain).view(1, 1, 3, 1, 1)

        self._build_backbone(backbone_size)

        if pretrained_path is not None:
            self._load_weights(pretrained_path=pretrained_path)

        if freeze:
            self.freeze_backbone()

    def _load_weights(self, pretrained_path: str) -> None:
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)["model"]

        if self.image_size != 224:
            interpolated_pos_embed = interpolate_pos_encoding(
                state_dict["pos_embed"],
                self.image_size,
                self.image_size,
                self.patch_size,
            )
            interpolated_decoder_pos_embed = interpolate_pos_encoding(
                state_dict["decoder_pos_embed"],
                self.image_size,
                self.image_size,
                self.patch_size,
            )
            state_dict["decoder_pos_embed"] = torch.cat(interpolated_decoder_pos_embed, dim=1) #fuze cls and pos embed
            state_dict["pos_embed"] = torch.cat(interpolated_pos_embed, dim=1) #fuze cls and pos embed
        else:
            state_dict["pos_embed"]=state_dict["pos_embed"]
            state_dict["decoder_pos_embed"]=state_dict["decoder_pos_embed"]

        self.encoder.load_state_dict(state_dict, strict=True)
        print(f"Loaded pretrained weights from {pretrained_path}")
        del state_dict
        torch.cuda.empty_cache()

    def _build_backbone(self, backbone_size) -> None:
        if backbone_size == "base":
            self.encoder = mae_vit_base_patch16_dec512d8b(img_size=self.image_size)
        elif backbone_size == "large":
            self.encoder = mae_vit_large_patch16_dec512d8b(img_size=self.image_size)
        else:
            raise ValueError(f"Backbone size {backbone_size} not recognized.")


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

    def freeze_backbone(self) -> None:
        """Freeze the backbone's parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def format_dates(self, dates: Tensor) -> Tensor:
        year = dates[:, :, 0, 0, 0]
        month = dates[:, :, 1, 0, 0]
        hour = dates[:, :, 2, 0, 0]

        return torch.stack([year, month, hour], dim=-1)

    def forward(
        self,
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Forward pass of the model.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (B, C, H, W).
        timestamps: torch.Tensor
            Timestamps tensor of shape (B, 3).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, embed_dim).

        """
        s2_input = batch["s2"] - self.mean.to(batch["s2"].device) / self.std.to(batch["s2"].device)
        timestamps = batch["s2_dates"]

        x = {}

        x["s2"] = self.encoder.forward_encoder(s2_input, self.format_dates(timestamps), 0)

        loss_pred, log_input, log_pred, log_target = self.compute_loss_pred(
            x,
            batch,
            ssl_phase,
            stage,
        )
        self.metrics[f"loss_pred_{stage}"].update(loss_pred)
        return None, log_input, log_pred, log_target, loss_pred

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, same_mask=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed((img_size,img_size), patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - 384), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim - 192), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.same_mask = same_mask
        self.initialize_weights()
        self.counter = 0

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, mask=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        if self.same_mask:
            L2 = L // 3
            assert 3 * L2 == L
            noise = torch.rand(N, L2, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_shuffle = [ids_shuffle + i * L2 for i in range(3)]
            ids_shuffle_keep = [z[: ,:int(L2 * (1 - mask_ratio))] for z in ids_shuffle]
            ids_shuffle_disc = [z[: ,int(L2 * (1 - mask_ratio)):] for z in ids_shuffle]
            ids_shuffle = []
            for z in ids_shuffle_keep:
                ids_shuffle.append(z)
            for z in ids_shuffle_disc:
                ids_shuffle.append(z)
            ids_shuffle = torch.cat(ids_shuffle, dim=1)
            # print(ids_shuffle[0])
            # assert False
        else:
            if mask is None:
                # sort noise for each sample
                ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            else:
                ids_shuffle = mask
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, timestamps, mask_ratio, mask=None):
        # embed patches
        x1 = self.patch_embed(x[:, 0])
        x2 = self.patch_embed(x[:, 1])
        x3 = self.patch_embed(x[:, 2])
        x = torch.cat([x1, x2, x3], dim=1)

        # print(timestamps.shape, x.shape)
        ts_embed = torch.cat([get_1d_sincos_pos_embed_from_grid_torch(128, timestamps.reshape(-1, 3)[:, 0].float()),
                   get_1d_sincos_pos_embed_from_grid_torch(128, timestamps.reshape(-1, 3)[:, 1].float()),
                   get_1d_sincos_pos_embed_from_grid_torch(128, timestamps.reshape(-1, 3)[:, 2].float())], dim=1).float()
        # print(ts_embed, ts_embed.shape)
        ts_embed = ts_embed.reshape(-1, 3, ts_embed.shape[-1]).unsqueeze(2)
        # print(ts_embed.shape)
        ts_embed = ts_embed.expand(-1, -1, x.shape[1] // 3, -1).reshape(x.shape[0], -1, ts_embed.shape[-1])
        # print(ts_embed.shape)
        # ts_embed = torch.zeros_like(ts_embed)


        # add pos embed w/o cls token
        x = x + torch.cat([self.pos_embed[:, 1:, :].repeat(ts_embed.shape[0], 3, 1), ts_embed], dim=-1)

        # masking: length -> length * mask_ratio
        #x, mask, ids_restore = self.random_masking(x, mask_ratio, mask=mask)

        # append cls token
        cls_token = self.cls_token #+ self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print(x.dtype)

        # apply Transformer blocks
        for blk in self.blocks:
            # print(x.dtype)
            x = blk(x)
        x = self.norm(x)

        return x[:,1:,:]#, mask, ids_restore

    def forward(self, imgs, timestamps, mask_ratio=0.75, mask=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, timestamps, mask_ratio, mask=mask)
        pred = self.forward_decoder(latent, timestamps, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=float, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb