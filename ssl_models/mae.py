"""MAE module. Adapted from clay foundation model: https://github.com/Clay-foundation/model."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, reduce, repeat
from torch import Tensor, nn
from vit_pytorch.vit import Transformer

from ssl_models.geopile_data import GeoPileDataset
from ssl_models.head import AttentiveHead, LinearHead
from ssl_models.utils import posemb_sincos_1d, posemb_sincos_2d


class Patchify(nn.Module):
    """Patchify the input cube & create embeddings per patch."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm_skip = nn.GroupNorm(1, embed_dim)
        if self.unpool:
            channels = []
            for idx in range(self.num_downsampling):
                pow_in = idx
                pow_out = idx + 1
                channels_in = 4 * 2**pow_in if idx > 0 else in_chans
                channels_out = 4 * 2**pow_out
                channels.append((channels_in, channels_out))

            self.convs = nn.ModuleDict(
                [
                    [
                        f"conv_{idx}",
                        nn.Conv2d(
                            channels_in,
                            channels_out,
                            kernel_size=2,
                            padding_mode="reflect",
                            padding="same",
                        ),
                    ]
                    for idx, (channels_in, channels_out) in enumerate(channels)
                ]
            )
            self.norms = nn.ModuleDict(
                [
                    [f"norm_{idx}", nn.GroupNorm(1, channels_out)]
                    for idx, (_, channels_out) in enumerate(channels)
                ]
            )
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.conv = nn.Conv2d(
                4 * 2**self.num_downsampling,
                embed_dim,
                kernel_size=1,
            )
            self.norm = nn.GroupNorm(1, embed_dim)

    def forward(self, xb: Tensor) -> Tensor:
        B, C, H, W = xb.shape  # noqa: N806
        xb = self.proj(xb)
        xb = rearrange(xb, "B D P1 P2 -> B (P1 P2) D")
        return self.norm(xb)


class Encoder(nn.Module):
    """MAE encoder."""

    def __init__(
        self,
        mask_ratio: float,
        image_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
        band_groups: dict,
        dropout: float,
        emb_dropout: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.band_groups = band_groups
        self.num_spatial_patches = (image_size // patch_size) ** 2
        self.num_group_patches = len(band_groups)
        self.num_patches = self.num_spatial_patches * self.num_group_patches

        # Split the embedding dimensions between spatial & band patches equally
        pos_dim = band_dim = dim // 2
        self.patch_embedding = nn.ModuleDict(
            {
                name: Patchify(len(bands), dim, patch_size, num_downsampling, unpool)
                for name, bands in self.band_groups.items()
            },
        )

        # Fix the position & band embedding to sine & cosine functions
        self.register_buffer(
            name="pos_encoding",
            tensor=posemb_sincos_2d(
                h=image_size // patch_size,
                w=image_size // patch_size,
                dim=pos_dim,
            ),  # [L D/2]
            persistent=False,
        )
        self.register_buffer(
            name="band_encoding",
            tensor=posemb_sincos_1d(
                length=self.num_group_patches,
                dim=band_dim,
            ),  # [G D/2]
            persistent=False,
        )

        # Freeze the weights of position & band encoding
        self.pos_encoding = self.pos_encoding.requires_grad_(requires_grad=False)
        self.band_encoding = self.band_encoding.requires_grad_(requires_grad=False)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim * mlp_ratio,
            dropout=dropout,
        )

    def to_patch_embed(self, cube: Tensor) -> Tensor:
        """Patchify the input cube & create embeddings per patch."""
        patches = []
        embed_inds = {}
        for name, bands in self.band_groups.items():
            cube_slice = cube[:, bands, :, :]  # [B C H W] -> [B C[slice[...]] H W]
            embed_slice, embed_inds[name] = self.patch_embedding[name](cube_slice)
            patches.append(embed_slice)

        return rearrange(patches, "G B L D -> B G L D")  # [B G L D]

    def add_encodings(self, patches: Tensor) -> Tensor:
        """Add position & band encoding to the patches."""
        B, G, L, D = patches.shape  # noqa: N806

        # Align position & band embeddings across patches
        pos_encoding = repeat(
            self.pos_encoding,
            "L D -> 1 repeat L D",
            repeat=G,
        )  # [1 G L D/2]

        band_encoding = repeat(
            self.band_encoding,
            "G D -> 1 G repeat D",
            repeat=L,
        )  # [1 G L D/2]

        pos_band_encoding = torch.cat(
            (pos_encoding, band_encoding),
            dim=-1,
        )  # [1 G L D]

        # Add position & band encoding to the input feature vector
        patches = patches + pos_band_encoding  # [B G L D] + [1 G L D] - broadcasting
        return self.dropout(patches)  # [B G L D]

    def mask_out(
        self,
        patches: Tensor,
        ssl_phase: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Mask out patches randomly.

        First shuffle the patches and then mask out the first N patches.
        """
        B, GL, D = patches.shape  # noqa: N806
        if self.num_patches != GL:
            raise ValueError

        # Shuffle the patches
        noise = torch.randn((B, GL), device=patches.device)  # [B GL]

        random_indices = torch.argsort(noise, dim=-1)  # [B GL]
        reverse_indices = torch.argsort(random_indices, dim=-1)  # [B GL]
        if ssl_phase == "pretrain":
            num_masked_patches = int(
                self.mask_ratio * self.num_patches,
            )  # Number of patches to be masked out
        else:
            num_masked_patches = 0
        masked_indices, unmasked_indices = (
            random_indices[:, :num_masked_patches],  # [B mask_ratio * GL]
            random_indices[:, num_masked_patches:],  # [B (1 - mask_ratio) * GL]
        )

        # create a mask of shape B G L, where 1 indicates a masked patch
        # and 0 indicates an unmasked patch
        masked_matrix = torch.zeros((B, GL), device=patches.device)  # [B GL] = 0
        masked_matrix[:, :num_masked_patches] = 1  # [B mask_ratio * GL] = 1
        masked_matrix = torch.gather(
            masked_matrix,
            dim=1,
            index=reverse_indices,
        )  # [B GL] -> [B GL] - reorder the patches
        masked_matrix = rearrange(
            masked_matrix,
            "B (G L) -> B G L",
            G=self.num_group_patches,  # [B G L]
        )

        # mask out the patches
        batch_indices = rearrange(
            torch.arange(B, device=patches.device),
            "B -> B 1",
        )  # [B 1]
        unmasked_patches = patches[
            batch_indices,
            unmasked_indices,
            :,
        ]  # [B GL:(1 - mask_ratio) D]
        _ = patches[batch_indices, masked_indices, :]  # [B GL:mask_ratio D]

        return (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B GL:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B G L]

    def forward(
        self,
        cube: Tensor,
        ssl_phase: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        B, C, H, W = cube["image"].shape  # noqa: N806

        patches = self.to_patch_embed(
            cube["image"],
        )  # [B G L D] - patchify & create embeddings per patch

        patches = self.add_encodings(
            patches,
        )  # [B G L D] - add position & band encoding to the embeddings

        patches = rearrange(patches, "B G L D -> B (G L) D")  # [B (GL) D]
        patches = self.dropout(patches)  # [B (GL) D]

        # mask out patches
        (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self.mask_out(
            patches,
            ssl_phase,
        )  # [B GL:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B G L]

        # pass the unmasked patches through the transformer
        encoded_unmasked_patches = self.transformer(
            unmasked_patches,
        )  # [B (GL:(1 - mask_ratio)) D]

        return (
            encoded_unmasked_patches,
            embed_inds,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B (GL:(1 - mask_ratio)) D], [(1-mask_ratio)], [mask_ratio], [B G L]


class Decoder(nn.Module):
    """MAE decoder."""

    def __init__(
        self,
        mask_ratio: float,
        image_size: int,
        patch_size: int,
        encoder_dim: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
        band_groups: dict,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.dim = dim
        self.band_groups = band_groups
        self.num_spatial_patches = (image_size // patch_size) ** 2
        self.num_group_patches = len(band_groups)
        self.num_patches = self.num_spatial_patches * self.num_group_patches

        self.enc_to_dec = (
            nn.Linear(encoder_dim, dim) if encoder_dim != dim else nn.Identity()
        )
        self.mask_patch = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim * mlp_ratio,
            dropout=dropout,
        )

        # Split the embedding dimensions between spatial & band patches equally
        pos_dim = band_dim = dim // 2

        # Fix the position & band embedding to sine & cosine functions
        self.register_buffer(
            name="pos_encoding",
            tensor=posemb_sincos_2d(
                h=image_size // patch_size,
                w=image_size // patch_size,
                dim=pos_dim,
            ),  # [L D/2]
            persistent=False,
        )
        self.register_buffer(
            name="band_encoding",
            tensor=posemb_sincos_1d(
                length=self.num_group_patches,
                dim=band_dim,
            ),  # [G D/2]
            persistent=False,
        )

        # Freeze the weights of position & band encoding
        self.pos_encoding = self.pos_encoding.requires_grad_(requires_grad=False)
        self.band_encoding = self.band_encoding.requires_grad_(requires_grad=False)

        self.embed_to_pixels = nn.ModuleDict(
            {
                name: Pixelify(dim, len(bands), patch_size, num_downsampling, unpool)
                for name, bands in self.band_groups.items()
            },
        )

    def reconstruct_and_add_encoding(
        self,
        unmasked_patches: Tensor,
        unmasked_indices: Tensor,
        masked_indices: Tensor,
        ssl_phase: str,
    ) -> Tensor:
        """Reconstruct input patches.

        Reconstruction is based on the combination of:
            - unmasked patches
            - masked patches
            - position & band encoding
        """
        B, _, _ = unmasked_patches.shape  # noqa: N806

        # Align position & band embeddings across patches
        pos_encoding = repeat(
            self.pos_encoding,
            "L D -> 1 repeat L D",
            repeat=self.num_group_patches,
        )  # [1 G L D/2]
        band_encoding = repeat(
            self.band_encoding,
            "G D -> 1 G repeat D",
            repeat=self.num_spatial_patches,
        )  # [1 G L D/2]

        pos_band_encoding = torch.cat(
            (pos_encoding, band_encoding),
            dim=-1,
        )  # [1 G L D]
        pos_band_encoding = rearrange(
            pos_band_encoding,
            "1 G L D -> 1 (G L) D",
        )  # [1 (GL) D]
        pos_band_encoding = repeat(
            pos_band_encoding,
            "1 (GL) D -> B (GL) D",
            B=B,
        )  # [B (GL) D]

        batch_indices = rearrange(
            torch.arange(B, device=unmasked_patches.device),
            "B -> B 1",
        )  # [B 1]
        unmasked_pos_band_encoding = pos_band_encoding[
            batch_indices,
            unmasked_indices,
            :,
        ]  # [B (GL:(1 - mask_ratio)) D]
        masked_pos_band_encoding = pos_band_encoding[
            batch_indices,
            masked_indices,
            :,
        ]  # [B (GL:mask_ratio) D]

        # Reconstruct the masked patches from the random mask patch &
        # add position & band encoding to them
        if ssl_phase == "pretrain":
            num_masked_patches = int(self.mask_ratio * self.num_patches)
        else:
            num_masked_patches = 0
        masked_patches = repeat(
            self.mask_patch,
            "D -> B GL D",
            B=B,
            GL=num_masked_patches,
        )  # [B GL:mask_ratio D]
        masked_patches = (
            masked_patches + masked_pos_band_encoding
        )  # [B GL:mask_ratio D] + [B GL:mask_ratio D]

        # Add position & band encoding to the unmasked patches
        unmasked_patches = (
            unmasked_patches + unmasked_pos_band_encoding
        )  # [B GL:(1 - masked_ratio) D] + [B GL:(1 - mask_ratio) D]

        # Concatenate the masked & unmasked patches
        decoder_patches = torch.zeros(
            (B, self.num_patches, self.dim),
            device=unmasked_patches.device,
        )  # [B GL D]
        decoder_patches[
            batch_indices,
            unmasked_indices,
            :,
        ] = unmasked_patches  # [B GL:(1 - mask_ratio) D]
        decoder_patches[
            batch_indices,
            masked_indices,
            :,
        ] = masked_patches  # [B GL:mask_ratio D]

        return decoder_patches  # [B GL D]

    def pixelify(self, patches: Tensor) -> Tensor:
        """Convert the patches into pixel space to compute the loss."""
        patches = rearrange(
            patches,
            "B (G L) D -> B G L D",
            G=len(self.band_groups),
        )  # [B G L D]
        pixels = []
        for i, (name, _) in enumerate(self.band_groups.items()):
            group_embeddings = patches[:, i, :, :]  # [B L D]
            group_pixels = self.embed_to_pixels[name](
                group_embeddings, embed_inds[name]
            )  # [B L (P P C)]
            group_pixels = rearrange(
                group_pixels,
                "B L (PP C) -> B C L PP",
                PP=(self.patch_size**2),
            )  # [B C L PP]
            pixels.append(group_pixels)  # [B C L PP]

        return torch.cat(pixels, dim=1)  # [B C L PP]

    def forward(
        self,
        encoded_unmasked_patches: Tensor,
        unmasked_indices: Tensor,
        masked_indices: Tensor,
        ssl_phase: str,
    ) -> Tensor:
        # Change the embedding dimension from encoder to decoder
        encoded_unmasked_patches = self.enc_to_dec(encoded_unmasked_patches)

        # Reconstruct the patches to feed into the decoder transformer
        decoder_patches = self.reconstruct_and_add_encoding(
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            ssl_phase,
        )  # [B GL D]

        # Pass the decoder patches through the transformer
        decoded_patches = self.transformer(decoder_patches)  # [B GL D]

        # Pixelify the decoded patches
        return self.pixelify(decoded_patches)  # [B C L PP]


class MAE(nn.Module):
    """Masked Auto Encoder (MAE)."""

    def __init__(  # noqa: PLR0913
        self,
        mask_ratio: float,
        image_size: int,
        patch_size: int,
        # encoder
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
        dropout: float,
        emb_dropout: float,
        # decoder
        decoder_dim: int,
        decoder_depth: int,
        decoder_heads: int,
        decoder_dim_head: int,
        decoder_mlp_ratio: float,
        decoder_dropout: float,
        band_groups: dict,
        type_head: str = "attentive",
        # band groups
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.band_groups = band_groups

        self.encoder = Encoder(
            mask_ratio=mask_ratio,
            image_size=image_size,
            patch_size=patch_size,
            num_downsampling=num_downsampling,
            unpool=unpool,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            band_groups=band_groups,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        self.decoder = Decoder(
            mask_ratio=mask_ratio,
            image_size=image_size,
            patch_size=patch_size,
            num_downsampling=num_downsampling,
            unpool=unpool,
            encoder_dim=dim,
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_ratio=decoder_mlp_ratio,
            band_groups=band_groups,
            dropout=decoder_dropout,
        )

        self.num_classes = GeoPileDataset.num_classes
        self.head = nn.ModuleDict(
            [
                [
                    dataset,
                    AttentiveHead(dim, num_classes)
                    if type_head == "attentive"
                    else LinearHead(dim, num_classes),
                ]
                for dataset, num_classes in self.num_classes.items()
            ],
        )

    def per_pixel_loss(
        self,
        cube: Tensor,
        pixels: Tensor,
        masked_matrix: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the per pixel loss."""
        patches = rearrange(
            cube,
            "B C (H P1) (W P2) -> B C (H W) (P1 P2)",
            P1=self.patch_size,
            P2=self.patch_size,
        )  # [B C L PP]

        loss = F.l1_loss(patches, pixels, reduction="none")  # loss per pixel
        loss = reduce(loss, "B C L PP -> B C L", reduction="mean")  # loss per patch

        # mask out the loss for unmasked patches
        actual_loss, masked_patches = 0.0, 0.0
        for i, (_, group) in enumerate(self.band_groups.items()):
            group_loss = reduce(
                loss[:, group, :],
                "B G L -> B L",
                "mean",
            )  # (B, L) - loss per group
            actual_loss += (
                group_loss * masked_matrix[:, i]
            ).sum()  # (B, L) * (B, L) -> (B, L) -> (B) -> scalar
            masked_patches += masked_matrix[:, i].sum()  # (B, L) -> (B) -> scalar

        masked = torch.zeros(patches.shape, device=patches.device)
        pred = torch.zeros(patches.shape, device=patches.device)
        for i, (_, group) in enumerate(self.band_groups.items()):
            masked[:, group, :, :] = 1 * masked_matrix[:, None, i, :, None] + patches[
                :,
                group,
                :,
                :,
            ] * (1 - masked_matrix[:, None, i, :, None])
            pred[:, group, :, :] = pixels[:, group, :, :] * masked_matrix[
                :,
                None,
                i,
                :,
                None,
            ] + patches[:, group, :, :] * (1 - masked_matrix[:, None, i, :, None])

        masked = rearrange(
            masked,
            "B C (H W) (P1 P2) -> B C (H P1) (W P2)",
            H=self.image_size // self.patch_size,
            P1=self.patch_size,
        )
        pred = rearrange(
            pred,
            "B C (H W) (P1 P2) -> B C (H P1) (W P2)",
            H=self.image_size // self.patch_size,
            P1=self.patch_size,
        )

        return actual_loss / masked_patches, masked, pred

    def forward(
        self,
        cube: Tensor,
        ssl_phase: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # encoder
        (
            encoded_unmasked_patches,
            embed_inds,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self.encoder(
            cube,
            ssl_phase,
        )  # [B (GL:(1 - mask_ratio)) D], [(1-mask_ratio)], [mask_ratio], [B G L]

        # decoder
        pixels = self.decoder(
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            ssl_phase,
        )  # [B C L PP]

        # loss
        rec_loss, masked, pred = self.per_pixel_loss(
            cube["image"],
            pixels,
            masked_matrix,
        )
        head_logits = {
            dataset: self.head[dataset](encoded_unmasked_patches, ssl_phase)
            for dataset in self.num_classes
        }
        return rec_loss, masked, pred, head_logits


def mae_tiny(**kwargs) -> MAE:  # noqa: ANN003
    """MAE tiny."""
    args = {
        # encoder
        "dim": 128,
        "depth": 4,
        "heads": 4,
        "dim_head": 64,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        # decoder
        "decoder_dim": 64,
        "decoder_depth": 2,
        "decoder_heads": 2,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 2,
        "decoder_dropout": 0.0,
    }
    args.update(kwargs)
    return MAE(**args)


def mae_small(**kwargs) -> MAE:  # noqa: ANN003
    """MAE small."""
    args = {
        # encoder
        "dim": 512,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "mlp_ratio": 4,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        # decoder
        "decoder_dim": 256,
        "decoder_depth": 8,
        "decoder_heads": 8,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
        "decoder_dropout": 0.0,
    }
    args.update(kwargs)
    return MAE(**args)


def mae_medium(**kwargs) -> MAE:  # noqa: ANN003
    """MAE base."""
    args = {
        # encoder
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        # decoder
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_heads": 16,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
        "decoder_dropout": 0.0,
    }
    args.update(kwargs)
    return MAE(**args)


def mae_large(**kwargs) -> MAE:  # noqa: ANN003
    """MAE large."""
    args = {
        # encoder
        "dim": 1280,
        "depth": 32,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        # decoder
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_heads": 16,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
        "decoder_dropout": 0.0,
    }
    args.update(kwargs)
    return MAE(**args)
