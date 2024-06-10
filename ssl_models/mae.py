"""
Code adapted from Clay Foundation Model: https://github.com/Clay-foundation/model
"""

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn
from vit_pytorch.vit import Transformer

from ssl_models.geopile_data import GeoPileDataset
from ssl_models.probing import AttentiveProbing, LinearProbing
from ssl_models.utils import posemb_sincos_1d, posemb_sincos_2d


class Patchify(nn.Module):
    """
    Patchify the input cube & create embeddings per patch
    """

    def __init__(self, in_chans, embed_dim, patch_size, num_downsampling, unpool):
        """
        Define layers of patch stem.

        Parameters
        ----------
        in_chans : int
            Number of input channels
        embed_dim : int
            Embedding dimension
        patch_size : int
            Patch size
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_downsampling = num_downsampling
        self.unpool = unpool

        self.conv_skip = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
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

    def forward(self, xb):
        _, _, h, w = xb.shape
        skip = xb
        skip = self.conv_skip(skip)
        skip = self.norm_skip(skip)
        skip = rearrange(
            skip,
            "b c h w -> (b h w) c 1 1",
        )

        inds = []
        if self.unpool:
            xb = rearrange(
                xb,
                "b c (h p1) (w p2) -> (b h w) c p1 p2",
                p1=self.patch_size,
                p2=self.patch_size,
            )
            for idx in range(self.num_downsampling):
                xb = self.convs[f"conv_{idx}"](xb)
                xb = self.norms[f"norm_{idx}"](xb)
                xb, ind = self.max_pool(xb)
                inds.append(ind)

            xb = self.conv(xb)
            xb = self.norm(xb)
            xb = xb + skip
        else:
            xb = skip
        xb = rearrange(
            xb,
            "(b h w) d 1 1 -> b (h w) d",
            h=h // self.patch_size,
            w=w // self.patch_size,
        )
        return xb, inds


class Pixelify(nn.Module):
    """
    Pixelify the patches
    """

    def __init__(self, embed_dim, out_chans, patch_size, num_downsampling, unpool):
        """ """
        super().__init__()
        self.patch_size = patch_size
        self.num_downsampling = num_downsampling
        self.unpool = unpool

        self.conv_skip = nn.Conv2d(
            embed_dim,
            out_chans * self.patch_size**2,
            kernel_size=1,
        )
        if self.unpool:
            channels = []
            for idx in range(self.num_downsampling):
                pow_in = self.num_downsampling - idx
                pow_out = self.num_downsampling - (idx + 1)
                channels_in = 4 * 2**pow_in
                channels_out = 4 * 2**pow_out
                channels.append((channels_in, channels_out))

            self.conv = nn.Conv2d(
                embed_dim,
                4 * 2**num_downsampling,
                kernel_size=1,
            )
            self.norm = nn.GroupNorm(1, 4 * 2**num_downsampling)
            self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
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
            self.fc = nn.Conv2d(4, out_chans, kernel_size=1)

    def forward(self, xb, inds):
        b, _, _ = xb.shape
        xb = rearrange(
            xb,
            "b l c -> (b l) c 1 1",
        )
        skip = xb
        skip = self.conv_skip(skip)
        skip = rearrange(
            skip, "b (p1 p2 c) 1 1 -> b c p1 p2", p1=self.patch_size, p2=self.patch_size
        )
        if self.unpool:
            xb = self.conv(xb)
            xb = self.norm(xb)
            for idx, ind in enumerate(inds[::-1]):
                xb = self.max_unpool(xb, ind)
                xb = self.convs[f"conv_{idx}"](xb)
                xb = self.norms[f"norm_{idx}"](xb)
            xb = self.fc(xb)
            xb = xb + skip
        else:
            xb = skip

        xb = rearrange(xb, "(b l) c p1 p2 -> b l (p1 p2 c)", b=b)
        return xb


class Encoder(nn.Module):
    def __init__(
        self,
        mask_ratio,
        image_size,
        patch_size,
        num_downsampling,
        unpool,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        band_groups,
        dropout,
        emb_dropout,
        pretraining: bool = True,
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.band_groups = band_groups
        self.num_spatial_patches = (image_size // patch_size) ** 2
        self.num_group_patches = len(band_groups)
        self.num_patches = self.num_spatial_patches * self.num_group_patches
        self.pretraining = pretraining

        # Split the embedding dimensions between spatial & band patches equally
        pos_dim = band_dim = dim // 2
        self.patch_embedding = nn.ModuleDict(
            {
                name: Patchify(len(bands), dim, patch_size, num_downsampling, unpool)
                for name, bands in self.band_groups.items()
            }
        )

        # Fix the position & band embedding to sine & cosine functions
        self.register_buffer(
            name="pos_encoding",
            tensor=posemb_sincos_2d(
                h=image_size // patch_size, w=image_size // patch_size, dim=pos_dim
            ),  # [L D/2]
            persistent=False,
        )
        self.register_buffer(
            name="band_encoding",
            tensor=posemb_sincos_1d(
                length=self.num_group_patches, dim=band_dim
            ),  # [G D/2]
            persistent=False,
        )

        # Freeze the weights of position & band encoding
        self.pos_encoding = self.pos_encoding.requires_grad_(False)
        self.band_encoding = self.band_encoding.requires_grad_(False)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim * mlp_ratio,
            dropout=dropout,
        )

    def to_patch_embed(self, cube):
        """
        Patchify the input cube & create embeddings per patch

        Parameters
        ----------
        cube : torch.Tensor
            A tensor of shape (B, C, H, W) containing the pixels of the
            cube.

        Returns
        -------
        patches : torch.Tensor
            A tensor of shape (B, G, L, D) containing the embeddings of the
            patches.
        """
        patches = []
        embed_inds = {}
        for name, bands in self.band_groups.items():
            cube_slice = cube[:, bands, :, :]  # [B C H W] -> [B C[slice[...]] H W]
            embed_slice, embed_inds[name] = self.patch_embedding[name](cube_slice)
            patches.append(embed_slice)

        patches = rearrange(patches, "G B L D -> B G L D")  # [B G L D]
        return patches, embed_inds  # [B G L D]

    def add_encodings(self, patches):
        """
        Add position & band encoding to the patches

        Parameters
        ----------
        patches : torch.Tensor
            A tensor of shape (B, G, L, D) containing the embeddings of the
            patches.

        Returns
        -------
        patches : torch.Tensor
            A tensor of shape (B, G, L, D) containing the embeddings of the
            patches + position & band encoding.
        """
        self.B, G, L, D = patches.shape

        # Align position & band embeddings across patches
        pos_encoding = repeat(
            self.pos_encoding, "L D -> 1 repeat L D", repeat=G
        )  # [1 G L D/2]

        band_encoding = repeat(
            self.band_encoding, "G D -> 1 G repeat D", repeat=L
        )  # [1 G L D/2]

        pos_band_encoding = torch.cat(
            (pos_encoding, band_encoding), dim=-1
        )  # [1 G L D]

        # Add position & band encoding to the input feature vector
        patches = patches + pos_band_encoding  # [B G L D] + [1 G L D] - broadcasting
        patches = self.dropout(patches)  # [B G L D]
        return patches  # [B G L D]

    def mask_out(self, patches):
        """
        Mask out patches randomly by shuffling the patches & masking out the
        first N patches

        Parameters
        ----------
        patches : torch.Tensor
            A tensor of shape (B, GL, D) containing the embeddings of the
            patches.

        Returns
        -------
        unmasked_patches : torch.Tensor
            A tensor of shape (B, GL:(1 - mask_ratio), D) containing the
            embeddings of the unmasked patches.
        unmasked_indices : torch.Tensor
            A tensor of shape (B, (1 - mask_ratio)) containing the indices of
            the unmasked patches.
        masked_indices : torch.Tensor
            A tensor of shape (B, mask_ratio) containing the indices of the
            masked patches.
        masked_matrix : torch.Tensor
            A tensor of shape (B, G, L) containing the mask matrix.
        """
        B, GL, D = patches.shape
        assert (
            GL == self.num_patches
        ), f"Expected {self.num_patches} patches, got {GL} patches."

        # Shuffle the patches
        noise = torch.randn((B, GL), device=patches.device)  # [B GL]

        random_indices = torch.argsort(noise, dim=-1)  # [B GL]
        reverse_indices = torch.argsort(random_indices, dim=-1)  # [B GL]
        if self.pretraining:
            num_masked_patches = int(
                self.mask_ratio * self.num_patches
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
            masked_matrix, dim=1, index=reverse_indices
        )  # [B GL] -> [B GL] - reorder the patches
        masked_matrix = rearrange(
            masked_matrix,
            "B (G L) -> B G L",
            G=self.num_group_patches,  # [B G L]
        )

        # mask out the patches
        batch_indices = rearrange(
            torch.arange(B, device=patches.device), "B -> B 1"
        )  # [B 1]
        unmasked_patches = patches[
            batch_indices, unmasked_indices, :
        ]  # [B GL:(1 - mask_ratio) D]
        _ = patches[batch_indices, masked_indices, :]  # [B GL:mask_ratio D]

        return (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B GL:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B G L]

    def forward(self, cube):
        B, C, H, W = cube["image"].shape

        patches, embed_inds = self.to_patch_embed(
            cube["image"]
        )  # [B G L D] - patchify & create embeddings per patch

        patches = self.add_encodings(
            patches
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
            patches
        )  # [B GL:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B G L]

        # pass the unmasked patches through the transformer
        encoded_unmasked_patches = self.transformer(
            unmasked_patches
        )  # [B (GL:(1 - mask_ratio)) D]

        return (
            encoded_unmasked_patches,
            embed_inds,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B (GL:(1 - mask_ratio)) D], [(1-mask_ratio)], [mask_ratio], [B G L]


class Decoder(nn.Module):
    def __init__(
        self,
        mask_ratio,
        image_size,
        patch_size,
        num_downsampling,
        unpool,
        encoder_dim,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        band_groups,
        dropout,
    ):
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
                h=image_size // patch_size, w=image_size // patch_size, dim=pos_dim
            ),  # [L D/2]
            persistent=False,
        )
        self.register_buffer(
            name="band_encoding",
            tensor=posemb_sincos_1d(
                length=self.num_group_patches, dim=band_dim
            ),  # [G D/2]
            persistent=False,
        )

        # Freeze the weights of position & band encoding
        self.pos_encoding = self.pos_encoding.requires_grad_(False)
        self.band_encoding = self.band_encoding.requires_grad_(False)

        self.embed_to_pixels = nn.ModuleDict(
            {
                name: Pixelify(dim, len(bands), patch_size, num_downsampling, unpool)
                for name, bands in self.band_groups.items()
            }
        )

    def reconstruct_and_add_encoding(
        self, unmasked_patches, unmasked_indices, masked_indices
    ):
        """
        Reconstruct the input patches from the random mask patch & add position
        & band encoding to them.

        Parameters
        ----------
        unmasked_patches : torch.Tensor
            A tensor of shape (B, GL:(1 - mask_ratio), D) containing the
            embeddings of the unmasked patches.
        unmasked_indices : torch.Tensor
            A tensor of shape (B, (1 - mask_ratio)) containing the indices of
            the unmasked patches.
        masked_indices : torch.Tensor
            A tensor of shape (B, mask_ratio) containing the indices of the
            masked patches.

        Returns
        -------
        decoder_patches : torch.Tensor
            A tensor of shape (B, GL, D) containing the embeddings for the
            decoder part of the model.
        """
        B, *_ = unmasked_patches.shape

        # Align position & band embeddings across patches
        pos_encoding = repeat(
            self.pos_encoding, "L D -> 1 repeat L D", repeat=self.num_group_patches
        )  # [1 G L D/2]
        band_encoding = repeat(
            self.band_encoding, "G D -> 1 G repeat D", repeat=self.num_spatial_patches
        )  # [1 G L D/2]

        pos_band_encoding = torch.cat(
            (pos_encoding, band_encoding), dim=-1
        )  # [1 G L D]
        pos_band_encoding = rearrange(
            pos_band_encoding, "1 G L D -> 1 (G L) D"
        )  # [1 (GL) D]
        pos_band_encoding = repeat(
            pos_band_encoding, "1 (GL) D -> B (GL) D", B=B
        )  # [B (GL) D]

        batch_indices = rearrange(
            torch.arange(B, device=unmasked_patches.device), "B -> B 1"
        )  # [B 1]
        unmasked_pos_band_encoding = pos_band_encoding[
            batch_indices, unmasked_indices, :
        ]  # [B (GL:(1 - mask_ratio)) D]
        masked_pos_band_encoding = pos_band_encoding[
            batch_indices, masked_indices, :
        ]  # [B (GL:mask_ratio) D]

        # Reconstruct the masked patches from the random mask patch &
        # add position & band encoding to them
        num_masked_patches = int(self.mask_ratio * self.num_patches)
        masked_patches = repeat(
            self.mask_patch, "D -> B GL D", B=B, GL=num_masked_patches
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
            (B, self.num_patches, self.dim), device=unmasked_patches.device
        )  # [B GL D]
        decoder_patches[batch_indices, unmasked_indices, :] = (
            unmasked_patches  # [B GL:(1 - mask_ratio) D]
        )
        decoder_patches[batch_indices, masked_indices, :] = (
            masked_patches  # [B GL:mask_ratio D]
        )

        return decoder_patches  # [B GL D]

    def pixelify(self, patches, embed_inds):
        """
        Convert the patches into pixel space to compute the loss

        Parameters
        ----------
        patches : torch.Tensor
            A tensor of shape (B, GL, D) containing the embeddings from the
            decoder part of the model.

        Returns
        -------
        pixels : torch.Tensor
            A tensor of shape (B, C, L, PP) containing the pixels of the
            cube.
        """
        patches = rearrange(
            patches, "B (G L) D -> B G L D", G=len(self.band_groups)
        )  # [B G L D]
        pixels = []
        for i, (name, bands) in enumerate(self.band_groups.items()):
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

        pixels = torch.cat(pixels, dim=1)  # [B C L PP]
        return pixels  # [B C L PP]

    def forward(
        self, encoded_unmasked_patches, embed_inds, unmasked_indices, masked_indices
    ):
        # Change the embedding dimension from encoder to decoder
        encoded_unmasked_patches = self.enc_to_dec(encoded_unmasked_patches)

        # Reconstruct the patches to feed into the decoder transformer
        decoder_patches = self.reconstruct_and_add_encoding(
            encoded_unmasked_patches, unmasked_indices, masked_indices
        )  # [B GL D]

        # Pass the decoder patches through the transformer
        decoded_patches = self.transformer(decoder_patches)  # [B GL D]

        # Pixelify the decoded patches
        pixels = self.pixelify(decoded_patches, embed_inds)  # [B C L PP]
        return pixels


class MAE(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        image_size,
        patch_size,
        num_downsampling,
        unpool,
        # encoder
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        dropout,
        emb_dropout,
        # decoder
        decoder_dim,
        decoder_depth,
        decoder_heads,
        decoder_dim_head,
        decoder_mlp_ratio,
        decoder_dropout,
        type_probing="attentive",
        # band groups
        band_groups={
            "rgb": (2, 1, 0),
        },
        **kwargs,
    ):
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

        if type_probing is not None:
            self.num_classes = GeoPileDataset.num_classes
            self.probe = nn.ModuleDict(
                [
                    [
                        dataset,
                        AttentiveProbing(dim, num_classes)
                        if type_probing == "attentive"
                        else LinearProbing(dim, num_classes),
                    ]
                    for dataset, num_classes in self.num_classes.items()
                ]
            )
        else:
            self.num_classes = {}

    def per_pixel_loss(self, cube, pixels, masked_matrix):
        """
        Compute the per pixel loss

        Parameters
        ----------
        cube : torch.Tensor
            A tensor of shape (B, C, H, W) containing the pixels of the
            cube.
        pixels : torch.Tensor
            A tensor of shape (B, C, L, PP) containing the pixels per patch of
            the cube.
        masked_matrix : torch.Tensor
            A tensor of shape (B, G, L) containing the mask matrix.

        Returns
        -------
        loss
        """
        patches = rearrange(
            cube,
            "B C (h p1) (w p2) -> B C (h w) (p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )  # [B C L PP]

        # loss = (patches - pixels) ** 2  # loss per pixel
        loss = F.l1_loss(patches, pixels, reduction="none")  # loss per pixel
        loss = reduce(loss, "B C L PP -> B C L", reduction="mean")  # loss per patch

        # mask out the loss for unmasked patches
        actual_loss, masked_patches = 0.0, 0.0
        for i, (_, group) in enumerate(self.band_groups.items()):
            group_loss = reduce(
                loss[:, group, :], "B G L -> B L", "mean"
            )  # (B, L) - loss per group
            actual_loss += (
                group_loss * masked_matrix[:, i]
            ).sum()  # (B, L) * (B, L) -> (B, L) -> (B) -> scalar
            masked_patches += masked_matrix[:, i].sum()  # (B, L) -> (B) -> scalar

        masked = torch.zeros(patches.shape, device=patches.device)
        pred = torch.zeros(patches.shape, device=patches.device)
        for i, (_, group) in enumerate(self.band_groups.items()):
            masked[:, group, :, :] = 1 * masked_matrix[:, None, i, :, None] + patches[
                :, group, :, :
            ] * (1 - masked_matrix[:, None, i, :, None])
            pred[:, group, :, :] = pixels[:, group, :, :] * masked_matrix[
                :, None, i, :, None
            ] + patches[:, group, :, :] * (1 - masked_matrix[:, None, i, :, None])

        masked = rearrange(
            masked,
            "b c (h w) (p1 p2) -> b c (h p1) (w p2)",
            h=self.image_size // self.patch_size,
            p1=self.patch_size,
        )
        pred = rearrange(
            pred,
            "b c (h w) (p1 p2) -> b c (h p1) (w p2)",
            h=self.image_size // self.patch_size,
            p1=self.patch_size,
        )

        return actual_loss / masked_patches, masked, pred

    def forward(self, cube):
        # encoder
        (
            encoded_unmasked_patches,
            embed_inds,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self.encoder(
            cube
        )  # [B (GL:(1 - mask_ratio)) D], [(1-mask_ratio)], [mask_ratio], [B G L]

        # decoder
        pixels = self.decoder(
            encoded_unmasked_patches, embed_inds, unmasked_indices, masked_indices
        )  # [B C L PP]

        # loss
        rec_loss, masked, pred = self.per_pixel_loss(
            cube["image"], pixels, masked_matrix
        )
        probe_logits = {
            dataset: self.probe[dataset](encoded_unmasked_patches)
            for dataset in self.num_classes
        }
        return rec_loss, masked, pred, probe_logits


def mae_tiny(**kwargs):
    args = {
        # encoder
        "dim": 256,
        "depth": 4,
        "heads": 4,
        "dim_head": 64,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        # decoder
        "decoder_dim": 128,
        "decoder_depth": 2,
        "decoder_heads": 2,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 2,
        "decoder_dropout": 0.0,
    }
    args.update(kwargs)
    return MAE(**args)


def mae_small(**kwargs):
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


def mae_medium(**kwargs):
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


def mae_large(**kwargs):
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
