"""MAE module. Adapted from clay foundation model: https://github.com/Clay-foundation/model."""

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, reduce, repeat
from torch import Tensor, nn
from vit_pytorch.vit import Transformer

from ssl_models.head import AttentiveHead, LinearHead
from ssl_models.odeon_data import OdeonData
from ssl_models.utils import (
    downsampling_from_patch_size,
    posemb_sincos_1d,
    posemb_sincos_2d,
)


class Patchify(nn.Module):
    """Patchify the input cube & create embeddings per patch."""

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_size: int,
        unpool: bool,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.unpool = unpool
        self.num_downsampling = downsampling_from_patch_size(unpool, patch_size)

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
                ],
            )
            self.norms = nn.ModuleDict(
                [
                    [f"norm_{idx}", nn.GroupNorm(1, channels_out)]
                    for idx, (_, channels_out) in enumerate(channels)
                ],
            )
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.conv = nn.Conv2d(
                4 * 2**self.num_downsampling,
                embed_dim,
                kernel_size=1,
            )
            self.norm = nn.GroupNorm(1, embed_dim)

    def forward(self, xb: Tensor) -> tuple[Tensor, list[Tensor]]:
        _, _, H, W = xb.shape  # noqa: N806
        skip = xb
        skip = self.conv_skip(skip)
        skip = self.norm_skip(skip)
        skip = rearrange(
            skip,
            "B C H W -> (B H W) C 1 1",
        )

        inds = []
        if self.unpool:
            xb = rearrange(
                xb,
                "B C (H P1) (W P2) -> (B H W) C P1 P2",
                P1=self.patch_size,
                P2=self.patch_size,
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
            "(B H W) C 1 1 -> B (H W) C",
            H=H // self.patch_size,
            W=W // self.patch_size,
        )
        return xb, inds


class Pixelify(nn.Module):
    """Pixelify patches."""

    def __init__(
        self,
        embed_dim: int,
        out_chans: int,
        patch_size: int,
        unpool: bool,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.unpool = unpool
        self.num_downsampling = downsampling_from_patch_size(unpool, patch_size)

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
                4 * 2**self.num_downsampling,
                kernel_size=1,
            )
            self.norm = nn.GroupNorm(1, 4 * 2**self.num_downsampling)
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
                ],
            )
            self.norms = nn.ModuleDict(
                [
                    [f"norm_{idx}", nn.GroupNorm(1, channels_out)]
                    for idx, (_, channels_out) in enumerate(channels)
                ],
            )
            self.fc = nn.Conv2d(4, out_chans, kernel_size=1)

    def forward(self, xb: Tensor, inds: list[Tensor]) -> Tensor:
        B, _, _ = xb.shape  # noqa: N806
        xb = rearrange(
            xb,
            "B L C -> (B L) C 1 1",
        )
        skip = xb
        skip = self.conv_skip(skip)
        skip = rearrange(
            skip,
            "B (P1 P2 C) 1 1 -> B C P1 P2",
            P1=self.patch_size,
            P2=self.patch_size,
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

        return rearrange(xb, "(B L) C P1 P2 -> B L (P1 P2 C)", B=B)


class Encoder(nn.Module):
    """MAE encoder."""

    def __init__(
        self,
        mask_ratio: float,
        image_size: int,
        patch_size: int,
        unpool: bool,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
        band_groups: dict,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            msg = (
                f"Invalid patch_size {patch_size}."
                f" Expected divisor of image_size {image_size}"
            )
            raise ValueError(msg)
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
                name: Patchify(len(bands), dim, patch_size, unpool)
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

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim * mlp_ratio,
        )

    def to_patch_embed(self, cube: Tensor) -> tuple[Tensor, dict]:
        """Patchify the input cube & create embeddings per patch."""
        patches = []
        embed_inds = {}
        for name, bands in self.band_groups.items():
            cube_slice = cube[:, bands, :, :]  # [B C H W] -> [B C[slice[...]] H W]
            embed_slice, embed_inds[name] = self.patch_embedding[name](cube_slice)
            patches.append(embed_slice)
        patches = rearrange(patches, "G B L D -> B G L D")  # [B G L D]
        return patches, embed_inds

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
        # [B G L D] + [1 G L D] - broadcasting
        return patches + pos_band_encoding

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
            msg = f"Invalid number of patches {GL}. Expected {self.num_patches}."
            raise ValueError(msg)

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

        patches, embed_inds = self.to_patch_embed(
            cube["image"],
        )  # [B G L D] - patchify & create embeddings per patch

        patches = self.add_encodings(
            patches,
        )  # [B G L D] - add position & band encoding to the embeddings

        patches = rearrange(patches, "B G L D -> B (G L) D")  # [B (GL) D]

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
        unpool: bool,
        encoder_dim: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
        band_groups: dict,
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
                name: Pixelify(dim, len(bands), patch_size, unpool)
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

    def pixelify(self, patches: Tensor, embed_inds: dict) -> Tensor:
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
                group_embeddings,
                embed_inds[name],
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
        embed_inds: dict,
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
        return self.pixelify(decoded_patches, embed_inds)  # [B C L PP]


class MAE(nn.Module):
    """Masked Auto Encoder (MAE)."""

    def __init__(  # noqa: PLR0913
        self,
        mask_ratio: float,
        image_size: int,
        patch_size: int,
        unpool: bool,
        # encoder
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
        # decoder
        decoder_dim: int,
        decoder_depth: int,
        decoder_heads: int,
        decoder_dim_head: int,
        decoder_mlp_ratio: float,
        band_groups: dict | None = None,
        type_head: str = "attentive",
        final_mlp: bool = False,
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        if not band_groups:
            band_groups = {"rgb": [0, 1, 2]}
        self.band_groups = band_groups

        self.encoder = Encoder(
            mask_ratio=mask_ratio,
            image_size=image_size,
            patch_size=patch_size,
            unpool=unpool,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            band_groups=band_groups,
        )

        self.decoder = Decoder(
            mask_ratio=mask_ratio,
            image_size=image_size,
            patch_size=patch_size,
            unpool=unpool,
            encoder_dim=dim,
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_ratio=decoder_mlp_ratio,
            band_groups=band_groups,
        )

        self.num_classes = OdeonData.NUM_CLASSES
        self.head = nn.ModuleDict(
            [
                [
                    dataset,
                    AttentiveHead(
                        dim=dim,
                        heads=heads,
                        num_classes=num_classes,
                        final_mlp=final_mlp,
                    )
                    if type_head == "attentive"
                    else LinearHead(
                        dim=dim,
                        num_classes=num_classes,
                        final_mlp=final_mlp,
                    ),
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
            embed_inds,
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
    """Construct MAE tiny."""
    args = {
        "dim": 192,
        "depth": 6,
        "heads": 4,
        "dim_head": 48,
        "mlp_ratio": 2,
        "decoder_dim": 96,
        "decoder_depth": 3,
        "decoder_heads": 2,
        "decoder_dim_head": 48,
        "decoder_mlp_ratio": 2,
    }
    args.update(kwargs)
    return MAE(**args)


def mae_small(**kwargs) -> MAE:  # noqa: ANN003
    """Construct MAE small."""
    args = {
        "dim": 384,
        "depth": 6,
        "heads": 6,
        "dim_head": 64,
        "mlp_ratio": 2,
        "decoder_dim": 192,
        "decoder_depth": 4,
        "decoder_heads": 4,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 2,
    }
    args.update(kwargs)
    return MAE(**args)


def mae_medium(**kwargs) -> MAE:  # noqa: ANN003
    """Construct MAE base/medium."""
    args = {
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "mlp_ratio": 4,
        "decoder_dim": 512,
        "decoder_depth": 6,
        "decoder_heads": 6,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return MAE(**args)


def mae_large(**kwargs) -> MAE:  # noqa: ANN003
    """Construct MAE large."""
    args = {
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_heads": 8,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return MAE(**args)
