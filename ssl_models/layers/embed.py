"""Patchify/pixelify layers."""

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from ssl_models.layers.utils import downsamplings_from_patch_size


class Patchify(nn.Module):
    """Patchify the input cube & create embeddings."""

    def __init__(
        self,
        bands: int | list[list[int]],
        embed_dim: int,
        patch_size: int,
        unpool: bool,
    ) -> None:
        super().__init__()
        self.num_bands = [bands] if isinstance(bands, int) else list(map(len, bands))
        self.patchify_bands = nn.ModuleList(
            [
                PatchifyBands(
                    embed_dim=embed_dim,
                    in_chans=in_chans,
                    patch_size=patch_size,
                    unpool=unpool,
                )
                for in_chans in self.num_bands
            ],
        )

    def forward(self, x: Tensor) -> tuple[Tensor, list[list[Tensor]]]:
        x = torch.split(x, self.num_bands, dim=2)
        x, inds = zip(
            *[self.patchify_bands[idx](x[idx]) for idx, _ in enumerate(x)],
        )
        return torch.cat(x, dim=1), list(inds)


class PatchifyBands(nn.Module):
    """Patchify the input cube & create embeddings per patch/bands."""

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
        self.downsamplings = downsamplings_from_patch_size(unpool, patch_size)

        self.conv_skip = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm_skip = nn.GroupNorm(1, embed_dim)
        if self.unpool:
            channels = []
            for idx, _ in enumerate(self.downsamplings):
                pow_in = idx
                pow_out = idx + 1
                channels_in = 8 * 2**pow_in if idx > 0 else in_chans
                channels_out = 8 * 2**pow_out
                channels.append((channels_in, channels_out))

            self.convs = nn.ModuleList(
                [
                    nn.Conv2d(
                        channels_in,
                        channels_out,
                        kernel_size=self.downsamplings[idx],
                        padding_mode="reflect",
                        padding="same",
                    )
                    for idx, (channels_in, channels_out) in enumerate(channels)
                ],
            )
            self.norms = nn.ModuleList(
                [
                    nn.GroupNorm(1, channels_out)
                    for idx, (_, channels_out) in enumerate(channels)
                ],
            )
            self.max_pool = nn.ModuleList(
                [
                    nn.MaxPool2d(
                        kernel_size=downsampling,
                        stride=downsampling,
                        return_indices=True,
                    )
                    for downsampling in self.downsamplings
                ],
            )
            self.conv = nn.Conv2d(
                (
                    8 * 2 ** len(self.downsamplings)
                    if len(self.downsamplings) > 0
                    else in_chans
                ),
                embed_dim,
                kernel_size=1,
            )
            self.norm = nn.GroupNorm(1, embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        _, D, _, H, W = x.shape  # noqa: N806
        x = rearrange(x, "b d c hp1 wp2 -> (b d) c hp1 wp2")
        skip = self.conv_skip(x)
        skip = self.norm_skip(skip)
        skip = rearrange(
            skip,
            "bd c h w -> (bd h w) c 1 1",
        )

        inds = []
        if self.unpool:
            x = rearrange(
                x,
                "bd c (h p1) (w p2) -> (bd h w) c p1 p2",
                p1=self.patch_size,
                p2=self.patch_size,
            )
            for idx, _ in enumerate(self.downsamplings):
                x = self.convs[idx](x)
                x = self.norms[idx](x)
                x, ind = self.max_pool[idx](x)
                inds.append(ind)

            x = self.conv(x)
            x = self.norm(x)
            x = x + skip
        else:
            x = skip
        x = rearrange(
            x,
            "(b d h w) c 1 1 -> b d (h w) c",
            d=D,
            h=H // self.patch_size,
            w=W // self.patch_size,
        )
        return x, inds


class Pixelify(nn.Module):
    """Pixelify per patch."""

    def __init__(
        self,
        embed_dim: int,
        bands: int | list[list[int]],
        patch_size: int,
        unpool: bool,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_bands = [bands] if isinstance(bands, int) else list(map(len, bands))
        self.pixelify_bands = nn.ModuleList(
            [
                PixelifyBands(
                    embed_dim=embed_dim,
                    out_chans=out_chans,
                    patch_size=patch_size,
                    unpool=unpool,
                )
                for out_chans in self.num_bands
            ],
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        inds: list[list[Tensor]],
    ) -> tuple[Tensor, Tensor]:
        x = rearrange(x, "b (d g) l c -> g b d l c", g=len(self.num_bands))
        mask = repeat(
            mask,
            "b (d g) (h w) c -> g b d c (h p1) (w p2)",
            g=len(self.num_bands),
            p1=self.patch_size,
            p2=self.patch_size,
            h=round(mask.shape[2] ** 0.5),
        )

        x_rec, mask_rec = [], []
        for idx, pixelify in enumerate(self.pixelify_bands):
            x_rec.append(pixelify(x[idx], inds[idx]))
            mask_rec.append(mask[idx].expand((-1, -1, self.num_bands[idx], -1, -1)))

        if len(self.num_bands) > 1:
            x_rec = torch.cat(x_rec, dim=2)
            mask_rec = torch.cat(mask_rec, dim=2)
        else:
            x_rec = x_rec.pop()
            mask_rec = mask_rec.pop()

        return x_rec, mask_rec


class PixelifyBands(nn.Module):
    """Pixelify per patch/bands."""

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
        self.downsamplings = downsamplings_from_patch_size(unpool, patch_size)

        self.conv_skip = nn.Conv2d(
            embed_dim,
            out_chans * self.patch_size**2,
            kernel_size=1,
        )
        if self.unpool:
            channels = []
            for idx, _ in enumerate(self.downsamplings):
                pow_in = len(self.downsamplings) - idx
                pow_out = len(self.downsamplings) - (idx + 1)
                channels_in = 8 * 2**pow_in
                channels_out = 8 * 2**pow_out
                channels.append((channels_in, channels_out))

            self.conv = nn.Conv2d(
                embed_dim,
                8 * 2 ** len(self.downsamplings),
                kernel_size=1,
            )
            self.norm = nn.GroupNorm(1, 8 * 2 ** len(self.downsamplings))
            self.convs = nn.ModuleList(
                [
                    nn.Conv2d(
                        channels_in,
                        channels_out,
                        kernel_size=self.downsamplings[::-1][idx],
                        padding_mode="reflect",
                        padding="same",
                    )
                    for idx, (channels_in, channels_out) in enumerate(channels)
                ],
            )
            self.norms = nn.ModuleList(
                [
                    nn.GroupNorm(1, channels_out)
                    for idx, (_, channels_out) in enumerate(channels)
                ],
            )
            self.max_unpool = nn.ModuleList(
                [
                    nn.MaxUnpool2d(kernel_size=downsampling, stride=downsampling)
                    for downsampling in self.downsamplings[::-1]
                ],
            )
            self.fc = nn.Conv2d(8, out_chans, kernel_size=1)

    def forward(
        self,
        x: Tensor,
        inds: list[Tensor],
    ) -> Tensor:
        B, D, L, _ = x.shape  # noqa: N806
        H = round(L**0.5)  # noqa: N806

        x = rearrange(
            x,
            "b d l c -> (b d l) c 1 1",
        )
        skip = self.conv_skip(x)
        skip = rearrange(
            skip,
            "bdl (p1 p2 c) 1 1 -> bdl c p1 p2",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        if self.unpool:
            x = self.conv(x)
            x = self.norm(x)
            for idx, ind in enumerate(inds[::-1]):
                x = self.max_unpool[idx](x, ind)
                x = self.convs[idx](x)
                x = self.norms[idx](x)
            x = self.fc(x)
            x = x + skip
        else:
            x = skip

        return rearrange(
            x,
            "(b d h w) c p1 p2 -> b d c (h p1) (w p2)",
            b=B,
            d=D,
            h=H,
        )
