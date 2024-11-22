"""Patchify/pixelify layers."""

from einops import rearrange
from torch import Tensor, nn

from ssl_models.layers.utils import downsampling_from_patch_size


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
        _, D, _, H, W = xb.shape  # noqa: N806
        xb = rearrange(xb, "b d c hp1 wp2 -> (b d) c hp1 wp2")
        skip = self.conv_skip(xb)
        skip = self.norm_skip(skip)
        skip = rearrange(
            skip,
            "bd c h w -> (bd h w) c 1 1",
        )

        inds = []
        if self.unpool:
            xb = rearrange(
                xb,
                "bd c (h p1) (w p2) -> (bd h w) c p1 p2",
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
            "(b d h w) c 1 1 -> b d (h w) c",
            d=D,
            h=H // self.patch_size,
            w=W // self.patch_size,
        )
        return xb, inds


class Pixelify(nn.Module):
    """Pixelify for reconstruction."""

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

    def forward(
        self,
        xb: Tensor,
        inds: list[Tensor],
    ) -> Tensor:
        B, D, L, _ = xb.shape  # noqa: N806
        H = round(L**0.5)  # noqa: N806

        xb = rearrange(
            xb,
            "b d l c -> (b d l) c 1 1",
        )
        skip = self.conv_skip(xb)
        skip = rearrange(
            skip,
            "bdl (p1 p2 c) 1 1 -> bdl c p1 p2",
            p1=self.patch_size,
            p2=self.patch_size,
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

        return rearrange(
            xb,
            "(b d h w) c p1 p2 -> b d c (h p1) (w p2)",
            b=B,
            d=D,
            h=H,
        )
