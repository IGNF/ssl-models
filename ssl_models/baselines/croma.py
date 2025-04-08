import itertools
import math
import warnings
from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, einsum, nn
from torch.nn import Module

from ssl_models.baselines.base import BaseModule


class CROMABaseline(BaseModule):
    def __init__(
        self,
        datasets,
        backbone_size: str,
        unpool_dim: int,
        pretrained_path: str = None,
        freeze: bool = False,
        type_head: Literal["linear", "attentive"] = "linear",
        multimodal: str = "shared",
        add_date_enc: bool = False,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        keep_norm: bool = True,
        **kwargs,
    ) -> None:
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

        if self.backbone_size == "base":
            self.encoder_dim = 768
            self.encoder_depth = 12
            self.num_heads = 16
            self.patch_size = 8
        else:
            # large by default
            self.encoder_dim = 1024
            self.encoder_depth = 24
            self.num_heads = 16
            self.patch_size = 8

        super().__init__(
            datasets,
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

        self.image_size = self._check_image_size()

        self._build_backbone()

    def _build_backbone(self) -> Module:
        modalities = self.dataset.inputs.keys()

        if "s2" in modalities and "s1_asc" in modalities:
            self.croma_modality = "both"
        elif "s2" in modalities:
            self.croma_modality = "optical"
        elif "s1_asc" in modalities:
            self.croma_modality = "SAR"
        else:
            raise NotImplementedError

        self.encoder = PretrainedCROMA(
            pretrained_path=self.pretrained_path,
            size=self.backbone_size,
            modality=self.croma_modality,
            image_resolution=self.image_size,
        )

    def _check_image_size(self):
        all_image_size = None

        for name_mod in self.dataset.inputs:
            mod_image_size = self.dataset.inputs[name_mod].image_size

            if all_image_size is None:
                all_image_size = mod_image_size
            else:
                assert (
                    mod_image_size == all_image_size
                ), "All modalities should have the same image size."

        return all_image_size

    def freeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def grouped_parameters(self, base_lr: float, rate_decay: float = 0.75):
        lr_map = {}
        params_dict = {}

        s2_depth = self.encoder_depth
        s1_depth = s2_depth // 2

        lr_0 = base_lr * rate_decay ** (s2_depth + 1)
        lr_map["s1_encoder.linear_input"] = lr_0
        lr_map["s2_encoder.linear_input"] = lr_0

        params_dict[
            "s1_encoder.linear_input"
        ] = self.encoder.s1_encoder.linear_input.parameters()

        params_dict[
            "s2_encoder.linear_input"
        ] = self.encoder.s2_encoder.linear_input.parameters()

        for i in range(s1_depth):
            s1_key = f"s1_encoder.base_transformer.{i}"

            lr_i = base_lr * rate_decay ** (s2_depth - i)
            lr_map[s1_key] = lr_i

            params_dict[
                s1_key
            ] = self.encoder.s1_encoder.transformer.layers[i].parameters()

        for i in range(s2_depth):
            s2_key = f"s2_encoder.base_transformer.{i}"

            lr_i = base_lr * rate_decay ** (s2_depth - i)
            lr_map[s2_key] = lr_i

            params_dict[
                s2_key
            ] = self.encoder.s2_encoder.transformer.layers[i].parameters()

        for i in range(s1_depth):
            ce_key = f"cross_encoder.{i}"
            lr_i = base_lr * rate_decay ** (s1_depth - i)

            lr_map[ce_key] = lr_i
            params_dict[ce_key] = self.encoder.cross_encoder.layers[i].parameters()

        lr_map["heads"] = base_lr
        params_dict["heads"] = self.heads.parameters()

        grouped_parameters = []
        for k, lr in lr_map.items():
            grouped_parameters.append({"params": params_dict[k], "lr": lr, "name": k})
            print(f'{k}:{lr}')
        return grouped_parameters

    def _expand_channels(self, input: Tensor, channels_to_copy: list[int]):
        s = input[:, channels_to_copy, :, :]
        return torch.cat((input, s), dim=1)

    def _ungroup_s1(self, outputs):
        n_asc, n_des = self.num_dates["s1_asc"], self.num_dates["s1_des"]
        outputs = outputs.unflatten(0, (-1, n_asc + n_des))
        x_s1 = {}
        x_s1["s1_asc"] = outputs[:, :n_asc]
        x_s1["s1_des"] = outputs[:, n_asc : n_asc + n_des]
        x_s1 = self.group(x_s1)

        return x_s1["s1_asc"], x_s1["s1_des"]

    def forward(
        self,
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ):
        batch["s1"] = torch.concat((batch["s1_asc"], batch["s1_des"]), dim=1)
        input = self.group(batch)

        sar_input = input["s1"]
        optical_input = self._expand_channels(input["s2"], [9, 9])

        output_croma = self.encoder(SAR_images=sar_input, optical_images=optical_input)

        x = {}

        if self.croma_modality in ["both", "optical"]:
            x["s2"] = output_croma["optical_encodings"]
        if self.croma_modality in ["both", "SAR"]:
            s1_outputs = output_croma["SAR_encodings"]
            x["s1_asc"], x["s1_des"] = self._ungroup_s1(s1_outputs)

        if self.add_date_enc:
            dates = {}
            for name_mod in self.dataset.inputs:
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


class PretrainedCROMA(nn.Module):
    def __init__(
        self,
        pretrained_path="CROMA_base.pt",
        size="base",
        modality="both",
        image_resolution=120,
    ):
        """
        NOTE: image_resolution is not the spatial, spectral, or temporal resolution. It
        is the height and width of the image, in pixels. E.g., CROMA was pretrained on
        120x120px images, hence image_resolution is 120 by default
        """
        super().__init__()
        # check types
        assert isinstance(
            pretrained_path, str
        ), f"pretrained_path must be a string, not {type(pretrained_path)}"

        assert isinstance(size, str), f"size must be a string, not {type(size)}"

        assert isinstance(
            modality, str
        ), f"modality must be a string, not {type(modality)}"

        assert isinstance(
            image_resolution, int
        ), f"image_resolution must be an int, not {type(image_resolution)}"

        # check values
        assert size in [
            "base",
            "large",
        ], f"size must be either base or large, not {size}"

        assert (
            image_resolution % 8 == 0
        ), f"image_resolution must be a multiple of 8, not {image_resolution}"

        assert modality in [
            "both",
            "SAR",
            "optical",
        ], f"modality must be either both, SAR, or optical, not {modality}"

        # warn the user if the path contains a different size than the size parameter
        if size == "base" and "large" in pretrained_path:
            warnings.warn(
                "The size is set to base, but 'large' appears in the pretrained path!"
            )
        elif size == "large" and "base" in pretrained_path:
            warnings.warn(
                "The size is set to large, but 'base' appears in the pretrained path!"
            )

        if size == "base":
            self.encoder_dim = 768
            self.encoder_depth = 12
            self.num_heads = 16
            self.patch_size = 8
        else:
            # large by default
            self.encoder_dim = 1024
            self.encoder_depth = 24
            self.num_heads = 16
            self.patch_size = 8

        self.modality = modality
        self.num_patches = int((image_resolution / 8) ** 2)
        self.s1_channels = 2  # fixed at 2 SAR backscatter channels
        self.s2_channels = 12  # fixed at 12 multispectral optical channels
        self.attn_bias = get_2dalibi(
            num_heads=self.num_heads, num_patches=self.num_patches
        )

        if modality in ["SAR", "both"]:
            self.s1_encoder = ViT(
                dim=self.encoder_dim,
                depth=int(self.encoder_depth / 2),
                in_channels=self.s1_channels,
            )

            self.GAP_FFN_s1 = nn.Sequential(
                nn.LayerNorm(self.encoder_dim),
                # (BSZ, inner_dim)
                nn.Linear(self.encoder_dim, int(4 * self.encoder_dim)),
                nn.GELU(),  # (BSZ, inner_dim)
                nn.Linear(int(4 * self.encoder_dim), self.encoder_dim),  # (BSZ, dim)
            )

            # load weights
            self.s1_encoder.load_state_dict(torch.load(pretrained_path)["s1_encoder"])
            self.GAP_FFN_s1.load_state_dict(torch.load(pretrained_path)["s1_GAP_FFN"])

        if modality in ["optical", "both"]:
            self.s2_encoder = ViT(
                dim=self.encoder_dim,
                depth=self.encoder_depth,
                in_channels=self.s2_channels,
            )
            self.GAP_FFN_s2 = nn.Sequential(
                nn.LayerNorm(self.encoder_dim),
                # (BSZ, inner_dim)
                nn.Linear(self.encoder_dim, int(4 * self.encoder_dim)),
                nn.GELU(),  # (BSZ, inner_dim)
                nn.Linear(int(4 * self.encoder_dim), self.encoder_dim),  # (BSZ, dim)
            )

            # load weights
            self.s2_encoder.load_state_dict(torch.load(pretrained_path)["s2_encoder"])
            self.GAP_FFN_s2.load_state_dict(torch.load(pretrained_path)["s2_GAP_FFN"])

        if modality == "both":
            self.cross_encoder = BaseTransformerCrossAttn(
                dim=self.encoder_dim,
                depth=int(self.encoder_depth / 2),
                num_heads=self.num_heads,
            )

            # load weights
            self.cross_encoder.load_state_dict(
                torch.load(pretrained_path)["joint_encoder"]
            )

    def forward(self, SAR_images=None, optical_images=None):
        return_dict = {}
        if self.modality in ["SAR", "both"]:
            assert (
                SAR_images is not None
            ), f"Modality is set to {self.modality}, but SAR_images are None"

            SAR_encodings = self.s1_encoder(
                imgs=SAR_images, attn_bias=self.attn_bias.to(SAR_images.device)
            )  # (bsz, num_patches, encoder_dim)
            SAR_GAP = self.GAP_FFN_s1(SAR_encodings.mean(dim=1))  # (bsz, encoder_dim)
            return_dict["SAR_encodings"] = SAR_encodings
            return_dict["SAR_GAP"] = SAR_GAP

        if self.modality in ["optical", "both"]:
            assert (
                optical_images is not None
            ), f"Modality is set to {self.modality}, but optical_images are None"
            optical_encodings = self.s2_encoder(
                imgs=optical_images, attn_bias=self.attn_bias.to(optical_images.device)
            )  # (bsz, num_patches, encoder_dim)

            # (bsz, encoder_dim)
            optical_GAP = self.GAP_FFN_s2(optical_encodings.mean(dim=1))
            return_dict["optical_encodings"] = optical_encodings
            return_dict["optical_GAP"] = optical_GAP

        if self.modality == "both":
            joint_encodings = self.cross_encoder(
                x=SAR_encodings,
                context=optical_encodings,
                relative_position_bias=self.attn_bias.to(optical_images.device),
            )  # (bsz, num_patches, encoder_dim)

            joint_GAP = joint_encodings.mean(dim=1)  # (bsz, encoder_dim)
            return_dict["joint_encodings"] = joint_encodings
            return_dict["joint_GAP"] = joint_GAP

        return return_dict


def get_2dalibi(num_heads, num_patches):
    # inspired by: https://github.com/ofirpress/attention_with_linear_biases
    points = list(
        itertools.product(
            range(int(math.sqrt(num_patches))), range(int(math.sqrt(num_patches)))
        )
    )

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.Tensor(get_slopes(num_heads)).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, num_heads, num_patches, num_patches)


class FFN(nn.Module):
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),  # (BSZ, num_patches, inner_dim)
            nn.GELU(),  # (BSZ, num_patches, inner_dim)
            nn.Dropout(dropout),  # (BSZ, num_patches, inner_dim)
            nn.Linear(inner_dim, dim),  # (BSZ, num_patches, dim)
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        return self.net(x)  # (BSZ, num_patches, dim)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be evenly divisible by num_heads"
        dim_head = int(dim / num_heads)
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, relative_position_bias):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # (BSZ, num_patches, dim)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )  # (BSZ, num_heads, num_patches, dim_head)

        # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = attention_scores + relative_position_bias

        # (BSZ, num_heads, num_patches, num_patches)
        attn = attention_scores.softmax(dim=-1)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        # (BSZ, num_heads, num_patches, dim_head)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be evenly divisible by num_heads"
        dim_head = int(dim / num_heads)
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, relative_position_bias):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        context = self.input_norm(context)  # (BSZ, num_patches, dim)

        q = self.to_q(x)  # (BSZ, num_patches, dim)
        k = self.to_k(context)  # (BSZ, num_patches, dim)
        v = self.to_v(context)  # (BSZ, num_patches, dim)

        # (BSZ, num_heads, num_patches, dim_head)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )

        # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = attention_scores + relative_position_bias

        # (BSZ, num_heads, num_patches, num_patches)
        attn = attention_scores.softmax(dim=-1)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        # (BSZ, num_heads, num_patches, dim_head)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


class BaseTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        final_norm=True,
    ):
        super().__init__()
        self.final_norm = final_norm
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                        FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        if self.final_norm:
            self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, relative_position_bias=False):
        for self_attn, ffn in self.layers:
            x = self_attn(x, relative_position_bias) + x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)

        if self.final_norm:
            return self.norm_out(x)
        else:
            return x


class BaseTransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                        CrossAttention(
                            dim=dim, num_heads=num_heads, dropout=attn_dropout
                        ),
                        FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, context, relative_position_bias):
        for self_attn, cross_attn, ffn in self.layers:
            x = self_attn(x, relative_position_bias) + x  # (BSZ, num_patches, dim)
            # (BSZ, num_patches, dim)
            x = cross_attn(x, context, relative_position_bias) + x
            x = ffn(x) + x  # (BSZ, num_patches, dim)

        x = self.norm_out(x)
        return x  # (BSZ, num_patches, dim)


class ViT(nn.Module):
    def __init__(self, dim, depth, in_channels):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.dim = dim
        self.num_heads = 16  # always 16, for base and large models
        self.patch_size = 8  # always 8, for base and large models

        pixels_per_patch = int(self.patch_size * self.patch_size * in_channels)
        self.linear_input = nn.Linear(pixels_per_patch, self.dim)
        self.transformer = BaseTransformer(
            dim=self.dim,
            depth=self.depth,
            num_heads=self.num_heads,
        )

    def forward(self, imgs, attn_bias):
        x = rearrange(
            imgs,
            "b c (h i) (w j) -> b (h w) (c i j)",
            i=self.patch_size,
            j=self.patch_size,
        )
        # x is shape -> (bsz,num_patches,self.channels*self.patch_size*self.patch_size)

        x = self.linear_input(x)  # (bsz, num_patches, dim)
        x = self.transformer(x, relative_position_bias=attn_bias)
        return x
