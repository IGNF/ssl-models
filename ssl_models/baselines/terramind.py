"""TerraMind module."""

from typing import Literal

import torch
from terratorch import BACKBONE_REGISTRY
from torch import Tensor
from torch.nn import Module

from ssl_models.baselines.base import BaseModule
from ssl_models.conf.datasets import DatasetsConfig


class TerramindBaseline(BaseModule):
    """Baseline model based on Terramind."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        backbone_size: str = "large",
        version: str = "v1",
        freeze: bool = False,
        pretrained_path: str | None = None,
        type_head: Literal["linear", "attentive"] = "linear",
        interpolate: Literal["nearest", "bilinear", "bicubic"] = "nearest",
        fusion_mode: Literal["mod"] = "shared",
        add_date_enc: bool = False,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        keep_norm: bool = True,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Create the Terramind encoder model.

        Parameters
        ----------
        datasets: DatasetsConfig
            The dataset config used in the probing/finetuning phase.
        backbone_size: str
            Backbone size to use. To choose in "base", "large".
        version: str
            TerraMind version. Currently, only TerraMind v1 exists.
        freeze: bool
            Whether to freeze the backbone.
        pretrained_path: str
            Path to the location of the pretrained weights.
        type_head: str
            Segmentation head to use. Either "linear" or "attentive".
        interpolate: str
            Interpolation used in the image resizing before patchification.
        fusion_mode: str
            Fusion strategy. Necessarily "shared".
        add_date_enc: bool
            Whether to add the date encodings to the embeddings.
        fac_date_enc: float
            Factor used to compute the date encodings.
        date_dim: int
            Dimension of the date embeddings.
        keep_norm: bool
            Whether to keep the final layernorm layer.
        kwargs:
            Arguments to pass to `BaseModel` constructor.

        """
        self.dataset = datasets.dataset
        self.backbone_size = backbone_size
        self.version = version
        self.freeze = freeze

        self.pretrained_path = pretrained_path
        self.type_head = type_head
        self.interpolate = interpolate
        self.fusion_mode = fusion_mode

        self.add_date_enc = add_date_enc
        self.fac_date_enc = fac_date_enc
        self.date_dim = date_dim
        self.keep_norm = keep_norm

        match self.backbone_size:
            case "base":
                self.encoder_dim = 768
                self.encoder_depth = 12
                self.num_heads = 16
                self.patch_size = 16
            case "large":
                self.encoder_dim = 1024
                self.encoder_depth = 24
                self.num_heads = 16
                self.patch_size = 16
            case _:
                msg = "Backbone's size should be `base` or `large`"
                raise ValueError(msg)

        super().__init__(
            datasets,
            self.fusion_mode,
            self.patch_size,
            self.encoder_dim,
            self.type_head,
            self.interpolate,
            self.add_date_enc,
            self.fac_date_enc,
            self.date_dim,
            self.keep_norm,
            **kwargs,
        )

        self.mod_assoc, self.out_mod_assoc = self._assoc_mods_to_terramind_tokenizers(
            self.num_dates,
        )

        self._build_backbone()

        if self.freeze:
            self.freeze_backbone()

    def _assoc_mods_to_terramind_tokenizers(
        self,
        num_dates: dict[str, int],
    ) -> tuple[dict[str, list], dict[str, list]]:
        mod_assoc = {
            "aerial": ["RGB_aerial"],
            "dem": ["DEM"],
            "spot": ["RGB_spot"],
        }
        dataset_mods = self.dataset.inputs

        if "s2" in dataset_mods and num_dates["s2"] > 0:
            mod_assoc["s2"] = [f"S2L2A_{idx+1}" for idx in range(num_dates["s2"])]

        if "s1_asc" in dataset_mods and num_dates["s1_asc"] > 0:
            mod_assoc["s1_asc"] = [
                f"S1GRD_asc_{idx+1}" for idx in range(num_dates["s1_asc"])
            ]

        if "s1_des" in dataset_mods and num_dates["s1_des"] > 0:
            mod_assoc["s1_des"] = [
                f"S1GRD_des_{idx+1}" for idx in range(num_dates["s1_des"])
            ]

        out_mod_assoc = {
            name_mod: assoc_mod
            for name_mod, assoc_mods in mod_assoc.items()
            for assoc_mod in assoc_mods
        }

        return mod_assoc, out_mod_assoc

    def _build_backbone(self) -> Module:
        """Build the Terramind backbone."""
        dataset_mods = self.dataset.inputs
        list_of_mods = [
            self.mod_assoc[mod] for mod in dataset_mods if mod in self.mod_assoc
        ]
        terramind_mods = []

        for mods in list_of_mods:
            terramind_mods.extend(mods)

        if "s2" in dataset_mods and self.num_dates["s2"] > 0:
            s2_bands_list = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
            s2_bands = {k: s2_bands_list for k in self.mod_assoc["s2"]}

        self.encoder = BACKBONE_REGISTRY.build(
            f"terratorch_terramind_v1_{self.backbone_size}",
            pretrained=True,
            merge_method="dict",
            modalities=terramind_mods,
            bands=s2_bands if "S2L2A_1" in terramind_mods else None,
        )
        self.encoder.output_mod_name_mapping = {mod: mod for mod in terramind_mods}

    def _check_model_version_and_size(self) -> bool:
        pass

    def freeze_backbone(self) -> None:
        """Freeze the backbone's parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def grouped_parameters(
        self,
        **kwargs,  # noqa: ANN003
    ) -> list[dict]:
        """Skip implem."""

    def _smash_dates(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x_smashed = {}

        for mod in ("aerial", "dem", "spot"):
            if mod in x:
                tn = self.mod_assoc[mod][0]
                x_smashed[tn] = x[mod]

        for mod in ("s2", "s1_asc", "s1_des"):
            if mod in x:
                terramind_names = self.mod_assoc[mod]
                mod_splitted = torch.split(x[mod], 1, dim=1)

                if len(mod_splitted) != len(terramind_names):
                    msg = (
                        f"Number of dates for {mod} do not match: "
                        f"{len(mod_splitted)} vs {len(terramind_names)}"
                    )
                    raise ValueError(msg)

                for idx, name_mod in enumerate(terramind_names):
                    x_smashed[name_mod] = mod_splitted[idx]

        return x_smashed

    def _unsmash_dates(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x_unsmashed = {}

        for mod in ("aerial", "dem", "spot"):
            if mod in self.mod_assoc:
                terramind_name = self.mod_assoc[mod][0]
                if terramind_name in x:
                    x_unsmashed[mod] = x[terramind_name]

        for mod in ("s2", "s1_asc", "s1_des"):
            if mod in self.mod_assoc:
                terramind_names = self.mod_assoc["s2"]
                mod_splitted = [x[name] for name in terramind_names if name in x]
                if len(mod_splitted) == len(terramind_names):
                    x_unsmashed[mod] = torch.cat(mod_splitted, dim=0)

        return x_unsmashed

    def forward(
        self,
        batch: dict[str, Tensor],
        name_dataset: str,  # noqa: ARG002
        ssl_phase: Literal["pretrain", "probe", "finetune"],  # noqa: ARG002
    ) -> tuple[
        dict[str, Tensor] | None,
        None,
        None,
        dict[str, Tensor] | None,
        dict[str, Tensor] | None,
    ]:
        """Terramind v1 forward pass.

        Parameters
        ----------
        batch: dict
            Dictionary containing the data.
        name_dataset: str
            Dataset name (unused, kept for compatibility with MIM modules).
        ssl_phase: str
            SSL phase (pretrain, probe, finetune). Only probing and finetuning
            are used for baselines.

        Returns
        -------
        tuple: Terramind's outputs.

        """
        batch = self.resize_and_rescale(batch)

        x_smashed = self._smash_dates(batch)
        x_smashed = self.group(x_smashed)

        features_terramind = self.encoder(x_smashed)

        x_enc = features_terramind[-1]
        x_enc = self._unsmash_dates(x_enc)

        if self.add_date_enc:
            dates = {}
            for name_mod in self.dataset.inputs:
                dates[name_mod] = batch[f"{name_mod}_dates"]

            ref_date = batch["ref_date"]
            x_enc = self._add_date_encodings(x_enc, dates, ref_date)

        logits = self.compute_logits(x_enc)
        return batch, None, None, x_enc, logits
