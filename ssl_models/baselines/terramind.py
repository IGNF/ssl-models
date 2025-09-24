from typing import Literal

import torch
from terratorch import BACKBONE_REGISTRY
from torch import Tensor
from torch.nn import Module

from conf.datasets import DatasetsConfig
from ssl_models.baselines.base import BaseModule


class TerramindBaseline(BaseModule):
    """Baseline model based on Terramind."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        unpool_dim: int,
        freeze: bool = False,
        version: str = "v1",
        backbone_size: str = "large",
        type_head: Literal["linear", "attentive"] = "linear",
        multimodal: Literal["mod"] = "shared",
        pretrained_path: str | None = None,
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
            Defines the backbone to use. To choose in "small", "base", "large", "huge".
        unpool_dim: int
            Parameter given to the pixelify head.
        freeze: bool
            To freeze or not to freeze the DinoV2 backbone.
        type_head: str
           Segmentation head to use. Either "linear" (default) or "attentive".
        multimodal: str
           Multimodal strategy: "shared" (default).
        pretrained_path: str
           Path to the location of the pretrained weights.
        add_date_enc: bool
           Whether to add the date encodings to the embeddings.
        fac_date_enc: float
           Factor used to compute the date encodings.
        date_dim: int
           Dimension of the date embeddings.
        keep_norm: bool
           Choose to keep the final layernorm layer.
        kwargs:
           Arguments to pass to `BaseModel` constructor.

        """
        self.datasets = datasets
        self.version = version
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
            self.patch_size = 16
        else:
            # large by default
            self.encoder_dim = 1024
            self.encoder_depth = 24
            self.num_heads = 16
            self.patch_size = 16

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

        self.mod_assoc, self.out_mod_assoc = self._assoc_mods_to_terramind_tokenizers(
            self.num_dates,
        )

        self._build_backbone()

        if self.freeze:
            self.freeze_backbone()

    def _assoc_mods_to_terramind_tokenizers(self, num_dates):
        mod_assoc = {
            "aerial": ["RGB_aerial"],
            "dem": ["DEM"],
            "spot": ["RGB_spot"],
        }
        dataset_mods = self.datasets.dataset.inputs

        if "s2" in dataset_mods and num_dates["s2"] > 0:
            mod_assoc["s2"] = ["S2L2A_{}".format(r + 1) for r in range(num_dates["s2"])]

        if "s1_asc" in dataset_mods and num_dates["s1_asc"] > 0:
            mod_assoc["s1_asc"] = [
                "S1GRD_asc_{}".format(r + 1) for r in range(num_dates["s1_asc"])
            ]

        if "s1_des" in dataset_mods and num_dates["s1_des"] > 0:
            mod_assoc["s1_des"] = [
                "S1GRD_des_{}".format(r + 1) for r in range(num_dates["s1_des"])
            ]

        out_mod_assoc = {v: k for k, l in mod_assoc.items() for v in l}

        return mod_assoc, out_mod_assoc

    def _build_backbone(self) -> Module:
        """Build the Terramind backbone."""
        dataset_mods = self.datasets.dataset.inputs
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

    def _smash_dates(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x_smashed = {}

        for mod in ["aerial", "dem", "spot"]:
            if mod in x:
                tn = self.mod_assoc[mod][0]
                x_smashed[tn] = x[mod]

        for mod in ["s2", "s1_asc", "s1_des"]:
            if mod in x:
                terramind_names = self.mod_assoc[mod]
                mod_splitted = torch.split(x[mod], 1, dim=1)

                assert (
                    len(mod_splitted) == len(terramind_names)
                ), f"Number of dates for {mod} do not match: {len(mod_splitted)} vs {len(terramind_names)}"

                for i, tn in enumerate(terramind_names):
                    x_smashed[tn] = mod_splitted[i]

        return x_smashed

    def _unsmash_dates(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x_unsmashed = {}

        for mod in ["aerial", "dem", "spot"]:
            if mod in self.mod_assoc:
                terramind_name = self.mod_assoc[mod][0]
                if terramind_name in x:
                    x_unsmashed[mod] = x[terramind_name]

        for mod in ["s2", "s1_asc", "s1_des"]:
            if mod in self.mod_assoc:
                terramind_names = self.mod_assoc["s2"]
                mod_splitted = [x[name] for name in terramind_names if name in x]
                if len(mod_splitted) == len(terramind_names):
                    x_unsmashed[mod] = torch.cat(mod_splitted, dim=0)

        return x_unsmashed

    def forward(
        self,
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Terramind v1 forward pass.

        Parameters
        ----------
        batch: dict
            Dictionary containing the data.
        ssl_phase: str
            SSL phase (pretrain, probe, finetune). Only probing and finetuning
            are needed for baselines.
        stage: str
            Learning stage: train, val, test.

        Returns
        -------
        tuple: Terramind's outputs.

        """
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

        # print("x_enc after date enc:", {k: (v.shape, v[0,:,514], torch.isnan(v).any()) for k, v in x_enc.items()})
        loss_pred, log_input, log_pred, log_target = self.compute_loss_pred(
            x_enc,
            batch,
            ssl_phase,
            stage,
        )
        self.metrics[f"loss_pred_{stage}"].update(loss_pred)
        return None, log_input, log_pred, log_target, loss_pred
