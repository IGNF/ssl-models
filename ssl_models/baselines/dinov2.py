from abc import ABC

from torch.nn import ModuleDict, Module, LayerNorm
from torchmetrics import MeanMetric
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
import torch

from conf.dataset.utils import RasterConfig
from conf.datasets import DatasetsConfig

from ssl_models.layers.head import ClassificationHead, PixelifyHead
from ssl_models.train.metric import MonoLabelMetric, MultiLabelMetric
from ssl_models.layers.utils import (
    group_mods,
    ungroup_mods,
    encode_dates
)
from ssl_models.baselines.utils import get_imagenet_dinov2_state_dict, get_canopy_sate_dict, filter_dict
from ssl_models.baselines.base import BaseModule

from einops import rearrange
from typing import Literal
from functools import partial

from transformers import Dinov2Config, AutoConfig
from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings, Dinov2Encoder

class DinoSegmentor(BaseModule):
    """Semantic Segmentation model based on DinoV2."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        backbone_size: str,
        image_size: int,
        unpool_dim: int,
        freeze: bool = False,
        type_head: str = 'linear',
        weight_source: str = 'imagenet',
        weights_path: str = None,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        **kwargs,
    ) -> None:
        """Constructor.

        Create the dinov2 encoder model using transformers.

        Parameters
        ----------
        backbone_size: str
            Defines the backbone to use. To choose in "small", "base", "large", "giant"
        num_classes: int
            Number of classes.
        image_size: int
            Image size (should be multiple of 14, e.g. 224 or 252)
        freeze: bool
            To freeze or not to freeze the DinoV2 backbone.
        type_head: str
           Segmentation head to use. Either 'linear' or 'attentive'. Default is 'linear'.
        weight_source: str
           Either "random", "imagenet" or "canopy". Default is "imagenet"
        hidden_dim: int
           Used if "seg_head" is set to "progressive".
        kwargs:
            Arguments to pass to `BaseModel` constructor.
        """

        self.dataset = datasets.dataset
        self.image_size = image_size
        self.type_head = type_head
        self.weight_source = weight_source
        self.weights_path = weights_path
        self.backbone_size = backbone_size
        self.std = 0.01

        if self.weight_source == 'imagenet':
            self.patch_size = 14
        else:
            self.patch_size = 16

        self.embed_dim_by_size =  {'tiny': 192, 'small': 384, 'base': 768, 'large': 1024, 'huge': 1280}
        
        assert self.image_size % self.patch_size == 0, "Image size ({}) must be multiple of patch size ({})".format(self.image_size, self.patch_size)
        assert ((self.weight_source == 'imagenet' and backbone_size in ['small', 'base', 'large', 'huge']) \
                or (self.weight_source == 'canopy' and backbone_size in ['large']) \
                or (self.weight_source == 'random' and backbone_size in ['tiny'])), 'Weight source {} is not compatible with backbone_size {}'.format(weight_source, backbone_size)
        assert (self.weights_path is not None or self.weight_source == 'imagenet'), 'A weight path should be given.'
        
        self.embed_dim = self.embed_dim_by_size[self.backbone_size]

        super().__init__(datasets, self.patch_size, self.embed_dim, type_head, unpool_dim, fac_date_enc, date_dim, **kwargs)
        
        self.input_image_size = (self.image_size // self.patch_size) * self.patch_size

        self.num_bands = {
            name_mod: (
                [mod.bands] if isinstance(mod.bands, int) else list(map(len, mod.bands))
            )
            for name_mod, mod in datasets.dataset.inputs.items()
        }
        self.len_bands = {
            name_mod: len(num_bands) for name_mod, num_bands in self.num_bands.items()
        }

        self._build_backbone(backbone_size, weight_source)

        if freeze:
            self._freeze_backbone()
        
        self.grid_size = {}

        for name_mod, mod in self.dataset.inputs.items():
            self.grid_size[name_mod] = mod.image_size // self.patch_size

        # heads
        self.heads = ModuleDict()
        for name_target, target in self.dataset.targets.items():
            if isinstance(target, RasterConfig):
                if self.dataset.ref_input is None:
                    msg = f"Ref input must be provided for raster target {name_target}"
                    raise ValueError(msg)
                target_image_size = round(
                    self.dataset.crop_meters / target.resolution_meters,
                )
                ref_grid_size = self.grid_size[self.dataset.ref_input]
                if target_image_size % ref_grid_size:
                    msg = (
                        f"Target image size {target_image_size} "
                        f"is not a multiple of ref input grid {ref_grid_size}"
                    )
                    raise ValueError(msg)
                self.heads[name_target] = PixelifyHead(
                    type_head,
                    self.embed_dim,
                    unpool_dim,
                    target.num_classes,
                    target_image_size // ref_grid_size,
                )
            else:
                self.heads[name_target] = ClassificationHead(
                    type_head,
                    self.embed_dim,
                    target.num_classes,
                )

        self.loss_pred = {}
        self.metrics = ModuleDict()
        for name_target, target in datasets.dataset.targets.items():
            match target.type_target:
                case "classif" | "segment":
                    self.loss_pred[name_target] = F.cross_entropy
                    metric_partial = partial(
                        MonoLabelMetric,
                        type_target=target.type_target,
                        num_classes=target.num_classes,
                    )
                case "change_detect":
                    self.loss_pred[name_target] = F.binary_cross_entropy_with_logits
                    metric_partial = partial(
                        MonoLabelMetric,
                        type_target=target.type_target,
                        num_classes=target.num_classes,
                    )
                case "multilabel_classif":
                    self.loss_pred[name_target] = F.binary_cross_entropy_with_logits
                    metric_partial = partial(
                        MultiLabelMetric,
                        num_labels=target.num_classes,
                    )
            for stage in ("train", "val", "test"):
                self.metrics[f"{name_target}_{stage}"] = metric_partial()

        for name_loss in ("loss_rec", "loss_pred"):
            for stage in ("train", "val", "test"):
                self.metrics[f"{name_loss}_{stage}"] = MeanMetric(
                    dist_sync_on_step=False,
                )

        # flattening/unflattening of date dimensions
        self.group = partial(
            group_mods,
            multimodal='shared',
            groups=self.dataset.groups,
        )
        self.ungroup = partial(
            ungroup_mods,
            multimodal='shared',
            groups=self.dataset.groups,
            num_dates={
                name_mod: mod.num_dates * self.len_bands[name_mod]
                for name_mod, mod in self.dataset.inputs.items()
            },
            grid_size=self.grid_size,
        )

        self.enc_date_encoding = {
            name_mod: partial(
                encode_dates,
                dim=self.embed_dim,
                date_dim=date_dim,
                fac_date_enc=fac_date_enc,
                grid_size=self.grid_size[name_mod],
                len_bands=self.len_bands[name_mod],
            )
            for name_mod in self.dataset.inputs
        }
    
    def _get_base_config(self, backbone_size):
        original_config = AutoConfig.from_pretrained(f"facebook/dinov2-{backbone_size}")
        config_dict = original_config.to_dict()
        config_dict["patch_size"] = self.patch_size
        config_dict["image_size"] = 224 if self.weight_source=='canopy' else 518

        return config_dict
    
    def _build_patch_embed(self, config_dict, state_dict=None):
        patch_embed = ModuleDict()

        for name_mod in self.dataset.inputs:
            config_dict["num_channels"] = self.num_bands[name_mod][0]            
            mod_config = Dinov2Config(**config_dict)

            patch_embed[name_mod] = Dinov2Embeddings(mod_config)
            
            if state_dict is not None:
                self._transfer(state_dict, patch_embed[name_mod], config_dict["num_channels"], 'embeddings')

        return patch_embed
    
    def _build_encoder(self, config_dict, state_dict=None):
        model_config = Dinov2Config(**config_dict)
        encoder = Dinov2Encoder(model_config)
        
        if state_dict is not None:
            self._transfer(state_dict, encoder, 3, 'encoder')

        return encoder
    
    def _build_layer_norm(self, config_dict, state_dict=None):
        layernorm = LayerNorm(config_dict['hidden_size'], eps=config_dict['layer_norm_eps'])

        if state_dict is not None:
            self._transfer(state_dict, layernorm, 3, 'layernorm')

        return layernorm
    
    def _transfer(self, state_dict, module, num_channels, headname):
        if num_channels>3:
            adapted_state_dict = self._transfer_patch_embedding_to_more_channels(state_dict, num_channels, self.std)
            module.load_state_dict(filter_dict(adapted_state_dict, headname), strict=False)
        else:
            module.load_state_dict(filter_dict(state_dict, headname), strict=False)
    
    def _build_backbone(self, backbone_size, weight_source):
        #config dict
        config_dict = self._get_base_config(backbone_size)
        
        #state dict
        if weight_source == 'imagenet':
            state_dict = get_imagenet_dinov2_state_dict(backbone_size)
        elif weight_source == 'canopy':
            state_dict = get_canopy_sate_dict(self.weights_path)
        else: #random initialization
            state_dict = None
        
        #patch embedding layers (per modality)
        self.patch_embed = self._build_patch_embed(config_dict, state_dict=state_dict)
        #encoder
        self.encoder = self._build_encoder(config_dict, state_dict=state_dict)
        #layer norm
        self.layernorm = self._build_layer_norm(config_dict, state_dict=state_dict)
        
    def _transfer_patch_embedding_to_more_channels(self, state_dict, num_channels, std):        
        with torch.no_grad():
            new_state_dict = state_dict.copy()

            old_weights = new_state_dict['embeddings.patch_embeddings.projection.weight']
            bs, _, hs, ws = old_weights.shape
            
            means = torch.zeros((bs, num_channels-3, hs, ws))
            stds = torch.ones((bs, num_channels-3, hs, ws))*std

            pad_tensor = torch.normal(means, stds)
            new_weights = torch.cat((old_weights, pad_tensor), dim=1)
            new_state_dict['embeddings.patch_embeddings.projection.weight'] = new_weights
    
        return new_state_dict

    def _freeze_backbone(self):
        """
        Freeze the backbone's parameters (self.dino).

        Parameters
        ----------
        Returns
        -------
        """
        for name_mod in self.patch_embed.keys():
            # Necessary to unfreeze patchify layer if there are more than 3 channels
            if self.len_bands[name_mod]<=3:
                for param in self.patch_embed[name_mod].parameters():
                    param.requires_grad = False
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.layernorm.parameters():
            param.requires_grad = False

    def grouped_parameters(self, base_lr, rate_decay=0.75):
        """
        Return the grouped parameters dictionnary with layerwise learning rate.
        
        Parameters
        ----------
        base_lr: float
            Base learning rate.
        rate_decay: float
            Input's key.

        Returns
        -------
        List: A list of dictionnaries, containing 'params' and 'lr' keys, in which parameters are grouped.

        """
        num_layers = len(self.encoder.layer)
        lr_map = dict([('encoder.layer.{}'.format(i), base_lr * rate_decay ** (num_layers - i)) for i in range(num_layers)])
        
        for name_mod in self.patch_embed:
            if self.len_bands[name_mod]<=3:
                lr_map['patch_embed.{}'.format(name_mod)] = base_lr
            else: #RGB
                lr_map['patch_embed.{}'.format(name_mod)] = base_lr * rate_decay ** (num_layers+1)

        lr_map['heads'] = base_lr
        
        params_dict = {}
        for name_mod in self.patch_embed:
            params_dict['patch_embed.{}'.format(name_mod)] = self.patch_embed[name_mod].parameters()
        params_dict['heads'] = self.heads.parameters()

        for i in range(num_layers):
            params_dict['encoder.layer.{}'.format(i)] = self.encoder.layer[i].parameters()

        print('Layerwise learning rates:')
        grouped_parameters = []
        for k, lr in lr_map.items():
            grouped_parameters.append({'params': params_dict[k], 'lr': lr, 'name': k})
            print('Layer "{}": {}'.format(k, lr))

        return grouped_parameters
    
    def forward(self,
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ):
        """
        Dinov2 forward pass.

        Parameters
        ----------
        batch: dict
            Dictionary containing the data.
        key: str
            Input's key.

        Returns
        -------
        tuple: DinoV2's outputs.
        """
        # Forward pass through the dinov2 encoder, get encoder's feature maps.
        input = self.group(batch)
        x, dates = {}, {}

        for name_mod in self.dataset.inputs:
            input_mod = input[name_mod]

            x_tokenized = self.patch_embed[name_mod](input_mod)
            x_enc = self.encoder(x_tokenized)['last_hidden_state']
            x_enc = self.layernorm(x_enc)
            x_enc = x_enc[:, 1:]
        
            x[name_mod] = x_enc
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
