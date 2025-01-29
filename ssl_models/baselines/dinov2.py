from transformers import Dinov2Config, Dinov2Model
from abc import ABC, abstractmethod
from ssl_models.baselines.utils import load_and_dequantize, get_backbone_weights
import torch
from torch.nn import Conv2d, Parameter, ModuleDict, Module
from torchmetrics import MeanMetric
from functools import partial, reduce
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from conf.dataset.utils import RasterConfig
from conf.datasets import DatasetsConfig
from ssl_models.layers.embed import Patchify, Pixelify
from ssl_models.layers.head import ClassificationHead, PixelifyHead
from ssl_models.train.metric import MonoLabelMetric, MultiLabelMetric
from einops import rearrange
from typing import Literal

from ssl_models.layers.utils import (
    group_mods,
    ungroup_mods,
)

class DinoSegmentor(Module, ABC):
    """Semantic Segmentation model based on DinoV2."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        backbone_size: str,
        in_channels: str,
        image_size: int,
        unpool_dim: int,
        freeze: bool = False,
        type_head: str = 'naive',
        weight_source: str = 'imagenet',
        weights_path: str = None,
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
        in_channels: int
            Number of channels.
        image_size: int
            Image size (should be multiple of 14, e.g. 224 or 252)
        freeze: bool
            To freeze or not to freeze the DinoV2 backbone.
        type_head: str
           Segmentation head to use. Either 'naive' or 'progressive' or 'upernet'. Default is 'naive'.
        weight_source: str
           Either "imagenet" or "canopy". Default is "imagenet"
        hidden_dim: int
           Used if "seg_head" is set to "progressive".
        kwargs:
            Arguments to pass to `BaseModel` constructor.
        """

        super().__init__(**kwargs)

        self.dataset = datasets.dataset
        self.image_size = image_size
        self.type_head = type_head
        self.weight_source = weight_source
        self.weights_path = weights_path
        self.backbone_size = backbone_size

        if self.weight_source == 'imagenet':
            self.patch_size = 14
        else:
            self.patch_size = 16
        
        self.embed_dim_by_size =  {'small': 384, 'base': 768, 'large': 1024, 'huge': 1280}
        
        assert self.image_size % self.patch_size == 0, "Image size ({}) must be multiple of patch size ({})".format(self.image_size, self.patch_size)
        assert ((self.weight_source == 'imagenet' and backbone_size in ['small', 'base', 'large', 'huge']) \
                or (self.weight_source == 'canopy' and backbone_size in ['large'])), 'Weight source {} is not compatible with backbone_size {}'.format(weight_source, backbone_size)
        assert (self.weights_path is not None or self.weight_source == 'imagenet'), 'A weight path should be given.'
        assert in_channels in ['RGB', 'RGBI', 'IRC'], 'in_channels should be RGB RGBI or IRC'
        
        self.embed_dim = self.embed_dim_by_size[self.backbone_size]
        self.num_channels = len(in_channels)
        self.in_channels = in_channels

        if self.in_channels == 'RGB':
            self.channels_to_use = {'aerial':[0,1,2], 'spot':[0,1,2], 's2':[2,1,0]}
        elif self.in_channels == 'RGBI':
            self.channels_to_use = {'aerial':[0,1,2,3]}
        else: #IRC
            self.channels_to_use = {'aerial':[3,0,1]}

        self.input_image_size = (self.image_size // self.patch_size) * self.patch_size
        
        if self.weight_source == 'imagenet':
            self.dino = self.get_hub_backbone(backbone_size)
            self.dino.mask_token.requires_grad = False
            self.dim = self.adapt_to_channels_number(self.num_channels)
        else:
            self.dino, self.dim = self.get_huggingface_backbone(backbone_size, self.num_channels)
            self.load_cano_weights(self.weights_path, self.num_channels)
                
        if freeze:
            self.freeze_backbone()
        
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

        self.num_bands = {
            name_mod: (
                [mod.bands] if isinstance(mod.bands, int) else list(map(len, mod.bands))
            )
            for name_mod, mod in datasets.dataset.inputs.items()
        }
        self.len_bands = {
            name_mod: len(num_bands) for name_mod, num_bands in self.num_bands.items()
        }

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

    def get_hub_backbone(self, backbone_size):
        """Allows to load the right torch.hub dinov2 backbone, matching the backbone_size.
        Only used when weight_source='imagenet'.

        Parameters
        ----------
        backbone_size: str
            Either 'small', 'base', 'large', or 'giant'.

        Returns
        -------
        tuple: The loaded backbone.
        """
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[backbone_size]
        backbone_name = f"dinov2_{backbone_arch}"

        return(torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name, skip_validation=True))
    
    def get_huggingface_backbone(self, backbone_size, num_channels):
        """Allows to load the right huggingface dinov2 backbone, matching the backbone_size.
        Only used when weight_source='canopy'.
        
        Parameters
        ----------
        backbone_size: str
            Only 'large'.
        num_channels: int
            Number of channels.

        Returns
        -------
        tuple: The loaded backbone.
        """

        if backbone_size == 'large':
            config_dict = {
                'patch_size':16,
                'hidden_size': 1024,
                'num_channels': num_channels,
                'num_attention_heads': 16,
                'num_hidden_layers': 24,
                'out_features':['stage24'],
                'out_indices':[24],
                'stage_names':['stem']+['stage{}'.format(i) for i in range(1,25)]
            }
            configuration = Dinov2Config(**config_dict)
            backbone = Dinov2Model(configuration)
            dim = 1024

            return backbone, dim
        else:
            raise NotImplementedError

    def load_cano_weights(self, weights_path, num_channels):
        """
        Load weights from the paper "Very high resolution canopy height maps from RGB imagery" (https://arxiv.org/abs/2304.07213).

        Parameters
        ----------
        weights_path: str
            Path to the weights.
        num_channels: int
            Number of input channels.
        Returns
        -------

        """
        if 'compressed' in weights_path:
            state_dict = load_and_dequantize(weights_path)
        else:
            state_dict = torch.load(weights_path)
        
        #Rename and load
        backbone_state_dict = get_backbone_weights(state_dict)

        if num_channels>3:
            old_weights = backbone_state_dict['embeddings.patch_embeddings.projection.weight']
            bs, _, hs, ws = old_weights.shape

            pad_tsr = torch.zeros((bs, num_channels-3, hs, ws))
            new_weights = torch.cat((old_weights, pad_tsr), dim=1)
            backbone_state_dict['embeddings.patch_embeddings.projection.weight'] = new_weights

        self.dino.load_state_dict(backbone_state_dict, strict=False)

    def adapt_to_channels_number(self, num_channels):
        """Adapt the first layer of the backbone to take `num_channels` channels as input.

        Parameters
        ----------
        num_channels: int
            Number of channels.

        Returns
        -------
        dim: The embedding number of dimensions.
        
        """
        first_layer = self.dino.patch_embed.proj
        pretrained_weights = first_layer.weight
        dim = first_layer.out_channels
        self.dim = dim

        if num_channels>3:
            print('Adjusting the number of channels...')
            self.dino.patch_embed.proj = Conv2d(num_channels, dim, kernel_size=first_layer.kernel_size, stride=first_layer.stride)
            self.dino.patch_embed.proj.weight.data.normal_(0, 0.001)
            self.dino.patch_embed.proj.weight.data[:, :3, :, :] = Parameter(pretrained_weights)
        
        return dim
    
    def freeze_backbone(self):
        """
        Freeze the backbone's parameters (self.dino).

        Parameters
        ----------
        Returns
        -------
        """
        for param in self.dino.parameters():
             param.requires_grad = False
        
        if self.num_channels > 3:
            # Necessary to unfreeze patchify layer if there are more than 3 channels
            if self.weight_source=='imagenet':
                for param in self.dino.patch_embed.parameters():
                    param.requires_grad = True
            
            elif self.weight_source=='canopy':
                for param in self.dino.embeddings.parameters():
                    param.requires_grad = True

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

        if self.weight_source=='imagenet':
            num_layers = len(self.dino.blocks)
            lr_map = dict([('dino.blocks.{}'.format(i), base_lr * rate_decay ** (num_layers - i)) for i in range(num_layers)])

            if self.in_channels in ['RGBI', 'IRC']:
                lr_map['dino.patch_embed'] = base_lr
            else: #RGB
                lr_map['dino.patch_embed'] = base_lr * rate_decay ** (num_layers+1)
                
            lr_map['segmentation_head'] = base_lr
            
            params_dict = {}
            params_dict['dino.patch_embed'] = self.dino.patch_embed.parameters()
            params_dict['segmentation_head'] = self.segmentation_head.parameters()

            for i in range(num_layers):
                params_dict['dino.blocks.{}'.format(i)] = self.dino.blocks[i].parameters()
                    
        elif self.weight_source=='canopy':
            num_layers = len(self.dino.encoder.layer)
            lr_map = dict([('dino.encoder.layer.{}'.format(i), base_lr * rate_decay ** (num_layers - i)) for i in range(num_layers)])

            if self.in_channels in ['RGBI', 'IRC']:
                lr_map['dino.embeddings'] = base_lr
            else: #RGB
                lr_map['dino.embeddings'] = base_lr * rate_decay ** (num_layers+1)

            lr_map['segmentation_head'] = base_lr
            
            params_dict = {}
            params_dict['dino.embeddings'] = self.dino.embeddings.parameters()
            params_dict['segmentation_head'] = self.segmentation_head.parameters()

            for i in range(num_layers):
                params_dict['dino.encoder.layer.{}'.format(i)] = self.dino.encoder.layer[i].parameters()
            
        print('Layerwise learning rates:')
        grouped_parameters = []
        for k, lr in lr_map.items():
            grouped_parameters.append({'params': params_dict[k], 'lr': lr, 'name': k})
            print('Layer "{}": {}'.format(k, lr))

        return grouped_parameters
    
    def compute_loss_pred(
        self,
        x_enc: dict[str, Tensor],
        batch: dict[str, Tensor],
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        stage: Literal["train", "val", "test"],
    ) -> Tensor:
        """Shared loss computation for all MIM modules."""
        x_enc = self.ungroup(x_enc)

        ref_input = self.dataset.ref_input
        if ref_input is not None:
            x_ref = {}
            for name_mod in x_enc:
                D = x_enc[name_mod].shape[1]  # noqa: N806
                H = self.grid_size[name_mod]  # noqa: N806
                x_ref[name_mod] = rearrange(
                    x_enc[name_mod],
                    "b d (h w) c -> (b d) c h w",
                    h=H,
                )
                x_ref[name_mod] = F.interpolate(
                    x_ref[name_mod],
                    (self.grid_size[ref_input],) * 2,
                    mode="bilinear",
                )
                x_ref[name_mod] = rearrange(
                    x_ref[name_mod],
                    "(b d) c h w -> b d (h w) c",
                    d=D,
                )
            x_ref = torch.cat(
                [x_ref[name_mod] for name_mod in x_ref],
                dim=1,
            )

        x_enc = torch.cat(
            [x_enc[name_mod].flatten(1, 2) for name_mod in x_enc],
            dim=1,
        )

        loss_pred = 0
        for name_target, target in self.dataset.targets.items():
            targets = batch[name_target]

            match target.type_target:
                case "segment":
                    logits = self.heads[name_target](x_ref, ssl_phase)
                    logits = rearrange(logits, "b 1 c h w -> (b h w) c")
                    targets = rearrange(targets, "b 1 1 h w -> (b h w)")
                    targets = targets.long()
                case "change_detect":
                    logits = self.heads[name_target](x_ref, ssl_phase)
                    logits = rearrange(logits, "b 1 1 h w -> (b h w)")
                    targets = rearrange(targets, "b 1 1 h w -> (b h w)")
                    targets = targets.float()
                case "multilabel_classif":
                    logits = self.heads[name_target](x_enc, ssl_phase)
                    targets = targets.float()
                case "classif":
                    logits = self.heads[name_target](x_enc, ssl_phase)
                    targets = targets.long()

            if targets.ndim > 1:
                inds = (targets != target.missing_val).all(dim=1)
            else:
                inds = targets != target.missing_val

            inds = inds.nonzero().squeeze(dim=1)
            if len(inds) == 0:
                continue

            logits_selected = torch.index_select(
                logits,
                dim=0,
                index=inds,
            )
            targets_selected = torch.index_select(
                targets,
                dim=0,
                index=inds,
            )
            loss_pred += self.loss_pred[name_target](
                logits_selected,
                targets_selected,
            )
            self.metrics[f"{name_target}_{stage}"].update(
                logits_selected,
                targets_selected,
            )
        return loss_pred

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
        x = {}

        for name_mod in self.dataset.inputs:
            input_mod = input[name_mod][:, self.channels_to_use[name_mod], :, :]
        
            if self.weight_source == 'imagenet':
                x_enc = self.dino.forward_features(input_mod)
                x_enc = x_enc['x_norm_patchtokens']
            else:
                x_enc = self.dino(input_mod)['last_hidden_state']
                x_enc = x_enc[:, 1:]
        
            x[name_mod] = x_enc

        loss_pred = self.compute_loss_pred(
            x,
            batch,
            ssl_phase,
            stage,
        )
        self.metrics[f"loss_pred_{stage}"].update(loss_pred)

        return None, None, None, loss_pred
