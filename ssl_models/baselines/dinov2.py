from torch.nn import ModuleDict, LayerNorm
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
import torch

from conf.datasets import DatasetsConfig

from ssl_models.baselines.utils import get_imagenat_dinov2_state_dict, get_sat_dinov2_state_dict, filter_dict
from ssl_models.baselines.base import BaseModule

from typing import Literal

from transformers import Dinov2Config, AutoConfig
from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings, Dinov2Encoder

class Dinov2Baseline(BaseModule):
    """Semantic Segmentation model based on DinoV2."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        backbone_size: str,
        image_size: int,
        unpool_dim: int,
        freeze: bool = False,
        type_head: str = 'linear',
        multimodal: str = 'shared',
        weight_source: str = 'imagenat',
        pretrained_path: str = None,
        fac_date_enc: float = 1.0,
        date_dim: int = 8,
        keep_norm: bool = True,
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
           Either "random", "imagenat" or "sat". Default is "imagenat"
        hidden_dim: int
           Used if "seg_head" is set to "progressive".
        kwargs:
            Arguments to pass to `BaseModel` constructor.
        """

        self.dataset = datasets.dataset
        self.image_size = image_size
        self.type_head = type_head
        self.multimodal = multimodal
        self.weight_source = weight_source
        self.pretrained_path = pretrained_path
        self.backbone_size = backbone_size
        self.std = 0.01
        self.keep_norm = keep_norm
        self.unpool_dim = unpool_dim
        self.fac_date_enc = fac_date_enc
        self.date_dim = date_dim

        if self.weight_source == 'imagenat':
            self.patch_size = 14
        else:
            self.patch_size = 16

        self.embed_dim_by_size =  {'tiny': 192, 'small': 384, 'base': 768, 'large': 1024, 'huge': 1280}
        self.depth_by_size = {'small': 12, 'base': 12, 'large': 24, 'huge': 32}
        assert self.image_size % self.patch_size == 0, "Image size ({}) must be multiple of patch size ({})".format(self.image_size, self.patch_size)
        assert ((self.weight_source == 'imagenat' and backbone_size in ['small', 'base', 'large', 'huge']) \
                or (self.weight_source == 'sat' and backbone_size in ['large']) \
                or (self.weight_source == 'random' and backbone_size in ['tiny'])), 'Weight source {} is not compatible with backbone_size {}'.format(weight_source, backbone_size)
        assert (self.pretrained_path is not None or self.weight_source == 'imagenat'), 'A weight path should be given.'
        
        self.embed_dim = self.embed_dim_by_size[self.backbone_size]

        super().__init__(datasets, self.patch_size, self.embed_dim, self.unpool_dim, self.type_head, self.multimodal, self.fac_date_enc, self.date_dim, self.keep_norm, **kwargs)
        
        self._build_backbone(backbone_size, weight_source)

        if freeze:
            self._freeze_backbone()
    
    def _get_base_config(self, backbone_size):
        original_config = AutoConfig.from_pretrained(f"facebook/dinov2-{backbone_size}")
        config_dict = original_config.to_dict()
        config_dict["patch_size"] = self.patch_size
        config_dict["image_size"] = 224 if self.weight_source=='sat' else 518

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
        encoder = ModuleDict()
        
        model_names = self.dataset.inputs if self.multimodal != 'shared' else ['shared']
        model_config = Dinov2Config(**config_dict)

        for name_mod in model_names:
            encoder_mod = Dinov2Encoder(model_config)

            if state_dict is not None:
                self._transfer(state_dict, encoder, 3, 'encoder')

            encoder[name_mod] = encoder_mod

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
        if weight_source == 'imagenat':
            state_dict = get_imagenat_dinov2_state_dict(backbone_size)
        elif weight_source == 'sat':
            state_dict = get_sat_dinov2_state_dict(self.pretrained_path)
        else: #random initialization
            state_dict = None
        
        #patch embedding layers (per modality)
        self.patch_embed = self._build_patch_embed(config_dict, state_dict=state_dict)
        #encoder
        self.encoder = self._build_encoder(config_dict, state_dict=state_dict)
        #layer norm
        if self.keep_norm:
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
        
        for name_mod in self.encoder.keys():
            for param in self.encoder[name_mod].parameters():
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
        lr_map = {}

        for name_mod in self.patch_embed:
            if self.len_bands[name_mod]<=3:
                lr_map['patch_embed.{}'.format(name_mod)] = base_lr
            else: #RGB
                lr_map['patch_embed.{}'.format(name_mod)] = base_lr * rate_decay ** (self.depth+1)

        for name_mod in self.encoder:
            for i in range(self.depth):
                lr_map['encoder.{}.{}'.format(name_mod, i)] = base_lr * rate_decay ** (self.depth-i)
        
        lr_map['heads'] = base_lr
                
        params_dict = {}
        for name_mod in self.patch_embed:
            params_dict['patch_embed.{}'.format(name_mod)] = self.patch_embed[name_mod].parameters()

        for i in range(self.depth):
            params_dict['encoder.layer.{}'.format(i)] = self.encoder.layer[i].parameters()

        if self.keep_norm:
            lr_map['layernorm'] = base_lr
            params_dict['layernorm'] =self.layernorm.parameters()

        params_dict['heads'] = self.heads.parameters()

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

            encoder_mod = 'shared' if self.multimodal == 'shared' else name_mod

            x_enc = self.encoder[encoder_mod](x_tokenized)['last_hidden_state']

            if self.keep_norm:
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
