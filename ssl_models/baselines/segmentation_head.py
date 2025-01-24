import torch
from torch.nn import Conv2d, Upsample, Module, Linear, Softmax, Sequential, ReLU
from einops import rearrange
from transformers import Dinov2Config, UperNetConfig
from transformers.models.upernet.modeling_upernet import UperNetHead
import torch.nn.functional as F

class NaiveHead(Module):
    def __init__(self, patch_size, features_dim, img_size, num_classes):
        super(NaiveHead, self).__init__()
        self.grid_size = img_size // patch_size
        #self.num_patches = self.grid_size*self.grid_size
        self.features_dim = features_dim

        self.image_size = img_size
        self.num_classes = num_classes

        self.conv = Conv2d(features_dim, num_classes, (1,1))
        self.up = Upsample(size=(img_size, img_size), mode='bilinear')#, align_corners=True)

    def forward(self,x):
        # input x : bs . num_patches . features_dim
        x = rearrange(x,
                      "bs (gh gw) d -> bs gh gw d",
                      gh = self.grid_size,
                      d = self.features_dim)
        x = x.contiguous()

        x = rearrange(x,
                      "bs gh gw d -> bs d gh gw")
        
        x = x.contiguous()
        
        #x = x.view(-1, self.grid_size, self.grid_size, self.features_dim).permute(0,3,1,2)
        x = self.conv(x)
        x = self.up(x)

        return x
    
class ProgressiveHead(Module):
    def __init__(self, patch_size, features_dim, hidden_dim, img_size, num_classes):
        super(ProgressiveHead, self).__init__()
        self.grid_size = img_size // patch_size
        #self.num_patches = self.grid_size*self.grid_size
        self.features_dim = features_dim
        self.image_size = img_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Calculer le nombre d'étapes nécessaires pour atteindre l'image_size
        num_upsample_steps = int(torch.log2(torch.tensor(patch_size)))
        
        layers = []
        current_dim = self.features_dim
        current_grid_size = self.grid_size
        
        # Construire une alternance de Conv1x1 et d'Upsample
        for _ in range(num_upsample_steps):
            # Conv 1x1 pour changer la dimensionnalité
            layers.append(Conv2d(current_dim, self.hidden_dim, (1,1)))
            layers.append(ReLU(inplace=True))
            
            # Upsample pour doubler la résolution spatiale
            current_grid_size *= 2
            layers.append(Upsample(size=(current_grid_size, current_grid_size), mode='bilinear'))
            
            # Mise à jour de la dimension actuelle
            current_dim = hidden_dim

        # Dernière couche Conv1x1 pour projeter vers num_classes
        layers.append(Conv2d(hidden_dim, num_classes, kernel_size=1))
        layers.append(Upsample(size=(self.image_size, self.image_size), mode='bilinear'))
        # Stocker les couches dans un Sequential
        self.layers = Sequential(*layers)
        
    def forward(self, x):
        # input x : bs . num_patches . features_dim
        x = rearrange(x,
                      "bs (gh gw) d -> bs gh gw d",
                      gh = self.grid_size,
                      d = self.features_dim)
        x = x.contiguous()

        x = rearrange(x,
                      "bs gh gw d -> bs d gh gw")
        
        x = x.contiguous()
        
        # Passer à travers les couches
        x = self.layers(x)
        
        return x
    
class UperNetSegmentationHead(Module):
    def __init__(self, patch_size, features_dim, img_size, num_classes, backbone_type='dinov2_large'):
        super().__init__()

        if backbone_type == 'dinov2_large':
            config_dict = {
                'hidden_size': 1024,
                'num_channels': [features_dim],
                'num_attention_heads': 16,
                'num_hidden_layers': 24,
                'out_features':['stage24'],
                'out_indices':[24],
                'stage_names':['stem']+['stage{}'.format(i) for i in range(1,25)]
            }
            self.num_classes = num_classes
            self.image_size = img_size
            self.patch_size = patch_size
            self.grid_size = self.image_size // self.patch_size
            self.features_dim = features_dim
            backbone_config = Dinov2Config(**config_dict)
            config = UperNetConfig(backbone_config=backbone_config, use_auxiliary_head=False, num_labels=self.num_classes)
            self.decode_head = UperNetHead(config, in_channels=[self.features_dim])

        else:
            raise NotImplementedError
    
    def forward(self, x):
        # input x : bs . num_patches . features_dim
        x = rearrange(x,
                      "bs (gh gw) d -> bs d gh gw",
                      gh = self.grid_size,
                      d = self.features_dim)
        
        feature_maps = (x,)
        logits = self.decode_head(feature_maps)
        logits = F.interpolate(logits, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        return logits