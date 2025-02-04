import torch
from collections import OrderedDict
from huggingface_hub import hf_hub_download

def replace_if_needed(k, dict_association):
    if k in dict_association:
        return dict_association[k]
    else:
        return k
            
def rename_weights(ckpt):
    dict_association = {
        'cls_token': 'embeddings.cls_token',
        'mask_token': 'embeddings.mask_token',
        'patch_embed': 'embeddings.patch_embeddings',
        'pos_embed': 'embeddings.position_embeddings',
        'proj':'projection',
        'blocks':'encoder.layer',
        'ls1': 'layer_scale1',
        'ls2': 'layer_scale2',
        'attn': 'attention',
        'qkv': 'attention.qkv',
        'gamma': 'lambda1' 
    }

    backbone_ckpt = {}
    for k,v in ckpt.items():
        if 'backbone' in k:
            modules = k.split('.')[1:]
            modules = [replace_if_needed(m, dict_association) for m in modules]
            new_k = '.'.join(modules)
            new_k = new_k.replace('attention.projection', 'attention.output.dense')

            if 'qkv.weight' in new_k:
                new_size = v.shape[0]//3
                k_query = new_k.replace('qkv.weight', 'query.weight')
                backbone_ckpt[k_query] = v[:new_size,:]
            
                k_key = new_k.replace('qkv.weight', 'key.weight')
                backbone_ckpt[k_key] = v[new_size:2*new_size,:]
                
                k_value = new_k.replace('qkv.weight', 'value.weight')
                backbone_ckpt[k_value] = v[2*new_size:3*new_size,:]
                
            elif 'qkv.bias' in new_k:
                new_size = v.shape[0]//3
                k_query = new_k.replace('qkv.bias', 'query.bias')
                backbone_ckpt[k_query] = v[:new_size]
            
                k_key = new_k.replace('qkv.bias', 'key.bias')
                backbone_ckpt[k_key] = v[new_size:2*new_size]
                
                k_value = new_k.replace('qkv.bias', 'value.bias')
                backbone_ckpt[k_value] = v[2*new_size:3*new_size]

            elif ".scale" in new_k or ".zero_point" in new_k:
                continue
            elif "norm.weight" in new_k or "norm.bias" in new_k:
                new_k = new_k.replace('norm', 'layernorm')
                backbone_ckpt[new_k] = v
            else:    
                backbone_ckpt[new_k] = v
            
    return OrderedDict(backbone_ckpt)

def load_and_dequantize(path_compressed):
    state_dict = torch.load(path_compressed, weights_only=True)
    deq_state_dict = []
    
    for key, value in state_dict.items():
        if '_packed_params._packed_params' in key:
            layer_name = '.'.join(key.split('.')[:-2])
            
            deq_weight = torch.dequantize(value[0])
            deq_bias = torch.dequantize(value[1])
        
            deq_state_dict.append(('{}.{}'.format(layer_name, 'weight'), deq_weight))
            deq_state_dict.append(('{}.{}'.format(layer_name, 'bias'), deq_bias))
        elif '_packed_params' not in key:
            deq_state_dict.append((key, value))

    return OrderedDict(deq_state_dict)

def get_canopy_sate_dict(weights_path):
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
            state_dict = torch.load(weights_path, weights_only=True)
        
        #Rename and load
        state_dict = rename_weights(state_dict)
        
        return state_dict

def get_imagenet_dinov2_state_dict(model_size="large"):
    """Récupère le state_dict brut sans initialiser le modèle complet"""
    # Déterminer le nom du repo selon la taille
    repo_id = f"facebook/dinov2-{model_size}"
    
    try:
        # Télécharger le fichier de poids directement
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_model.bin",
            cache_dir="./cache"
        )
        
        # Charger le state_dict
        return torch.load(model_path, weights_only=True)
    
    except Exception as e:
        raise ValueError(f"Erreur de chargement pour {repo_id}: {str(e)}")