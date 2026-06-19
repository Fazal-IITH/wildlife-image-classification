import torch
import torch.nn as nn

def activation_func(activation):
    valid_activations = ["ReLU","GELU","SiLU","Mish","LeakyReLU"]

    if activation not in valid_activations:
        raise ValueError(f"Unsupported activation: {activation}")
    
    if activation == 'GELU':
        return nn.GELU()
    elif activation== 'SiLU':
        return nn.SiLU()
    elif activation== 'Mish':
        return nn.Mish()
    elif activation== 'LeakyReLU':
        return nn.LeakyReLU()
    
    return nn.ReLU()
