import torch
import torch.nn as nn
import torch.optim as optim

def activation_func(activation):
    
    if activation == 'GELU':
        return nn.GELU()
    elif activation== 'SiLU':
        return nn.SiLU()
    elif activation== 'Mish':
        return nn.Mish()
    elif activation== 'LeakyReLU':
        return nn.LeakyReLU()
    elif activation== 'ReLU':
        return nn.ReLU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")


def get_optimizer(model_params, optimizer, learning_rate=1e-3, weight_decay=1e-4, gamma=0.9, beta1=0.9, beta2=0.999):

    if optimizer=='Adam':
        return optim.Adam(model_params, lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    elif optimizer=='Momentum':
        return optim.SGD(model_params, lr=learning_rate, momentum=gamma, weight_decay=weight_decay)
    elif optimizer=='AdamW':
        return optim.AdamW(model_params, lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    elif optimizer=='SGD':
        return optim.SGD(model_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported Optimzer: {optimizer}")
    
