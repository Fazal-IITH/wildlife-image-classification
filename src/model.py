import torch
import torch.nn as nn
from src.utils import activation_func

class CNN(nn.Module):

    def __init__(self, num_blocks, in_channels, in_height, in_width, kernel_channels, 
                 conv_kernel_sizes, conv_padding, conv_stride, 
                 pool_kernel_sizes, pool_stride, activation, num_FC_layers, FC_layers_sizes,
                 drop_prob, batch_norm=False, dropout=False):
        
        super().__init__()

        # \ indicates that the next line is continuation of the curr line.

        assert len(kernel_channels) == num_blocks, \
            "kernel_channels length must equal num_blocks"
        assert len(conv_kernel_sizes) == num_blocks, \
            "conv_kernel_sizes length must equal num_blocks"
        assert len(conv_padding) == num_blocks, \
            "conv_padding length must equal num_blocks"
        assert len(conv_stride) == num_blocks, \
            "conv_stride length must equal num_blocks"
        assert len(pool_kernel_sizes) == num_blocks, \
            "pool_kernel_sizes length must equal num_blocks"
        assert len(pool_stride) == num_blocks, \
            "pool_stride length must equal num_blocks"
        assert len(FC_layers_sizes) == num_FC_layers, \
            "FC_layers_sizes length must equal num_FC_layers"
        assert 0 <= drop_prob < 1, \
            "drop_prob must be between 0 and 1"

        self.hidden_layers= nn.ModuleList()
        self.FC_layers= nn.ModuleList()

        original_in_channels = in_channels
        current_in_channels = in_channels

        for i in range(num_blocks):
            # Conv -> Batch_Norm -> Activation -> MaxPool  
            self.hidden_layers.append(nn.Conv2d(in_channels=current_in_channels, out_channels=kernel_channels[i], 
                                                kernel_size=conv_kernel_sizes[i], stride=conv_stride[i], 
                                                padding=conv_padding[i]))
            if batch_norm==True:
                self.hidden_layers.append(nn.BatchNorm2d(kernel_channels[i]))

            self.hidden_layers.append(activation_func(activation))

            # if i==0:
            #     height=in_height
            #     width=in_width
            # height= ((height + 2*conv_padding[i] - conv_kernel_sizes[i])//conv_stride[i]) + 1
            # height= ((height-pool_kernel_sizes[i])//pool_stride[i]) + 1
            # width= ((width + 2*conv_padding[i] - conv_kernel_sizes[i])//conv_stride[i]) + 1
            # width= ((width-pool_kernel_sizes[i])//pool_stride[i]) + 1

            current_in_channels = kernel_channels[i]
            
            self.hidden_layers.append(nn.MaxPool2d(kernel_size=pool_kernel_sizes[i], stride=pool_stride[i]))
        
        # FC_input= (height * width * in_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        # Dummy Pass to calculate FC layer input dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, original_in_channels, in_height, in_width)

            for layer in self.hidden_layers:
                dummy = layer(dummy)

            dummy = self.global_pool(dummy)
            FC_input = dummy.flatten(1).shape[1] # x.flatten(1) starts falttening from dim 1, leave dim 0 for batches

        for i in range(num_FC_layers):

            self.FC_layers.append(nn.Linear(in_features=FC_input, out_features=FC_layers_sizes[i]))
            self.FC_layers.append(activation_func(activation))
            if dropout==True:
                self.FC_layers.append(nn.Dropout(drop_prob))
            FC_input= FC_layers_sizes[i]
        
        self.output_layer=nn.Linear(FC_input,10)

    def forward(self, x):

        for layer in self.hidden_layers:
            x=layer(x)

        x = self.global_pool(x)
        x= x.flatten(1)

        for layer in self.FC_layers:
            x=layer(x)

        return self.output_layer(x)
    
