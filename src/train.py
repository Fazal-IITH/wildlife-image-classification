from torchinfo import summary

model = CNN(
    num_blocks=3,
    in_channels=3,
    in_height=224,
    in_width=224,
    kernel_channels=[32,64,128],
    conv_kernel_sizes=[3,3,3],
    conv_padding=[1,1,1],
    conv_stride=[1,1,1],
    pool_kernel_sizes=[2,2,2],
    pool_stride=[2,2,2],
    activation="ReLU",
    num_FC_layers=1,
    FC_layers_sizes=[512]
)

summary(model, input_size=(1,3,224,224))