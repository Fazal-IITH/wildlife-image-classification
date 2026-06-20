import torch
from torchinfo import summary
from src.dataset import get_dataloaders
from src.model import CNN
from src.train import train_model

def main():
    epochs = 25
    num_blocks = 5
    kernel_channels = [32, 64, 128, 256, 512]
    conv_kernel_sizes = [3, 3, 3, 3, 3]
    conv_padding = [1, 1, 1, 1,1]
    conv_stride = [1, 1, 1, 1,1]
    pool_kernel_sizes = [2, 2, 2, 2,2]
    pool_stride = [2, 2, 2, 2,2]
    activation = "ReLU"
    num_FC_layers = 1
    FC_layers_sizes = [1024]
    optimizer_name = "AdamW"
    learning_rate = 1e-3
    weight_decay = 1e-4
    batch_norm = True
    dropout = False
    drop_prob = 0.5
    augmentation = True
    batch_size = 64
    gamma = 0.9      # for SGD/Momentum only
    beta1 = 0.9      # Adam/AdamW
    beta2 = 0.999

    image_channels = 3
    image_height = 224
    image_width = 224

    train_loader, val_loader, test_loader= get_dataloaders(image_height, image_width, batch_size, augmentation)

    results= train_model(train_loader, val_loader, epochs, num_blocks, image_channels, image_height, image_width, kernel_channels, 
                    conv_kernel_sizes, conv_padding, conv_stride, 
                    pool_kernel_sizes, pool_stride, activation, num_FC_layers, FC_layers_sizes,
                    optimizer_name, batch_norm, dropout, drop_prob, augmentation,
                    learning_rate,weight_decay, gamma, beta1, beta2, batch_size)

    print(results['validation_accuracies'])


if __name__ == "__main__":
    main()