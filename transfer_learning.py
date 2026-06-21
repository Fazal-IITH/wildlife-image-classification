from src.dataset import get_dataloaders_TL
from src.train_TL import train_model_TL
from src.model_TL import model_1, model_2, model_3
from src.test_TL import test_model_TL
from src.config import device
import torch

def transfer_learning():

    print(f"\nUsing device: {device}")
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())

    # Strategy 1: Fixed Feature Extractor
    # Strategy 2: Partial Fine-Tuning
    # Strategy 3: Full Fine-Tuning

    model=model_2()
    model=model.to(device)
    strategy="Fixed Feature Extractor"
    epochs = 5
    activation = "ReLU"
    optimizer_name = "AdamW"
    learning_rate = 1e-3
    weight_decay = 1e-4
    batch_norm = True
    dropout = True
    drop_prob = 0.2
    augmentation = True
    batch_size = 64
    gamma = 0.9      # for SGD/Momentum only
    beta1 = 0.9      # Adam/AdamW
    beta2 = 0.999

    train_loader, val_loader, _= get_dataloaders_TL(batch_size, augmentation)

    # results= train_model_TL(strategy, model, train_loader, val_loader, epochs, activation,
    #              optimizer_name, batch_norm, dropout, drop_prob, augmentation,
    #              learning_rate,weight_decay, gamma, beta1, beta2, batch_size)

    test_model_TL(visualize=True)


if __name__ == "__main__":
    transfer_learning()