import torch
import torch.nn as nn
from src.model import CNN
from src.utils import get_optimizer
from src.evaluate import evaluate

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
# device = torch.device(
#     "mps" if torch.backends.mps.is_available()
#     else "cpu"
# )

print(f"Using device: {device}")

def train_model(train_loader, val_loader, epochs, num_blocks, in_channels, in_height, in_width, kernel_channels, 
                 conv_kernel_sizes, conv_padding, conv_stride, 
                 pool_kernel_sizes, pool_stride, activation, num_FC_layers, FC_layers_sizes,
                 optimizer_name, batch_norm, dropout, drop_prob, augmentation,
                 learning_rate,weight_decay, gamma, beta1, beta2, batch_size=32):

    Model= CNN(num_blocks, in_channels, in_height, in_width, kernel_channels, 
                 conv_kernel_sizes, conv_padding, conv_stride, 
                 pool_kernel_sizes, pool_stride, activation, num_FC_layers, FC_layers_sizes,
                 drop_prob, batch_norm, dropout).to(device)
    
    loss_func= nn.CrossEntropyLoss()

    optimizer= get_optimizer(Model.parameters(), optimizer_name, learning_rate, 
                         weight_decay, gamma, beta1, beta2)

    training_losses=[]
    training_accuracies=[]
    validation_losses=[]
    validation_accuracies=[]
    validation_f1_scores=[]

    best_val_acc = 0

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")
        training_loss=0
        corrected=0
        total=0

        Model.train()
        for batch_X, batch_y in train_loader:
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs=Model(batch_X)

            loss=loss_func(outputs, batch_y)
            training_loss+=loss.item()

            # For training accuracies
            preds= outputs.argmax(dim=1)
            corrected+= (preds==batch_y).sum().item()
            total+= batch_y.shape[0]

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        training_loss= training_loss/len(train_loader)
        training_losses.append(training_loss)
        training_accuracy= (corrected/total)*100
        training_accuracies.append(training_accuracy)

        validation_loss, validation_accuracy, validation_f1_score=evaluate(Model, loss_func, val_loader)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)
        validation_f1_scores.append(validation_f1_score)

        if validation_accuracy > best_val_acc:
            best_val_acc = validation_accuracy

            torch.save({
                "epoch": epoch,
                "val_acc": validation_accuracy,

                "model_params": {
                    "num_blocks": num_blocks,
                    "in_channels": in_channels,
                    "in_height": in_height,
                    "in_width": in_width,
                    "kernel_channels": kernel_channels,
                    "conv_kernel_sizes": conv_kernel_sizes,
                    "conv_padding": conv_padding,
                    "conv_stride": conv_stride,
                    "pool_kernel_sizes": pool_kernel_sizes,
                    "pool_stride": pool_stride,
                    "activation": activation,
                    "num_FC_layers": num_FC_layers,
                    "FC_layers_sizes": FC_layers_sizes,
                    "drop_prob": drop_prob,
                    "batch_norm": batch_norm,
                    "dropout": dropout
                },

                "model_state_dict": Model.state_dict(), # Stores all the trained parameters
                "optimizer_state_dict": optimizer.state_dict() # Stores parameters for optimizer
            }, "best_model.pth")
        print(f"Train Loss: {training_loss:.4f}")
        print(f"Train Acc : {training_accuracy:.4f}")
        print(f"Val Loss  : {validation_loss:.4f}")
        print(f"Val Acc   : {validation_accuracy:.4f}")

    return {
        "training_losses": training_losses,
        "training_accuracies": training_accuracies,
        "validation_losses": validation_losses,
        "validation_accuracies": validation_accuracies,
        "validation_f1_scores": validation_f1_scores
    }
