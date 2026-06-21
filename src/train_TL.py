import torch
import torch.nn as nn
from src.model import CNN
from src.utils import get_optimizer
from src.evaluate import evaluate
from src.config import device


def train_model_TL(strategy, model, train_loader, val_loader, epochs, activation,
                 optimizer_name, batch_norm, dropout, drop_prob, augmentation,
                 learning_rate,weight_decay, gamma, beta1, beta2, batch_size=32):

    Model= model.to(device)
    loss_func= nn.CrossEntropyLoss()

    # Instead of sending all the Model params, send only those which are being updated(For model_1 and model_2)
    optimizer= get_optimizer([p for p in Model.parameters() if p.requires_grad==True], optimizer_name, learning_rate, 
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
                "epoch": epoch + 1,
                "val_acc": validation_accuracy,

                "model_params": {
                    "strategy": strategy,
                    "batch_size": batch_size,
                    "augmentation": augmentation,
                    "activation": activation,
                    "drop_prob": drop_prob,
                    "batch_norm": batch_norm,
                    "dropout": dropout,
                    "optimizer": optimizer_name,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay
                },

                "model_state_dict": Model.state_dict(), # Stores all the trained parameters
                "optimizer_state_dict": optimizer.state_dict() # Stores parameters for optimizer
            }, "best_resnet50.pth")

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
