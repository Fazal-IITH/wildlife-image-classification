import torch
from sklearn.metrics import f1_score


def evaluate(Model, loss_func, val_loader):

    device = next(Model.parameters()).device

    validation_loss=0
    validation_accuracy=0
    validation_f1_score=0
    
    # For F1-Score
    all_preds = []
    all_labels = []

    # For Accuracy
    corrected=0
    total=0

    # Model.eval() turns off the dropout and BatchNorm does not compute mean and variance from the validation batch,
    # Insted it uses running mean and var that were accumulated during training.
    # Pytorch uses running_mean = ((1 - momentum) * running_mean) + (momentum * batch_mean) (This momentum is diff from SGD)
    Model.eval()
    with torch.no_grad():
        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs=Model(batch_X)

            # For F1-Score
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist()) # Why cpu(), bcoz numpy can works only with CPU
            all_labels.extend(batch_y.cpu().tolist())

            # For accuracy
            corrected+= (preds==batch_y).sum().item()
            total+= batch_y.shape[0]

            # Loss Calculation
            loss=loss_func(outputs, batch_y)
            validation_loss+=loss.item()
        
    validation_accuracy=(corrected/total)*100
    validation_f1_score=f1_score(all_labels, all_preds, average='macro')    
    validation_loss=validation_loss/len(val_loader)

    return validation_loss, validation_accuracy, validation_f1_score
