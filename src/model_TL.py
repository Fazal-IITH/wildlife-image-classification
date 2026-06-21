from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

def get_resnet50():

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_parameters= model.fc.in_features
    model.fc=nn.Linear(num_parameters, 10)

    return model

# Strategy-1: Freeze evrything except output layer(Fixed Feature Extractor)
def model_1():

    Model_1=get_resnet50()
    
    for param in Model_1.parameters():
        param.requires_grad = False

    for param in Model_1.fc.parameters():
        param.requires_grad = True

    return Model_1

# Strategy-2: Fine-Tune from last block(Block before FC Layer), Partial Fine Tuning
# Why only from last block, why not from before? Bcoz, the initial layers learn common features like edge detector, etc.
# The later blocks mainly depends on the classes or the dataset they are working with
def model_2():

    Model_2=get_resnet50()

    for param in Model_2.parameters():
        param.requires_grad = False

    for param in Model_2.layer4.parameters():
        param.requires_grad = True

    for param in Model_2.fc.parameters():
        param.requires_grad = True

    return Model_2

# Strategy-3: Full Fine-Tuning
def model_3():
    
    Model_3=get_resnet50()

    for param in Model_3.parameters():
        param.requires_grad= True

    return Model_3
