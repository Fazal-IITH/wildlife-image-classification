from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet50_Weights
from src.config import TRAIN_PATH, TEST_PATH



def get_dataloaders(image_height, image_width, batch_size=32, augmentation=False):

    # Image transformations
    test_transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor()
    ])

    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((image_height, image_width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])
    else:
        train_transform = test_transform
    

    #Num of Images=9999
    full_train_dataset = datasets.ImageFolder(
        root=TRAIN_PATH,
        transform=train_transform
    )

    # Same images, different transform
    full_val_dataset = datasets.ImageFolder(
        root=TRAIN_PATH,
        transform=test_transform
    )

    #.targets gives us class labels according to the indices
    targets=full_train_dataset.targets

    #List of the indices from 0->9998
    indices= list(range(len(full_train_dataset)))

    #This splits the indices into train and val, stratify takes labels as input and then samples 20% from each class
    #Why use stratify? bcoz i want 20% from each class, not randomly. Stratify does that.
    train_indices, val_indices= train_test_split(indices, test_size=0.2, random_state=42, stratify=targets)

    #Extract training data from full_dataset using indices
    train_dataset=Subset(full_train_dataset, train_indices)
    
    #Extract val data from full_dataset using indices
    val_dataset=Subset(full_val_dataset, val_indices)
    
    test_dataset=datasets.ImageFolder(
        root=TEST_PATH,
        transform=test_transform
    )
   
    train_loader= DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader= DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, #Why False, bcoz model is not training and we want consistent measurement.
        num_workers=2
    )

    test_loader= DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader, test_loader



def get_dataloaders_TL(batch_size=32, augmentation=False):

    weights = ResNet50_Weights.DEFAULT

    # Image transformations
    test_transform= weights.transforms()

    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize(232),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = test_transform

    #Num of Images=9999
    full_train_dataset = datasets.ImageFolder(
        root=TRAIN_PATH,
        transform=train_transform
    )

    # Same images, different transform
    full_val_dataset = datasets.ImageFolder(
        root=TRAIN_PATH,
        transform=test_transform
    )

    #.targets gives us class labels according to the indices
    targets=full_train_dataset.targets

    #List of the indices from 0->9998
    indices= list(range(len(full_train_dataset)))

    #This splits the indices into train and val, stratify takes labels as input and then samples 20% from each class
    #Why use stratify? bcoz i want 20% from each class, not randomly. Stratify does that.
    train_indices, val_indices= train_test_split(indices, test_size=0.2, random_state=42, stratify=targets)

    #Extract training data from full_dataset using indices
    train_dataset=Subset(full_train_dataset, train_indices)
    
    #Extract val data from full_dataset using indices
    val_dataset=Subset(full_val_dataset, val_indices)
    
    test_dataset=datasets.ImageFolder(
        root=TEST_PATH,
        transform=test_transform
    )
   
    train_loader= DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader= DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, #Why False, bcoz model is not training and we want consistent measurement.
        num_workers=2
    )

    test_loader= DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader, test_loader
    
