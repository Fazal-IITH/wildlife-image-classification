import wandb
from src.train import train_model
from src.dataset import get_dataloaders

wandb.login()

# Sweep Configuration
import wandb
from src.train import train_model
from src.dataset import get_dataloaders

wandb.login()

# Sweep Configuration
sweep_config = {

    "method": "bayes",

    "metric": {
        "name": "validation_accuracy",
        "goal": "maximize"
    },

    "parameters": {

        "epochs": {
            "values": [5,10,15]
        },

        "num_blocks": {
            "value": 5
        },

        "batch_size": {
            "values": [32, 64]
        },

        "learning_rate": {
            "values": [1e-3, 5e-4, 1e-4]
        },

        "weight_decay": {
            "values": [0, 1e-5, 1e-4, 1e-3]
        },

        "optimizer_name": {
            "values": ["Adam", "AdamW"] #,"Momentum","SGD"
        },

        "activation": {
            "values": [
                "ReLU",
                # "GELU",
                # "SiLU",
                # "Mish",
                # "LeakyReLU"
            ]
        },

        "batch_norm": {
            "value": True
        },

        "augmentation": {
            "values": [True, False]
        },

        "dropout": {
            "values": [True, False]
        },

        "drop_prob": {
            "values": [
                0.2,
                0.3,
                0.5
            ]
        },

        # Filter organization

        "kernel_channels": {
            "values": [

                # Same filters throughout
                [16, 16, 16, 16, 16],

                [32, 32, 32, 32, 32],

                [64, 64, 64, 64, 64]

                # # Doubling
                # [16, 32, 64, 128, 256],

                # # Halving
                # [256, 128, 64, 32, 16]
            ]
        },

        "FC_layers_sizes": {
            "values": [
                [256],
                # [512],
                # [1024]
            ]
        }
    }
}

def train_wandb():

    wandb.init(
        project="Wildlife Image Classification using Custom CNNs and Transfer Learning"
    )

    config = wandb.config

    image_height = 224
    image_width = 224

    train_loader, val_loader, _ = get_dataloaders(
        image_height=image_height,
        image_width=image_width,
        batch_size=config.batch_size,
        augmentation=config.augmentation
    )

    results = train_model(

        train_loader=train_loader,
        val_loader=val_loader,

        epochs=config.epochs,

        num_blocks=config.num_blocks,

        in_channels=3,
        in_height=224,
        in_width=224,

        kernel_channels=config.kernel_channels,

        # Fixed architecture choices
        conv_kernel_sizes=[3, 3, 3, 3, 3],
        conv_padding=[1, 1, 1, 1, 1],
        conv_stride=[1, 1, 1, 1, 1],

        pool_kernel_sizes=[2, 2, 2, 2, 2],
        pool_stride=[2, 2, 2, 2, 2],

        activation=config.activation,

        num_FC_layers=1,
        FC_layers_sizes=config.FC_layers_sizes,

        optimizer_name=config.optimizer_name,

        batch_norm=config.batch_norm,

        dropout=True,
        drop_prob=config.drop_prob,

        augmentation=config.augmentation,

        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,

        gamma=0.9,
        beta1=0.9,
        beta2=0.999,

        batch_size=config.batch_size
    )

    for epoch in range(len(results["training_accuracies"])):

        wandb.log({

            "epoch": epoch + 1,

            "training_accuracy":
                results["training_accuracies"][epoch],

            "validation_accuracy":
                results["validation_accuracies"][epoch],

            "validation_f1":
                results["validation_f1_scores"][epoch],

            "training_loss":
                results["training_losses"][epoch],

            "validation_loss":
                results["validation_losses"][epoch]
        })

    wandb.finish()

sweep_id = wandb.sweep(
    sweep_config,
    project="Wildlife Image Classification using Custom CNNs and Transfer Learning"
)

wandb.agent(
    sweep_id,
    function=train_wandb,
    count=15
)