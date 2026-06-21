import wandb
from src.model_TL import model_1, model_2, model_3
from src.train_TL import train_model_TL
from src.dataset import get_dataloaders_TL

sweep_config = {
    "method": "bayes",

    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },

    "parameters": {

        "strategy": {
            "values": [
                "Fixed Feature Extractor",
                "Partial Fine-Tuning",
                "Full Fine-Tuning"
            ]
        },

        "learning_rate": {
            "values": [
                1e-2,
                1e-3,
                1e-4,
                1e-5
            ]
        },

        "batch_size": {
            "values": [
                16,
                32,
                64
            ]
        },

        "weight_decay": {
            "values": [
                0,
                1e-5,
                1e-4,
                1e-3
            ]
        },

        "augmentation": {
            "values": [
                True,
                False
            ]
        },

        "epochs": {
            "values": [5,10]
        },

        "optimizer": {
            "value": "AdamW"
        }
    }
}

def train_transfer_learning():

    wandb.init()

    config = wandb.config

    if config.strategy == "Fixed Feature Extractor":
        model = model_1()

    elif config.strategy == "Partial Fine-Tuning":
        model = model_2()

    else:
        model = model_3()

    train_loader, val_loader, _ = get_dataloaders_TL(
        batch_size=config.batch_size,
        augmentation=config.augmentation
    )

    results = train_model_TL(
        strategy=config.strategy,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        activation="ReLU",
        optimizer_name=config.optimizer,
        batch_norm=True,
        dropout=True,
        drop_prob=0.2,
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

            }, step=epoch + 1)
        
    wandb.finish()


sweep_id = wandb.sweep(
    sweep_config,
    project="Transfer Learning"
)

wandb.agent(
    sweep_id,
    function=train_transfer_learning,
    count=5
)