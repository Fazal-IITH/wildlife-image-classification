import torch
import torch.nn as nn
from src.model_TL import model_1, model_2, model_3
from src.dataset import get_dataloaders_TL
from src.evaluate import evaluate
from src.config import device
from src.visualize import visualize_predictions, visualize_first_layer_feature_maps

def test_model_TL(visualize=True):

    checkpoint = torch.load(
        "best_resnet50.pth",
        map_location=device
    )

    strategy = checkpoint["model_params"]["strategy"]

    # Recreate correct architecture
    if strategy == "Fixed Feature Extractor":
        model = model_1()

    elif strategy == "Partial Fine-Tuning":
        model = model_2()

    elif strategy == "Full Fine-Tuning":
        model = model_3()

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    _, _, test_loader = get_dataloaders_TL(
        batch_size=64,
        augmentation=False
    )

    loss_func = nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():

        test_loss, test_acc, test_f1 = evaluate(
            model,
            loss_func,
            test_loader
        )

    print("\n===== Test Results =====")

    print("Strategy:", strategy)
    print("Best Epoch:", checkpoint["epoch"])
    print("Validation Accuracy:", checkpoint["val_acc"])
    print("Test Accuracy:", test_acc)
    print("Test F1 Score:", test_f1)
    print("Test Loss:", test_loss)

    if visualize:

        visualize_predictions(model, test_loader)

        visualize_first_layer_feature_maps(
            model,
            test_loader
        )