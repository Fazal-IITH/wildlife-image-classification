import torch
import torch.nn as nn
from src.model import CNN
from src.dataset import get_dataloaders
from src.evaluate import evaluate
from src.visualize import visualize_predictions, visualize_first_layer_feature_maps

# device = torch.device(
#     "cuda" if torch.cuda.is_available()
#     else "cpu"
# )

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

checkpoint = torch.load("best_model.pth", map_location=device)

params = checkpoint["model_params"]

model = CNN(**params)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)



_, _, test_loader = get_dataloaders(
    image_height=224,
    image_width=224,
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

print("Best Epoch:", checkpoint["epoch"])
print("Validation Accuracy:", checkpoint["val_acc"])

print("Test Accuracy:", test_acc)
print("Test F1 Score:", test_f1)
print("Test Loss:", test_loss)

visualize_predictions(model, test_loader)

visualize_first_layer_feature_maps(
    model,
    test_loader
)
