import torch
from torchinfo import summary
from src.model import CNN
import random
import matplotlib.pyplot as plt

def print_summary():

    model = CNN(
        num_blocks=5,
        in_channels=3,
        in_height=224,
        in_width=224,

        kernel_channels=[32, 64, 128, 256, 512],

        conv_kernel_sizes=[3, 3, 3, 3, 3],
        conv_padding=[1, 1, 1, 1, 1],
        conv_stride=[1, 1, 1, 1, 1],

        pool_kernel_sizes=[2, 2, 2, 2, 2],
        pool_stride=[2, 2, 2, 2, 2],

        activation="ReLU",

        num_FC_layers=1,
        FC_layers_sizes=[1024],

        drop_prob=0.5,
        batch_norm=True,
        dropout=False
    )

    print(summary(
        model,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params"]
    ))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")


def visualize_predictions(model, test_loader):

    device = next(model.parameters()).device

    model.eval()

    images = []
    true_labels = []
    pred_labels = []

    # Randomly choose 10 indices from the test dataset
    dataset_size = len(test_loader.dataset)
    selected_indices = random.sample(
        range(dataset_size),
        min(10, dataset_size)
    )

    class_names = [
        "Amphibia",
        "Animalia",
        "Arachnida",
        "Aves",
        "Fungi",
        "Insecta",
        "Mammalia",
        "Mollusca",
        "Plantae",
        "Reptilia"
    ]

    with torch.no_grad():

        for idx in selected_indices:

            image, label = test_loader.dataset[idx]

            output = model(
                image.unsqueeze(0).to(device)
            )

            pred = output.argmax(dim=1).item()

            images.append(image)
            true_labels.append(label)
            pred_labels.append(pred)

    fig, axes = plt.subplots(
        10,
        3,
        figsize=(12, 30)
    )

    for i in range(len(images)):

        img = images[i].permute(1, 2, 0)

        img = (img - img.min()) / (
            img.max() - img.min()
        )

        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Image")

        axes[i, 1].text(
            0.5,
            0.5,
            class_names[true_labels[i]],
            fontsize=12,
            ha="center"
        )
        axes[i, 1].axis("off")

        color = (
            "green"
            if true_labels[i] == pred_labels[i]
            else "red"
        )

        axes[i, 2].text(
            0.5,
            0.5,
            class_names[pred_labels[i]],
            fontsize=12,
            ha="center",
            color=color
        )
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_first_layer_feature_maps(model, test_loader):

    device = next(model.parameters()).device

    model.eval()

    random_idx = random.randint(
        0,
        len(test_loader.dataset) - 1
    )

    image, label = test_loader.dataset[random_idx]

    image_for_display = image.permute(1, 2, 0)

    image = image.unsqueeze(0).to(device)

    first_conv = model.hidden_layers[0]

    with torch.no_grad():
        feature_maps = first_conv(image)

    feature_maps = feature_maps.squeeze(0).cpu()

    # 8 rows × 9 columns
    fig = plt.figure(figsize=(20, 16))

    # Original image occupies first column
    ax_img = plt.subplot2grid(
        (8, 9),
        (0, 0),
        rowspan=8
    )

    image_for_display = (
        image_for_display - image_for_display.min()
    ) / (
        image_for_display.max()
        - image_for_display.min()
        + 1e-8
    )

    ax_img.imshow(image_for_display)
    ax_img.set_title(
        "Original Image",
        fontsize=14
    )
    ax_img.axis("off")

    # Feature maps occupy remaining 8×8 grid
    for i in range(64):

        row = i // 8
        col = (i % 8) + 1

        ax = plt.subplot2grid(
            (8, 9),
            (row, col)
        )

        ax.imshow(
            feature_maps[i],
            cmap="gray"
        )

        ax.set_title(
            f"F{i+1}",
            fontsize=8
        )

        ax.axis("off")

    plt.suptitle(
        f"Original Image and First-Layer Feature Maps",
        fontsize=18
    )

    plt.tight_layout()

    plt.show()