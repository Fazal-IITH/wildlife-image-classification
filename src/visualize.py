import torch
from torchinfo import summary
from src.model import CNN
import random
import matplotlib.pyplot as plt


# Prints out Model Summary
def print_summary(num_blocks, in_channels, in_height, in_width, kernel_channels, conv_kernel_sizes, conv_padding,
                conv_stride, pool_kernel_sizes, pool_stride, activation, num_FC_layers, FC_layers_sizes,
                drop_prob, batch_norm, dropout):

    model = CNN(num_blocks, in_channels, in_height, in_width, kernel_channels, conv_kernel_sizes, conv_padding,
                conv_stride, pool_kernel_sizes, pool_stride, activation, num_FC_layers, FC_layers_sizes,
                drop_prob, batch_norm, dropout)

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



# Gives image as an output containing 30 random images from test set with their correct label and predicted label
def visualize_predictions(model, test_loader):

    device = next(model.parameters()).device

    model.eval()

    dataset_size = len(test_loader.dataset)

    selected_indices = random.sample(
        range(dataset_size),
        min(30, dataset_size)
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

    images = []
    preds = []
    labels = []

    with torch.no_grad():

        for idx in selected_indices:

            image, label = test_loader.dataset[idx]

            output = model(
                image.unsqueeze(0).to(device)
            )

            pred = output.argmax(dim=1).item()

            images.append(image)
            preds.append(pred)
            labels.append(label)

    fig, axes = plt.subplots(
        10,
        3,
        figsize=(12, 24)
    )

    axes = axes.flatten()

    for i in range(len(images)):

        img = images[i].permute(1, 2, 0)

        img = (
            img - img.min()
        ) / (
            img.max() - img.min() + 1e-8
        )

        ax = axes[i]

        ax.imshow(img)

        color = (
            "green"
            if preds[i] == labels[i]
            else "red"
        )

        ax.set_title(
            f"Predicted: {class_names[preds[i]]}, "
            f"Label: {class_names[labels[i]]}",
            fontsize=10,
            color=color
        )

        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# Gives image of the input image and the feature maps of the first layer
def visualize_first_layer_feature_maps(model, test_loader):

    import random
    import math
    import torch
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device

    model.eval()

    # ---------------------------------------
    # Random test image
    # ---------------------------------------

    random_idx = random.randint(
        0,
        len(test_loader.dataset) - 1
    )

    image, label = test_loader.dataset[random_idx]

    image_for_display = image.permute(1, 2, 0)

    image = image.unsqueeze(0).to(device)

    # ---------------------------------------
    # First convolution layer output
    # ---------------------------------------

    if hasattr(model, "hidden_layers"):
        first_conv = model.hidden_layers[0]
    elif hasattr(model, "conv1"):
        first_conv = model.conv1
    else:
        raise ValueError("Cannot find first convolution layer")

    with torch.no_grad():
        feature_maps = first_conv(image)

    feature_maps = feature_maps.squeeze(0).cpu()

    num_filters = feature_maps.shape[0]

    # ---------------------------------------
    # Dynamic Grid Size
    # ---------------------------------------

    grid_cols = math.ceil(math.sqrt(num_filters))
    grid_rows = math.ceil(num_filters / grid_cols)

    # +1 column for original image
    total_cols = grid_cols + 1

    fig = plt.figure(
        figsize=(3 * total_cols, 3 * grid_rows)
    )

    # ---------------------------------------
    # Original Image
    # ---------------------------------------

    ax_img = plt.subplot2grid(
        (grid_rows, total_cols),
        (0, 0),
        rowspan=grid_rows
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
        fontsize=14,
        fontweight="bold"
    )

    ax_img.axis("off")

    # ---------------------------------------
    # Feature Maps
    # ---------------------------------------

    for i in range(num_filters):

        row = i // grid_cols
        col = (i % grid_cols) + 1

        ax = plt.subplot2grid(
            (grid_rows, total_cols),
            (row, col)
        )

        fmap = feature_maps[i]

        fmap = (
            fmap - fmap.min()
        ) / (
            fmap.max()
            - fmap.min()
            + 1e-8
        )

        ax.imshow(
            fmap,
            cmap="viridis"
        )

        ax.set_title(
            f"F{i+1}",
            fontsize=8
        )

        ax.axis("off")

    plt.suptitle(
        f"First Layer Feature Maps ({num_filters} Filters)",
        fontsize=20,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.show()

    plt.close(fig)