import torch
from pathlib import Path
import os


if torch.cuda.is_available():
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")

# For kaggle
if os.path.exists("/kaggle"):

    TRAIN_PATH = "/kaggle/input/inaturalist-12k/train"
    TEST_PATH = "/kaggle/input/inaturalist-12k/test"

# For Google Colab
elif os.path.exists("/content"):

    TRAIN_PATH = "/content/wildlife-image-classification/inaturalist_12K/train"
    TEST_PATH = "/content/wildlife-image-classification/inaturalist_12K/test"

# For the local computer
else:

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    TRAIN_PATH = str(PROJECT_ROOT / "data" / "inaturalist_12K" / "train")
    TEST_PATH = str(PROJECT_ROOT / "data" / "inaturalist_12K" / "test")