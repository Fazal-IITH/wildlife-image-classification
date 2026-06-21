import torch
from pathlib import Path


if torch.cuda.is_available():
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = str(PROJECT_ROOT / "data" / "inaturalist_12K" / "train")
TEST_PATH = str(PROJECT_ROOT / "data" / "inaturalist_12K" / "test")