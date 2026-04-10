"""The skillet framework for robot skills."""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
