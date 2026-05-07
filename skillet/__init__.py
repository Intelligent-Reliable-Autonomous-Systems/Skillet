"""The skillet framework for robot skills."""

import os

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
