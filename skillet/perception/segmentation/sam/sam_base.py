"""Base class for all Segment Anything (SAM) clients."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from PIL import Image


class SAMClient(ABC):
    """Base class for all SAM clients."""

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
    ) -> None:
        """Initialize the base SAM client.

        Args:
            model_path: Path to the SAM model
            device: Device to load model on

        """
        self.device = device
        self.model_path = model_path

    def reset(self) -> None:  # noqa: B027
        """Reset the SAM session."""
        pass

    @abstractmethod
    def segment_from_bboxes(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        bboxes: Sequence[Float[torch.Tensor | np.ndarray, "n 4"]] | None = None,
    ) -> tuple[Float[torch.Tensor, "n 1 h w"], Float[torch.Tensor, " n"]]:
        """Segment detection results from bounding boxes with SAM.

        Args:
            rgb: RGB image to segment. Can be an rgb from an RGBD obs or a PIL image.
            bboxes: Bounding boxes in [ymin, xmin, ymax, xmax] format in pixel space.

        Returns:
            - masks: Segmentation masks of shape (N, 1, H, W).
            - scores: Confidence scores of the segmentation masks.

        """
        raise NotImplementedError

    @abstractmethod
    def segment_from_concepts(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        concepts: list[str],
    ) -> tuple[
        Float[torch.Tensor, "n 1 h w"], Int[torch.Tensor, "n 4"], Float[torch.Tensor, " n"], Int[torch.Tensor, " n"]
    ]:
        """Segment an image from a list of text concepts.

        This functionality was introduced in SAM3.

        Args:
            rgb: RGB image to segment. Can be an rgb from an RGBD obs or a PIL image.
            concepts: List of text concepts to segment.

        Returns:
            - masks: Segmentation masks of shape (N, 1, H, W).
            - boxes: Boxes of the segmentation masks in [ymin, xmin, ymax, xmax] format in pixel space.
            - scores: Confidence scores of the segmentation masks.
            - prompt_indices: Corresponding indices of the prompts that were used to segment the image.

        """
        raise NotImplementedError
