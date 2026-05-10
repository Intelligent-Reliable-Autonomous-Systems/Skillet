"""DefectClassifier for binary defect detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DefectResult:
    """Binary defect classification result."""

    defective: bool
    confidence: float


class DefectClassifier(ABC):
    """Abstract base class for single-image binary defect classifiers."""

    @abstractmethod
    def classify(self, image: np.ndarray, object_id: str) -> DefectResult:
        """Classify a single wrist-camera image of a block.

        Args:
            image: HxWxC uint8 BGR image from the wrist camera.
            object_id: Symbolic id of the block being inspected (for logging).

        Returns:
            DefectResult with binary defective flag and confidence in [0, 1].
        """
        raise NotImplementedError
