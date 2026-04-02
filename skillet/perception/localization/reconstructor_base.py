from abc import ABC, abstractmethod

import torch

from skillet.scene.base import Scene


class ReconstructorBase(ABC):
    """Base scene reconstructor for cube localization with AprilTags or segmentation + depth."""

    def __init__(self, scene: Scene | None = None) -> None:
        """Initialize the AprilTag state estimator.

        Args:
            scene: The scene to update with the estimated poses of the AprilTags.

        """
        self._scene = scene

    @abstractmethod
    def update_state(self, obs: dict[str, torch.Tensor], update: bool = True) -> None:
        """Update the state estimator with a new observation.

        Args:
            obs: The RGB-D observation to update the state estimator with.
            update: If to update the scene or not

        """
        raise NotImplementedError

    @property
    def scene(self) -> Scene:
        """Return the scene."""
        return self._scene
