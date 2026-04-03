from abc import ABC, abstractmethod

import torch
from typing import Any

from skillet.scene.base import Scene
from skillet.core import ObservationSpec
from skillet.core.env import BatchedEnvironment, TSpecObs


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

    @abstractmethod
    def get_observation(self, obs_spec: ObservationSpec[TSpecObs]) -> Any:
        """Return the current state of the scene built by the reconstructor."""
        raise NotImplementedError

    @property
    def scene(self) -> Scene:
        """Return the scene."""
        return self._scene
