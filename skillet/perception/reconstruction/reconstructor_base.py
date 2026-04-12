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
        self._bbox_frame = None
        self._mask_frame = None
        self._vlm_frame = None
        self._build_scene_flag = False
        self._task_instruction = None

    @property
    def build_scene(self) -> bool:
        return self._build_scene_flag

    @build_scene.setter
    def build_scene(self, build_flag: bool) -> None:
        self._build_scene_flag = build_flag

    @property
    def task_instruction(self) -> str:
        return self._task_instruction

    @task_instruction.setter
    def task_intstruction(self, task: str) -> None:
        self._task_instruction = task

    @task_instruction.setter
    def task_instruction(self, task_instruction: str) -> None:
        self._task_instruction = task_instruction

    @property
    def scene(self) -> Scene:
        return self._scene

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

    @property
    def masks(self) -> torch.Tensor:
        return None

    @property
    def segment_indices(self) -> torch.Tensor:
        return None
