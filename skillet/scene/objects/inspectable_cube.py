"""InspectableCube scene object for the inspection pick-and-place task."""

from __future__ import annotations

from typing import Any

import torch
from typing_extensions import override

from skillet.scene.scene_objs import Cube


class InspectableCube(Cube):
    """A Cube that carries a ground-truth defect label for the inspection task.

    The ``defective`` attribute is used only for:

    - Spawning defect textures in the MuJoCo scene.
    - Evaluating classifier accuracy after a run.

    It is *never* passed as input to ``DefectClassifier``; classifiers must
    infer the verdict from images alone.
    """

    def __init__(
        self,
        size: float,
        defective: bool | None = None,
        init_pose: torch.Tensor | None = None,
        face_apriltags: list[dict[str, Any]] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the inspectable cube.

        Args:
            size: Side length of the cube in metres.
            defective: Ground-truth defect label. ``None`` means unlabelled.
            init_pose: Initial pose in the world frame (x, y, z, w, x, y, z).
            face_apriltags: AprilTag configurations for cube faces.
            name: Human-readable name.

        """
        super().__init__(size=size, init_pose=init_pose, face_apriltags=face_apriltags, name=name)
        self._defective = defective

    @property
    def defective(self) -> bool | None:
        """Return the ground-truth defect label; ``None`` if not set."""
        return self._defective

    @defective.setter
    def defective(self, value: bool | None) -> None:
        """Set the ground-truth defect label."""
        self._defective = value

    @override
    def __str__(self) -> str:
        """Return a printable string."""
        return (
            f"InspectableCube | ID: {self.object_id} | Name: {self.name}"
            f" | Defective: {self._defective} | Centre: {self.pose.cpu().numpy()[:3]}"
        )
