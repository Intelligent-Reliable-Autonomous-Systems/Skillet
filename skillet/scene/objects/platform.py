"""Platform scene object for the inspection pick-and-place task."""

from __future__ import annotations

import torch
from typing_extensions import override

from skillet.scene.base import SceneObject


class Platform(SceneObject):
    """A raised platform where non-defective blocks are placed.

    The pose is the geometric centre of the platform body.
    """

    def __init__(
        self,
        width: float,
        depth: float,
        height: float,
        init_pose: torch.Tensor | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the platform.

        Args:
            width: Extent along the x axis in metres.
            depth: Extent along the y axis in metres.
            height: Extent along the z axis in metres.
            init_pose: Pose of the platform's geometric centre in world frame
                (x, y, z, w, x, y, z).
            name: Human-readable name.

        """
        super().__init__(name=name)
        self._width = width
        self._depth = depth
        self._height = height
        self._pose = init_pose

    @property
    def type_name(self) -> str:
        """Return the type identifier used for scene-level id counting."""
        return "platform"

    @property
    def pose(self) -> torch.Tensor | None:
        """Return the pose of the platform in the world frame, or ``None`` if unset."""
        return self._pose

    @pose.setter
    def pose(self, pose: torch.Tensor) -> None:
        """Set the pose of the platform in the world frame."""
        self._pose = pose

    @property
    def aabb(self) -> torch.Tensor:
        """Return the AABB as (min_x, min_y, min_z, max_x, max_y, max_z)."""
        if self._pose is None:
            raise RuntimeError(f"aabb called on platform {self.name!r} before pose is set")
        half = torch.tensor(
            [self._width / 2.0, self._depth / 2.0, self._height / 2.0],
            dtype=torch.float32,
            device=self._pose.device,
        )
        return torch.cat([self._pose[:3] - half, self._pose[:3] + half])

    @override
    def is_pose_known(self) -> bool:
        """Return whether the platform pose has been set."""
        return self._pose is not None

    @property
    def width(self) -> float:
        """Return the extent of the platform along the x axis in metres."""
        return self._width

    @property
    def depth(self) -> float:
        """Return the extent of the platform along the y axis in metres."""
        return self._depth

    @property
    def height(self) -> float:
        """Return the extent of the platform along the z axis in metres."""
        return self._height

    def __str__(self) -> str:
        """Return a printable string."""
        centre = self._pose.cpu().numpy()[:3] if self._pose is not None else None
        return f"Platform | ID: {self.object_id} | Name: {self.name} | Centre: {centre}"
