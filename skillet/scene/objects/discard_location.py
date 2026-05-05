"""DiscardLocation scene object for the inspection pick-and-place task."""

from __future__ import annotations

import torch
from typing_extensions import override

from skillet.scene.base import SceneObject


class DiscardLocation(SceneObject):
    """A flat region on the workspace surface where defective blocks are routed.

    The pose is the centre of the region at surface level.  The AABB has a
    small ``slab_thickness`` along z to support geometric placement queries.
    """

    def __init__(
        self,
        width: float,
        depth: float,
        slab_thickness: float = 0.001,
        init_pose: torch.Tensor | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the discard location.

        Args:
            width: Extent along the x axis in metres.
            depth: Extent along the y axis in metres.
            slab_thickness: Vertical thickness of the AABB slab in metres.
                Defaults to 1 mm; used only for geometric queries.
            init_pose: Pose of the region centre in world frame
                (x, y, z, w, x, y, z).
            name: Human-readable name.

        """
        super().__init__(name=name)
        self._width = width
        self._depth = depth
        self._slab_thickness = slab_thickness
        self._pose = init_pose

    @property
    def type_name(self) -> str:
        """Return the type identifier used for scene-level id counting."""
        return "discard"

    @property
    def pose(self) -> torch.Tensor | None:
        """Return the pose of the discard region centre in the world frame, or ``None`` if unset."""
        return self._pose

    @pose.setter
    def pose(self, pose: torch.Tensor) -> None:
        """Set the pose of the discard region centre in the world frame."""
        self._pose = pose

    @property
    def aabb(self) -> torch.Tensor:
        """Return the AABB as (min_x, min_y, min_z, max_x, max_y, max_z)."""
        if self._pose is None:
            raise RuntimeError(f"aabb called on discard location {self.name!r} before pose is set")
        half = torch.tensor(
            [self._width / 2.0, self._depth / 2.0, self._slab_thickness / 2.0],
            dtype=torch.float32,
            device=self._pose.device,
        )
        return torch.cat([self._pose[:3] - half, self._pose[:3] + half])

    @override
    def is_pose_known(self) -> bool:
        """Return whether the discard region pose has been set."""
        return self._pose is not None

    @property
    def width(self) -> float:
        """Return the extent of the discard region along the x axis in metres."""
        return self._width

    @property
    def depth(self) -> float:
        """Return the extent of the discard region along the y axis in metres."""
        return self._depth

    @property
    def slab_thickness(self) -> float:
        """Return the vertical thickness of the AABB slab in metres."""
        return self._slab_thickness

    def __str__(self) -> str:
        """Return a printable string."""
        centre = self._pose.cpu().numpy()[:3] if self._pose is not None else None
        return f"DiscardLocation | ID: {self.object_id} | Name: {self.name} | Centre: {centre}"
