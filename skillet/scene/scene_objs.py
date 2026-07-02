"""Object state representations."""

from typing import Any, ClassVar, Literal, override

import torch
from jaxtyping import Float

from skillet import DEVICE
from skillet.core.math import normalize, quat_apply, quat_from_matrix, quat_inv, quat_mul
from skillet.scene.base import SceneObject


class EMAFilter:
    """EMA filter for object positions."""

    def __init__(self, alpha: float = 0.7, init: torch.Tensor | None = None):
        self.alpha = alpha
        self.x = None if init is None else torch.as_tensor(init, dtype=torch.float32)

    def update(self, measurement: torch.Tensor) -> torch.Tensor:
        """Update the current pose."""
        if self.x is None:
            self.x = measurement
        else:
            self.x = self.alpha * measurement + (1 - self.alpha) * self.x

        return self.x


class Cube(SceneObject):
    """A cube in a scene."""

    FACE_INDICES: ClassVar[dict[str, int]] = {
        "front": 0,
        "back": 1,
        "left": 2,
        "right": 3,
        "top": 4,
        "bottom": 5,
    }

    # Canonical outward face normals in the cube's local frame, indexed by FACE_INDICES.
    FACE_NORMALS: ClassVar[list[list[float]]] = [
        torch.tensor([-1.0, 0.0, 0.0]),  # front
        torch.tensor([1.0, 0.0, 0.0]),  # back
        torch.tensor([0.0, 1.0, 0.0]),  # left
        torch.tensor([0.0, -1.0, 0.0]),  # right
        torch.tensor([0.0, 0.0, 1.0]),  # top
        torch.tensor([0.0, 0.0, -1.0]),  # bottom
    ]

    FACE_UPS: ClassVar[list[Float[torch.Tensor, "3"]]] = [
        torch.tensor([0.0, 0.0, 1.0]),  # toward top face
        torch.tensor([0.0, 0.0, 1.0]),  # toward top face
        torch.tensor([0.0, 0.0, 1.0]),  # toward top face
        torch.tensor([0.0, 0.0, 1.0]),  # toward top face
        torch.tensor([1.0, 0.0, 0.0]),  # toward back face
        torch.tensor([-1.0, 0.0, 0.0]),  # toward front face
    ]

    def __init__(
        self,
        size: float,
        init_pose: torch.Tensor | None = None,  # (x, y, z, w, x, y, z)
        face_apriltags: list[dict[str, Any]] | None = None,
        name: str | None = None,
        material: str = "plastic",
        color: str = "blue",
        moveable: bool = True,
    ) -> None:
        """Initialize the cube.

        Args:
            size: The side length of the cube in meters.
            init_pose: The initial pose of the cube in the world frame.
            face_to_apriltag: The mapping from the cube's faces to the AprilTag IDs.
                - id: The ID of the AprilTag.
                - size: The size of the AprilTag in meters.
                - orientation: The rotation of the AprilTag around the normal vector.
                    On the sides, 0 means the tag is upright.
                    For "top", 0 means the top of the tag is near the back side
                    For "bottom", 0 means the bottom of the tag is near the front side.

        """
        super().__init__(name=name, localizable=True)
        self._size = size
        self._pose = init_pose if init_pose is not None else torch.rand(size=(7,), device=DEVICE)
        self._face_apriltags = face_apriltags or []
        self._ema_filter = EMAFilter()
        self._material = material
        self._color = color
        self._moveable = moveable

    @property
    def pose(self) -> torch.Tensor:
        """The pose of the cube in the world frame."""
        if self._pose is None:
            raise AttributeError("The pose is not known.")
        return self._pose

    @pose.setter
    def pose(self, pose: torch.Tensor) -> None:
        """Set the pose of the cube in the world frame."""
        self._pose = self._ema_filter.update(pose)

    @property
    def aabb(self) -> torch.Tensor:
        """The axis-aligned bounding box of the cube."""
        return torch.cat([self._pose[:3] - self._size / 2.0, self._pose[:3] + self._size / 2.0], dim=-1)

    @property
    def object_type(self) -> str:
        """The type of the cube."""
        return "block"

    @property
    def size(self) -> float:
        """The size of the cube."""
        return self._size

    @override
    def is_pose_known(self) -> bool:
        return self._pose is not None

    def get_corners(self) -> torch.Tensor:
        """Get the corners of the cube in the world frame.

        Returns:
            The corners of the cube in the world frame. Shape is (8, 3).

        """
        # 8 corners in local frame, at ±size/2 along each axis
        half = self._size / 2
        offsets = (
            torch.tensor(
                [
                    [-1, -1, -1],  # 0: front-right-bottom
                    [+1, -1, -1],  # 1: back-right-bottom
                    [-1, +1, -1],  # 2: front-left-bottom
                    [+1, +1, -1],  # 3: back-left-bottom
                    [-1, -1, +1],  # 4: front-right-top
                    [+1, -1, +1],  # 5: back-right-top
                    [-1, +1, +1],  # 6: front-left-top
                    [+1, +1, +1],  # 7: back-left-top
                ],
                dtype=self._pose.dtype,
                device=self._pose.device,
            )
            * half
        )  # (8, 3)

        pos = self._pose[:3]  # (3,)
        quat = self._pose[3:]  # (4,) (w, x, y, z)

        # Rotate each corner offset into world frame, then translate
        return pos + quat_apply(quat.unsqueeze(0).expand(8, -1), offsets)  # (8, 3)

    def get_face_apriltags(self) -> list[dict[str, Any]]:
        """Get the AprilTags configurations for the faces of the cube."""
        return self._face_apriltags

    @classmethod
    def pose_from_face_center(
        cls,
        face: Literal["front", "back", "left", "right", "top", "bottom"] | int,
        center_pos: torch.Tensor,  # (x, y, z)
        normal: torch.Tensor,  # (x, y, z)
        up: torch.Tensor,  # (x, y, z)
        size: float,
    ) -> torch.Tensor:
        """Get a cube pose from the center of a face and the normal vector in the world frame.

        The pose is computed under the convention that with identity rotation:
            - face=front|back|left|right: the up vector is toward the top face
            - face=top: the up vector is toward the back face
            - face=bottom: the up vector is toward the front face
        Args:
            face: The face of the cube. Can be a string or an integer.
            center_pos: The position of the center of the face in the world frame.
            normal: The normal vector of the face in the world frame.
            up: The up vector of the face in the world frame.
            size: The side length of the cube.

        Returns:
            The pose of the cube in the world frame.

        """
        if isinstance(face, str):
            face = cls.FACE_INDICES[face]

        # Rotation of the face frame in the world = look_rotation(normal, up)
        q_world_face = cls._look_rotation(normal, up)

        # Rotation of the face frame in the cube body frame = look_rotation(body_normal, body_up)
        q_body_face = cls._look_rotation(
            cls.FACE_NORMALS[face],
            cls.FACE_UPS[face],
        ).to(normal.device, dtype=torch.float32)

        # q_world_cube = q_world_face * q_body_face^{-1}
        q_world_cube = quat_mul(q_world_face, quat_inv(q_body_face))

        # Cube center = face center - 0.5 * size * normal
        cube_pos = center_pos - 0.5 * size * normalize(normal)

        return torch.cat([cube_pos, q_world_cube])

    @staticmethod
    def _look_rotation(normal: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Build a quaternion (w,x,y,z) from a normal (+Z) and up (+Y) vector."""
        z = normalize(normal)
        x = normalize(torch.cross(up, z, dim=-1))
        y = torch.cross(z, x, dim=-1)

        R = torch.stack([x, y, z], dim=1)  # noqa: N806
        return quat_from_matrix(R)

    def __str__(self) -> str:
        """Return a printable string."""
        return f"Cube | ID: {self.object_id} | Name: {self.name} | Center: {self.pose.cpu().numpy()[:3]}"


class Table(SceneObject):
    """A table in a scene."""

    def __init__(
        self,
        height: float = 0.0,
        init_pose: torch.Tensor | None = None,  # (x, y, z, w, x, y, z)
        name: str | None = None,
    ) -> None:
        """Initialize the table.

        Args:
            height: the height of the top of the table surface in the world frame
            init_pose: The initial pose of the Table in the world frame.

        """
        super().__init__(name=name, localizable=False)
        self._height = height
        self._pose = init_pose
        self._supportable = False

    @property
    def pose(self) -> torch.Tensor:
        """The pose of the table in the world frame."""
        return self._pose

    @pose.setter
    def pose(self, pose: torch.Tensor) -> None:
        """Set the pose of the table in the world frame."""
        self._pose = pose

    @property
    def object_type(self) -> str:
        """The type of the table."""
        return "table"

    def is_pose_known(self) -> bool:
        """If the pose of the table is known.

        Return false to avoid plotting.
        """
        return False

    @property
    def height(self) -> float:
        """The height of the top of the table in the world frame."""
        return self._height

    def __str__(self) -> str:
        """Return a printable string."""
        return f"Table | ID: {self.object_id} | Name: {self.name} | Height: {self.height}"


class Target(SceneObject):
    """A target in a scene."""

    def __init__(
        self,
        radius: float = 0.00508,
        init_pose: torch.Tensor | None = None,  # (x, y, z, w, x, y, z)
        name: str | None = None,
    ) -> None:
        """Initialize the target.

        Args:
            radius: radius of the target
            init_pose: The initial pose of the target in the world frame.

        """
        super().__init__(name=name, localizable=True)
        self._radius = radius
        self._pose = init_pose

    @property
    def pose(self) -> torch.Tensor:
        """The pose of the target in the world frame."""
        return self._pose

    @pose.setter
    def pose(self, pose: torch.Tensor) -> None:
        """Set the pose of the target in the world frame."""
        self._pose = pose

    @property
    def object_type(self) -> str:
        """The type of the table."""
        return "target"

    def is_pose_known(self) -> bool:
        """If the pose of the target is known.

        Return false to avoid plotting.
        """
        return False

    @property
    def radius(self) -> float:
        """The radius of the target."""
        return self._radius

    def __str__(self) -> str:
        """Return a printable string."""
        return f"Target | ID: {self.object_id} | Name: {self.name} | Center: {self.pose.cpu().numpu()[:2]}"


class Location(SceneObject):
    """A location in a scene."""

    def __init__(
        self,
        size: float = 0.05,
        init_pose: torch.Tensor | None = None,  # (x, y, z, w, x, y, z)
        name: str | None = None,
    ) -> None:
        """Initialize the target.

        Args:
            size: size of the location
            init_pose: The initial pose of the target in the world frame.

        """
        super().__init__(name=name, localizable=False)
        self._size = size
        self._pose = init_pose
        self._supportable = True
        self._hoverable = True

    @property
    def pose(self) -> torch.Tensor:
        """The pose of the target in the world frame."""
        return self._pose

    @pose.setter
    def pose(self, pose: torch.Tensor) -> None:
        """Set the pose of the target in the world frame."""
        self._pose = pose

    @property
    def object_type(self) -> str:
        """The type of the location."""
        return "location"

    def is_pose_known(self) -> bool:
        """If the pose of the target is known.

        Return false to avoid plotting.
        """
        return False

    @property
    def size(self) -> float:
        """The size of the location."""
        return self._size

    @property
    def aabb(self) -> torch.Tensor:
        """The axis-aligned bounding box of the location."""
        return torch.cat([self._pose[:3] - self._size / 2.0, self._pose[:3] + self._size / 2.0], dim=-1)

    def __str__(self) -> str:
        """Return a printable string."""
        return f"Target | ID: {self.object_id} | Name: {self.name} | Center: {self.pose.cpu().numpu()[:2]}"


class Sponge(SceneObject):
    """A sponge in a scene."""

    def __init__(
        self,
        size: float,
        init_pose: torch.Tensor | None = None,  # (x, y, z, w, x, y, z)
        name: str | None = None,
        color: str = "blue",
    ) -> None:
        """Initialize the sponge.

        Args:
            size: The side length of the sponge in meters.
            init_pose: The initial pose of the sponge in the world frame.

        """
        super().__init__(name=name, localizable=True)
        self._size = size
        self._pose = init_pose if init_pose is not None else torch.rand(size=(7,), device=DEVICE)
        self._ema_filter = EMAFilter()
        self._color = color
        self._deformable = True
        self._graspable = True
        self._wet = self._color == "blue"

    @property
    def pose(self) -> torch.Tensor:
        """The pose of the sponge in the world frame."""
        if self._pose is None:
            raise AttributeError("The pose is not known.")
        return self._pose

    @pose.setter
    def pose(self, pose: torch.Tensor) -> None:
        """Set the pose of the sponge in the world frame."""
        self._pose = self._ema_filter.update(pose)

    @property
    def aabb(self) -> torch.Tensor:
        """The axis-aligned bounding box of the sponge."""
        return torch.cat([self._pose[:3] - self._size / 2.0, self._pose[:3] + self._size / 2.0], dim=-1)

    @property
    def object_type(self) -> str:
        """The type of the sponge."""
        return "sponge"

    @property
    def size(self) -> float:
        """The size of the sponge."""
        return self._size

    @override
    def is_pose_known(self) -> bool:
        return self._pose is not None

    def __str__(self) -> str:
        """Return a printable string."""
        return f"Sponge | ID: {self.object_id} | Name: {self.name} | Center: {self.pose.cpu().numpy()[:3]}"


class Spill(SceneObject):
    """A water spill in a scene."""

    def __init__(
        self,
        size: float,
        init_pose: torch.Tensor | None = None,  # (x, y, z, w, x, y, z)
        name: str | None = None,
        color: str | None = None,
    ) -> None:
        """Initialize the spill.

        Args:
            size: The side length of the spill in meters.
            init_pose: The initial pose of the spill in the world frame.
            face_to_apriltag: The mapping from the spill's faces to the AprilTag IDs.
                - id: The ID of the AprilTag.
                - size: The size of the AprilTag in meters.
                - orientation: The rotation of the AprilTag around the normal vector.
                    On the sides, 0 means the tag is upright.
                    For "top", 0 means the top of the tag is near the back side
                    For "bottom", 0 means the bottom of the tag is near the front side.

        """
        super().__init__(name=name, localizable=True)
        self._size = size
        self._pose = init_pose if init_pose is not None else torch.rand(size=(7,), device=DEVICE)
        self._ema_filter = EMAFilter()
        self._color = color

        self._bbox = None
        self._deformable = False
        self._graspable = False
        self._supportable = False

    @property
    def pose(self) -> torch.Tensor:
        """The pose of the spill in the world frame."""
        if self._pose is None:
            raise AttributeError("The pose is not known.")
        return self._pose

    @pose.setter
    def pose(self, pose: torch.Tensor) -> None:
        """Set the pose of the spill in the world frame."""
        self._pose = self._ema_filter.update(pose)

    @property
    def aabb(self) -> torch.Tensor:
        """The axis-aligned bounding box of the spill."""
        return torch.cat([self._pose[:3] - self._size / 2.0, self._pose[:3] + self._size / 2.0], dim=-1)

    @property
    def object_type(self) -> str:
        """The type of the spill."""
        return "spill"

    @property
    def size(self) -> float:
        """The size of the spill."""
        return self._size

    @override
    def is_pose_known(self) -> bool:
        return self._pose is not None

    def __str__(self) -> str:
        """Return a printable string."""
        return f"Spill | ID: {self.object_id} | Name: {self.name} | Center: {self.pose.cpu().numpy()[:3]}"


class Plate(SceneObject):
    """A plate in a scene."""

    def __init__(
        self,
        size: float,
        init_pose: torch.Tensor | None = None,  # (x, y, z, w, x, y, z)
        name: str | None = None,
        color: str | None = None,
        dirty: bool | None = None,
    ) -> None:
        """Initialize the plate.

        Args:
            size: The side length of the plate in meters.
            init_pose: The initial pose of the plate in the world frame.

        """
        super().__init__(name=name, localizable=True)
        self._size = size
        self._pose = init_pose if init_pose is not None else torch.rand(size=(7,), device=DEVICE)
        self._ema_filter = EMAFilter()
        self._color = color
        self._supportable = True
        self._deformable = False
        self._hoverable = True
        self._dirty = dirty

    @property
    def pose(self) -> torch.Tensor:
        """The pose of the plate in the world frame."""
        if self._pose is None:
            raise AttributeError("The pose is not known.")
        return self._pose

    @pose.setter
    def pose(self, pose: torch.Tensor) -> None:
        """Set the pose of the plate in the world frame."""
        self._pose = self._ema_filter.update(pose)

    @property
    def aabb(self) -> torch.Tensor:
        """The axis-aligned bounding box of the plate."""
        return torch.cat([self._pose[:3] - self._size / 2.0, self._pose[:3] + self._size / 2.0], dim=-1)

    @property
    def object_type(self) -> str:
        """The type of the plate."""
        return "plate"

    @property
    def size(self) -> float:
        """The size of the plate."""
        return self._size

    @override
    def is_pose_known(self) -> bool:
        return self._pose is not None

    def __str__(self) -> str:
        """Return a printable string."""
        return f"Plate | ID: {self.object_id} | Name: {self.name} | Center: {self.pose.cpu().numpy()[:3]}"


class Bin(SceneObject):
    """A bin in a scene."""

    def __init__(
        self,
        size: float,
        init_pose: torch.Tensor | None = None,  # (x, y, z, w, x, y, z)
        name: str | None = None,
        color: str | None = None,
    ) -> None:
        """Initialize the bin.

        Args:
            size: The side length of the bin in meters.
            init_pose: The initial pose of the bin in the world frame.

        """
        super().__init__(name=name, localizable=True)
        self._size = size
        self._pose = init_pose if init_pose is not None else torch.rand(size=(7,), device=DEVICE)
        self._ema_filter = EMAFilter()
        self._color = color
        self._supportable = False
        self._deformable = False
        self._hoverable = True

    @property
    def pose(self) -> torch.Tensor:
        """The pose of the bin in the world frame."""
        if self._pose is None:
            raise AttributeError("The pose is not known.")
        return self._pose

    @pose.setter
    def pose(self, pose: torch.Tensor) -> None:
        """Set the pose of the bin in the world frame."""
        self._pose = self._ema_filter.update(pose)

    @property
    def aabb(self) -> torch.Tensor:
        """The axis-aligned bounding box of the bin."""
        return torch.cat([self._pose[:3] - self._size / 2.0, self._pose[:3] + self._size / 2.0], dim=-1)

    @property
    def object_type(self) -> str:
        """The type of the bin."""
        return "bin"

    @property
    def size(self) -> float:
        """The size of the bin."""
        return self._size

    @override
    def is_pose_known(self) -> bool:
        return self._pose is not None

    def __str__(self) -> str:
        """Return a printable string."""
        return f"Bin | ID: {self.object_id} | Name: {self.name} | Center: {self.pose.cpu().numpy()[:3]}"


class Can(SceneObject):
    """A can in a scene."""

    def __init__(
        self,
        size: float,
        init_pose: torch.Tensor | None = None,  # (x, y, z, w, x, y, z)
        name: str | None = None,
        color: str | None = None,
    ) -> None:
        """Initialize the can.

        Args:
            size: The side length of the can in meters.
            init_pose: The initial pose of the can in the world frame.

        """
        super().__init__(name=name, localizable=True)
        self._size = size
        self._pose = init_pose if init_pose is not None else torch.rand(size=(7,), device=DEVICE)
        self._ema_filter = EMAFilter()
        self._color = color
        self._deformable = False
        self._supportable = False

    @property
    def pose(self) -> torch.Tensor:
        """The pose of the can in the world frame."""
        if self._pose is None:
            raise AttributeError("The pose is not known.")
        return self._pose

    @pose.setter
    def pose(self, pose: torch.Tensor) -> None:
        """Set the pose of the can in the world frame."""
        self._pose = self._ema_filter.update(pose)

    @property
    def aabb(self) -> torch.Tensor:
        """The axis-aligned bounding box of the can."""
        return torch.cat([self._pose[:3] - self._size / 2.0, self._pose[:3] + self._size / 2.0], dim=-1)

    @property
    def object_type(self) -> str:
        """The type of the can."""
        return "bin"

    @property
    def size(self) -> float:
        """The size of the can."""
        return self._size

    @override
    def is_pose_known(self) -> bool:
        return self._pose is not None

    def __str__(self) -> str:
        """Return a printable string."""
        return f"Can | ID: {self.object_id} | Name: {self.name} | Center: {self.pose.cpu().numpy()[:3]}"
