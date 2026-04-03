from abc import abstractmethod
from collections import defaultdict

import torch


class SceneObject:
    """A scene object is a 3D object in a scene."""

    def __init__(self, object_id: int) -> None:
        """Initialize the scene object.

        Args:
            object_id: The ID of this instance with respect to the scene.

        """
        self._object_id = object_id
        self._type_id = -1

    @property
    @abstractmethod
    def pose(self) -> torch.Tensor:
        """The pose of the object in the world frame."""
        raise NotImplementedError

    @property
    @abstractmethod
    def aabb(self) -> torch.Tensor:
        """The axis-aligned bounding box of the object."""
        raise NotImplementedError

    @property
    def type_name(self) -> str:
        """The type of the object."""
        return "object"

    @property
    def type_id(self) -> int:
        """The ID of this instance with respect to its type.

        The type_id is a monotonically increasing integer starting from 0 for each type.

        Returns:
            The type ID of this instance. -1 if the type ID is not initialized.

        """
        return self._type_id

    @property
    def object_id(self) -> int:
        """The ID of this instance with respect to the scene."""
        return self._object_id

    @property
    def identifier(self) -> str:
        """The identifier of this instance."""
        if self.type_id == -1:
            raise AttributeError("The type ID is not initialized.")
        return f"{self.type_name}_{self.type_id}"

    @abstractmethod
    def is_pose_known(self) -> bool:
        """Whether the pose of the object is known."""
        raise NotImplementedError


class Scene:
    """A scene is a collection of objects in a 3D space."""

    def __init__(
        self,
        objects: list[SceneObject] | None = None,
        closed_set: bool = True,
        bounds: tuple[float, float, float, float, float, float] | None = None,
    ) -> None:
        """Initialize the scene.

        Args:
            objects: The known initial objects in the scene.
            closed_set: Whether the set of known objects is complete.
            bounds: The bounds of the scene.

        """
        self._object_id_autoincrement = 0
        self._type_id_autoincrement = defaultdict[str, int](int)
        self.objects = objects or []
        for object in self.objects:
            object._object_id = self._object_id_autoincrement
            self._object_id_autoincrement += 1
            object._type_id = self._type_id_autoincrement[object.type_name]
            self._type_id_autoincrement[object.type_name] += 1
        self.closed_set = closed_set
        self.bounds = bounds

    def reset(self, task: str) -> None:
        """Reset the scene."""
        pass

    def perceive(self) -> None:
        """Perceive the scene."""
        pass
