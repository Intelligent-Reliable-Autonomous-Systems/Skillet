from abc import abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np
import torch


class SceneObject:
    """A scene object is a 3D object in a scene."""

    def __init__(self, object_id: int | None = None, name: str | None = None) -> None:
        """Initialize the scene object.

        Args:
            object_id: The ID of this instance with respect to the scene.

        """
        self._object_id = object_id
        self._type_id = -1
        self._name = name

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

    @property
    def name(self) -> str:
        """The name of the object assigned from the scene/VLM.

        Usually in the form <attribute>_<type>, i.e. purple_block
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Set the name of the object."""
        self._name = name

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
        contains_objects: bool = False,
    ) -> None:
        """Initialize the scene.

        Args:
            objects: The known initial objects in the scene.
            closed_set: Whether the set of known objects is complete.
            bounds: The bounds of the scene.

        """
        self._object_id_autoincrement = 0
        self._type_id_autoincrement = defaultdict[str, int](int)
        self._objects = objects or []
        for object in self._objects:
            object._object_id = self._object_id_autoincrement
            self._object_id_autoincrement += 1
            object._type_id = self._type_id_autoincrement[object.type_name]
            self._type_id_autoincrement[object.type_name] += 1
        self.closed_set = closed_set
        self.bounds = bounds
        self.contains_objects = contains_objects
        self._tcp_pose = None
        self._gripper_pos = None
        self._goal = None

    @property
    def objects(self) -> list[SceneObject]:
        return self._objects

    @property
    def object_ids(self) -> list[int]:
        return [obj.object_id for obj in self._objects]

    @property
    def object_names(self) -> list[int]:
        return [obj.name for obj in self._objects]

    @property
    def table(self) -> SceneObject | None:
        """Return the table."""
        from skillet.scene.cube import Table

        for obj in self._objects:
            if isinstance(obj, Table):
                return obj
        return None

    @property
    def tcp_pose(self) -> torch.Tensor:
        return self._tcp_pose

    @tcp_pose.setter
    def tcp_pose(self, pose: torch.Tensor):
        self._tcp_pose = pose

    @property
    def gripper_pos(self) -> torch.Tensor:
        return self._gripper_pos

    @gripper_pos.setter
    def gripper_pos(self, pos: torch.Tensor):
        self._gripper_pos = pos

    @property
    def goal(self) -> dict[str, Any]:
        return self._goal

    @goal.setter
    def goal(self, goal: dict[str, Any]) -> None:
        self._goal = goal

    def get_objects_from_id(self, id_list: list[int] | np.ndarray) -> list[SceneObject]:
        """Get a list of objects from the scene by ID."""
        return [self._objects[self.object_ids.index(i)] for i in id_list]

    def add_objects(self, objects: list[SceneObject] | None = None) -> None:
        """Add objects to the scene."""
        [self._objects.append(o) for o in objects]
        for object in self._objects:
            object._object_id = self._object_id_autoincrement
            self._object_id_autoincrement += 1
            object._type_id = self._type_id_autoincrement[object.type_name]
            self._type_id_autoincrement[object.type_name] += 1

    def resolve_names_to_ids(self, obj_names: list[str]) -> list[int]:
        """Resolve object names to object ids for a list of object names.

        Args:
            obj_names: list of object names

        Returns:
            list of object ids as integers

        """
        return [self._objects[self.object_names.index(n)].object_id for n in obj_names]

    def get_object_names(self, obj_type: SceneObject) -> list[str]:
        """Get the name of all objecs of a specific type."""
        ob_names = []
        for ob in self._objects:
            if not isinstance(ob, obj_type):
                continue
            ob_names.append(ob.name)
        return ob_names

    def serialize_scene_poses(self) -> None:
        """Return a numpy array of poses for all the objects in the scene."""
        poses = []
        obj_ids = []
        names = []
        for obj in self._objects:
            poses.append(obj.pose.cpu().numpy())
            obj_ids.append(obj.object_id)
            names.append(obj.name)

        return {
            "poses": np.asarray(poses),
            "ids": np.asarray(obj_ids),
            "bounds": np.asarray(self.bounds),
            "names": np.asarray(names),
        }

    def __str__(self) -> str:
        """Print the scene."""
        print_str = ""
        np.set_printoptions(suppress=True, precision=3)
        for ob in self._objects:
            print_str += (
                f"{ob.name} | ID: {ob.object_id} | Pose: {ob.pose.cpu().numpy() if ob.pose is not None else None}\n"
            )
        return print_str
