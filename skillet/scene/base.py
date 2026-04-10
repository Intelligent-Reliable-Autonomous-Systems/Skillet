from abc import abstractmethod
from collections import defaultdict

import torch
import numpy as np


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
        self.objects = objects or []
        for object in self.objects:
            object._object_id = self._object_id_autoincrement
            self._object_id_autoincrement += 1
            object._type_id = self._type_id_autoincrement[object.type_name]
            self._type_id_autoincrement[object.type_name] += 1
        self.closed_set = closed_set
        self.bounds = bounds
        self.contains_objects = contains_objects

    def add_objects(self, objects: list[SceneObject] | None = None) -> None:
        """Add objects to the scene."""
        [self.objects.append(o) for o in objects]
        for object in self.objects:
            object._object_id = self._object_id_autoincrement
            self._object_id_autoincrement += 1
            object._type_id = self._type_id_autoincrement[object.type_name]
            self._type_id_autoincrement[object.type_name] += 1

    def reset(self, task: str) -> None:
        """Reset the scene."""
        pass

    def perceive(self) -> None:
        """Perceive the scene."""
        pass

    def get_target_by_spec(self, spec: str | int) -> SceneObject:
        """Return the target by the specification, either string or int."""
        if isinstance(spec, str) or isinstance(spec, np.ndarray):  # TODO make this better
            spec = str(spec)
            for obj in self.objects:
                if spec in obj.name:  # Need to make sure this wont overlap with other names
                    return obj
        elif isinstance(spec, int):
            return self.objects[spec]
        else:
            raise ValueError(f"`{spec}` not found in Scene.")

        return None

    @property
    def table(self) -> SceneObject | None:
        """Return the table."""
        from skillet.scene.cube import Table

        for obj in self.objects:
            if isinstance(obj, Table):
                return obj
        return None

    def serialize_scene_poses(self) -> None:
        """Return a numpy array of poses for all the objects in the scene."""
        poses = []
        obj_ids = []
        names = []
        for obj in self.objects:
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
        for ob in self.objects:
            print_str += (
                f"{ob.name} | ID: {ob.object_id} | Pose: {ob.pose.cpu().numpy() if ob.pose is not None else None}\n"
            )
        return print_str
