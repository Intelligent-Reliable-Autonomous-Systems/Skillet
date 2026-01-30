"""
There are many different types of spaces that do not play nicely with type hints.
Space Specifications are a way to annotate the type of a space and associate with a Gymnasium space.
Additionally, it can track whether data is batched and whether it is a PyTorch tensor or numpy array.

Observation spaces describe the data observed from the environment.
Skill parameters spaces describe the parameters of a skill.
"""

# gym observation spaces types

# utility to batch an observation spec

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Mapping,
    NamedTuple,
    Protocol,
    Sequence,
    Type,
    TypeAlias,
    TypeVar,
    overload,
)
from typing_extensions import TypedDict

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    import torch
# try:
# except Exception:  # torch not available / typing-only env
#     class _FakeTorch:
#         tensor = Any
#     torch = _FakeTorch()

from jaxtyping import Float, Int, Bool

# =============================================
# Space specifications
# =============================================
Scalar: TypeAlias = int | float | bool
"""A scalar value."""
ListLike: TypeAlias = Sequence[Scalar]
"""A list-like sequence of scalars."""
SpaceItem: TypeAlias = Scalar | ListLike
"""A scalar or list-like value that can be stored in a space."""
SpaceItemNP: TypeAlias = Scalar | np.ndarray[Scalar]
"""A scalar or numpy array of scalars that can be stored in a space."""
SpaceItemTorch: TypeAlias = Scalar | torch.Tensor[Scalar]
"""A scalar or PyTorch tensor of scalars that can be stored in a space."""
SpaceValue: TypeAlias = SpaceItem | Mapping[str, SpaceItem]
"""A scalar or list-like value or dictionary of scalar or list-like values that can be stored in a space."""
SpaceValueNP: TypeAlias = SpaceItemNP | Mapping[str, SpaceItemNP]
"""A scalar or numpy array of scalars or dictionary of scalar or numpy array of scalars that can be stored in a space."""
SpaceValueTorch: TypeAlias = SpaceItemTorch | Mapping[str, SpaceItemTorch]
"""A scalar or PyTorch tensor of scalars or dictionary of scalar or PyTorch tensor of scalars that can be stored in a space."""
BatchedSpaceItem: TypeAlias = ListLike[SpaceItem]
"""A batched list-like sequence of scalar or list-like values that can be stored in a space."""
BatchedSpaceItemNP: TypeAlias = ListLike[SpaceItemNP]
"""A batched list-like sequence of numpy arrays of scalars that can be stored in a space."""
BatchedSpaceItemTorch: TypeAlias = ListLike[SpaceItemTorch]
"""A batched list-like sequence of PyTorch tensors of scalars that can be stored in a space."""
BatchedSpaceValue: TypeAlias = BatchedSpaceItem | Mapping[str, BatchedSpaceItem]
"""A batched scalar or list-like value or dictionary of batched scalar or list-like values that can be stored in a space."""
BatchedSpaceValueNP: TypeAlias = BatchedSpaceItemNP | Mapping[str, BatchedSpaceItemNP]
"""A batched scalar or numpy array of scalars or dictionary of batched scalar or numpy array of scalars that can be stored in a space."""
BatchedSpaceValueTorch: TypeAlias = BatchedSpaceItemTorch | Mapping[str, BatchedSpaceItemTorch]
"""A batched scalar or PyTorch tensor of scalars or dictionary of batched scalar or PyTorch tensor of scalars that can be stored in a space."""

TSpace = TypeVar("TSpace", bound=SpaceValue)
"""The generic type variable for a space type."""
@dataclass(frozen=True)
class SpaceSpecification(Generic[TSpace]):
    """
    Typed descriptor of the space of a data container.
    """
    space: gym.Space[Any]
    """The Gymnasium space that models the space of the data."""
    name: str
    """The name of the space."""
    is_torch: bool
    """Whether the space is a PyTorch tensor or numpy array."""
    is_batched: bool
    """Whether the space is batched."""
    n_envs: int | None = None
    """The number of environments in the batched space. If None, the space is not batched."""

    @overload
    def index(self, value: TSpace, env_ids: Sequence[bool]) -> TSpace: ...
    def index(self, value: TSpace, env_ids: Sequence[int]) -> TSpace:
        """Index the space value for the given environment ids.
        
        Args:
            value: The space value to index.
            env_ids: The environment ids to index the space value for.
                If a sequence of booleans, the environment ids are the indices of the True values.
                If a sequence of integers, select the elements at the given indices.
        """
        if isinstance(env_ids, Sequence[bool]):
            env_ids = np.where(env_ids)[0]
        if self.is_batched:
            if isinstance(value, Mapping):
                return {key: self.index(v, env_ids) for key, v in value.items()}
            elif isinstance(value, Sequence):
                return [self.index(v, env_ids) for v in value]
            else:
                return value[env_ids]
        else:
            return value

    def zeros(self) -> TSpace:
        """Create a zero-filled tensor or array conforming to the Gym space (including handling Dict spaces)."""

        def zeros_for_space(space: gym.Space[Any]) -> TSpace:
            if isinstance(space, gym.spaces.Box):
                if self.is_torch:
                    return torch.zeros(space.shape, dtype=space.dtype)
                else:
                    return np.zeros(space.shape, dtype=space.dtype)
            elif isinstance(space, gym.spaces.Dict):
                # Recursively fill dict space
                return {key: zeros_for_space(subspace) for key, subspace in space.spaces.items()}
            elif isinstance(space, gym.spaces.Discrete):
                if self.is_torch:
                    return torch.zeros((), dtype=torch.long)
                else:
                    return np.zeros((), dtype=int)
            elif isinstance(space, gym.spaces.MultiDiscrete):
                if self.is_torch:
                    return torch.zeros(space.nvec.shape, dtype=torch.long)
                else:
                    return np.zeros(space.nvec.shape, dtype=int)
            elif isinstance(space, gym.spaces.MultiBinary):
                if self.is_torch:
                    return torch.zeros(space.n, dtype=torch.long)
                else:
                    return np.zeros(space.n, dtype=int)
            else:
                raise NotImplementedError(f"zeros() not implemented for space type: {type(space)}")

        return zeros_for_space(self.space)

# =============================================
# Actions
# =============================================
Action: TypeAlias = SpaceItem
"""Represents an action in the environment."""
BatchedAction: TypeAlias = BatchedSpaceItem
"""A batched action in the environment."""
TAction = TypeVar("TAction", bound=Action)
# ActionSpec: TypeAlias = SpaceSpecification[TAction]
class ActionSpec(SpaceSpecification[TAction], Generic[TAction]):
    """The specification of an action space."""
    pass

def make_action_spec(
    obs_type: Type[TAction],
    space: gym.Space[TAction],
    name: str | None = None,
    is_torch: bool = False,
    is_batched: bool = False,
    n_envs: int | None = None,
) -> ActionSpec[TAction]:
    """
    Takes a Gymnasium space and returns an ActionSpec parameterized by the action type.
    """
    if name is None:
        if space:
            name = str(space)
    return ActionSpec[TAction](space=space, name=name, is_torch=is_torch, is_batched=is_batched, n_envs=n_envs)

# =============================================
# Observations
# =============================================
Observation: TypeAlias = SpaceValue
"""A scalar or list-like value or dictionary of scalar or list-like values that can be stored in a space."""
BatchedObservation: TypeAlias = BatchedSpaceValue
"""A batched scalar or list-like value or dictionary of batched scalar or list-like values that can be stored in a space."""
State: TypeAlias = Observation
"""The full state of the environment (full observability)."""

TObs = TypeVar("TObs", bound=Observation)
ObservationSpec: TypeAlias = SpaceSpecification[TObs]
"""The specification of an observation space."""

@overload
def make_observation_spec(space: gym.spaces.Box, *, name: str = "obs", is_torch: bool = False) -> ObservationSpec[Any]: ...
@overload
def make_observation_spec(space: gym.spaces.Dict, *, name: str = "obs", is_torch: bool = False) -> ObservationSpec[Mapping[str, Any]]: ...
@overload
def make_observation_spec(space: gym.Space[Any], *, name: str = "obs", is_torch: bool = False) -> ObservationSpec[Any]: ...


def make_observation_spec(space: gym.Space[Any], *, name: str = "obs", is_torch: bool = False) -> ObservationSpec[Any]:
    """
    Takes a Gymnasium space and returns an ObservationSpec parameterized by the
    observation object type produced by that space.

    Args:
        space: The Gymnasium space to create an observation spec for.
        name: The name of the observation spec.
        is_torch: If True, observation types will be torch.Tensor instead of np.ndarray.

    Supports:
      - spaces.Box  -> np.ndarray or torch.Tensor (based on is_torch)
      - spaces.Dict -> Mapping[str, Any] (runtime type: TypedDict class)
                     (recursively supports Box/Dict subspaces)
    """
    space_type = _runtime_space_type(space, name=name, is_torch=is_torch)
    return ObservationSpec[Any](space=space, space_type=space_type, name=name, is_torch=is_torch)

# =============================================
# Skill parameters
# =============================================
# Define parameter specifications that provide a structured way to describe the parameter space of a skill.

# Skill parameters are the parameters that are used to control the skill.
# Some skills may involve continuous parameterizations. Some skills may involve discrete parameterizations.

SkillParams: TypeAlias = SpaceValue
"""A scalar or list-like value or dictionary of scalar or list-like values that can be stored in a space."""
BatchedSkillParams: TypeAlias = BatchedSpaceValue
"""A batched scalar or list-like value or dictionary of batched scalar or list-like values that can be stored in a space."""

SkillParamsSpec: TypeAlias = SpaceSpecification[SkillParams]
"""The specification of a skill parameter space."""
BatchedSkillParamsSpec: TypeAlias = SpaceSpecification[BatchedSkillParams]
"""The specification of a batched skill parameter space."""

# Brainstorming parameter examples:
# Continuous-only: pick(xyz), place(xyzrpy), etc.
# - Box(n, float)
# Discrete: pick(box), place(box, table), etc.
# - Box(n, int)
# - List(n, object)
# Continuous and discrete: pick(box, xyz), place(box, table, xyzrpy), etc.
# Dictionary: pick(object=box, position=xyz), place(object=box, position=xyz, orientation=xyzrpy), etc.
# Batched continuous homogeneous: pick([xyz]), etc.
# Batched discrete homogeneous: 
# - unary: pick([box, table])
# - binary: pick([box, table], [ball, ground])
# Batched dictionary homogenous: pick(object=[box, table], position=[xyz, xyzrpy])
# Batched continuous heterogeneous: pick([xyz, xyzrpy]), etc. -> pad to max length to make it homogeneous
# Batched discrete heterogeneous: pick([box, table], [ball]), etc. -> list of lists needn't be padded, but might be useful


# =============================================
# Common observation space type aliases and definitions
# =============================================
Array1D: TypeAlias = Float[np.ndarray, "n"]
"""Represents a 1D array of floats ndarray[(n,), float]."""
BatchedArray1D: TypeAlias = Float[np.ndarray, "b n"]
"""Represents a batched 1D array of floats ndarray[(b, n), float]."""
ArrayEmpty: TypeAlias = Float[np.ndarray, "0"]
"""Represents an empty 1D array of floats ndarray[(0,), float]."""
BatchedArrayEmpty: TypeAlias = Float[np.ndarray, "b 0"]
"""Represents a batched empty 1D array of floats ndarray[(b, 0), float]."""
Array3: TypeAlias = Float[np.ndarray, "3"]
"""Represents a 3D position or orientation in Cartesian space as a 1x3 array."""
BatchedArray3: TypeAlias = Float[np.ndarray, "b 3"]
"""Represents a batched 3D position or orientation in Cartesian space as a (b, 3) array."""
Array6: TypeAlias = Float[np.ndarray, "6"]
"""Represents a 6D position and orientation in Cartesian space as a 1x6 array."""
BatchedArray6: TypeAlias = Float[np.ndarray, "b 6"]
"""Represents a batched 6D position and orientation in Cartesian space as a (b, 6) array."""
Array7: TypeAlias = Float[np.ndarray, "7"]
"""Represents a 7D position and orientation in Cartesian space with a weight as a 1x7 array."""
BatchedArray7: TypeAlias = Float[np.ndarray, "b 7"]
"""Represents a batched 7D position and orientation in Cartesian space with a weight as a (b, 7) array."""
ParamDC: TypeAlias = NamedTuple("ParamDC", [("discrete", Int[np.ndarray, "m"]), ("continuous", Float[np.ndarray, "n"])])
"""Represents a skill parameter set with m discrete parameters and n continuous parameters."""
BatchedParamDC: TypeAlias = NamedTuple("BatchedParamDC", [("discrete", Int[np.ndarray, "b m"]), ("continuous", Float[np.ndarray, "b n"])])
"""Represents a batched skill parameter set with m discrete parameters and n continuous parameters."""
Tensor1D: TypeAlias = Float[torch.Tensor, "n"]
"""Represents a 1D array of floats torch.Tensor[(n,), float]."""
BatchedTensor1D: TypeAlias = Float[torch.Tensor, "b n"]
"""Represents a batched 1D array of floats torch.Tensor[(b, n), float]."""
TensorEmpty: TypeAlias = Float[torch.Tensor, "0"]
"""Represents an empty 1D array of floats torch.Tensor[(0,), float]."""
BatchedTensorEmpty: TypeAlias = Float[torch.Tensor, "b 0"]
"""Represents a batched empty 1D array of floats torch.Tensor[(b, 0), float]."""
Tensor3: TypeAlias = Float[torch.Tensor, "3"]
"""Represents a 3D position or orientation in Cartesian space as a 1x3 tensor."""
BatchedTensor3: TypeAlias = Float[torch.Tensor, "b 3"]
"""Represents a batched 3D position or orientation in Cartesian space as a (b, 3) tensor."""
Tensor6: TypeAlias = Float[torch.Tensor, "6"]
"""Represents a 6D position and orientation in Cartesian space as a 1x6 tensor."""
BatchedTensor6: TypeAlias = Float[torch.Tensor, "b 6"]
"""Represents a batched 6D position and orientation in Cartesian space as a (b, 6) tensor."""
Tensor7: TypeAlias = Float[torch.Tensor, "7"]
"""Represents a 7D position and orientation in Cartesian space with a weight as a 1x7 tensor."""
BatchedTensor7: TypeAlias = Float[torch.Tensor, "b 7"]
"""Represents a batched 7D position and orientation in Cartesian space with a weight as a (b, 7) tensor."""
ParamDC_Torch: TypeAlias = NamedTuple("ParamDC_Torch", [("discrete", Int[torch.Tensor, "m"]), ("continuous", Float[torch.Tensor, "n"])])
"""Represents a skill parameter set with m discrete parameters and n continuous parameters as a PyTorch tensor."""
BatchedParamDC_Torch: TypeAlias = NamedTuple("BatchedParamDC_Torch", [("discrete", Int[torch.Tensor, "b m"]), ("continuous", Float[torch.Tensor, "b n"])])
"""Represents a batched skill parameter set with m discrete parameters and n continuous parameters as a PyTorch tensor."""

test: ActionSpec[Float[np.ndarray, "0"]] = ActionSpec[Float[np.ndarray, "0"]](
    space=gym.spaces.Box(low=0.0, high=1.0, shape=(0,), dtype="float32"),
    name="ArrayEmpty", is_torch=False, is_batched=False,
)
class CommonSpecs:
    # Predefined static types for common observation spaces
    ArrayEmpty: ClassVar[SpaceSpecification[ArrayEmpty]] = ActionSpec[ArrayEmpty](
        space=gym.spaces.Box(low=0.0, high=1.0, shape=(0,), dtype="float32"),
        name="ArrayEmpty", is_torch=False, is_batched=False,
    )
    TensorEmpty: ClassVar[SpaceSpecification[TensorEmpty]] = SpaceSpecification[TensorEmpty](
        space=gym.spaces.Box(low=0.0, high=1.0, shape=(0,), dtype="float32"),
        name="TensorEmpty", is_torch=True, is_batched=False,
    )
    Array1D: ClassVar[SpaceSpecification[Array1D]] = SpaceSpecification[Array1D](
        space=None,
        name="Array1D", is_torch=False, is_batched=False,
    )
    BatchedArray1D: ClassVar[SpaceSpecification[BatchedArray1D]] = SpaceSpecification[BatchedArray1D](
        space=None,
        name="BatchedArray1D", is_torch=False, is_batched=True,
    )
    Tensor1D: ClassVar[SpaceSpecification[Tensor1D]] = SpaceSpecification[Tensor1D](
        space=None,
        name="Tensor1D",
        is_torch=True,
        is_batched=False,
    )
    BatchedTensor1D: ClassVar[SpaceSpecification[BatchedTensor1D]] = SpaceSpecification[BatchedTensor1D](
        space=None,
        name="BatchedTensor1D",
        is_torch=True,
        is_batched=True,
    )

    ParamDC: ClassVar[SpaceSpecification[ParamDC]] = SpaceSpecification[ParamDC](
        space=None,
        name="ParamDC", is_torch=False, is_batched=False,
    )
    BatchedParamDC: ClassVar[SpaceSpecification[BatchedParamDC]] = SpaceSpecification[BatchedParamDC](
        space=None,
        name="BatchedParamDC", is_torch=False, is_batched=True,
    )
    ParamDC_Torch: ClassVar[SpaceSpecification[ParamDC_Torch]] = SpaceSpecification[ParamDC_Torch](
        space=None,
        name="ParamDC_Torch", is_torch=True, is_batched=False,
    )
    BatchedParamDC_Torch: ClassVar[SpaceSpecification[BatchedParamDC_Torch]] = SpaceSpecification[BatchedParamDC_Torch](
        space=None,
        name="BatchedParamDC_Torch", is_torch=True, is_batched=True,
    )

# =============================================
# Utility functions
# =============================================

def _box_space_type(space: gym.spaces.Box, *, is_torch: bool = False) -> Type[Float | Int | Bool]:
    """
    Return the space type for a Box space.
    
    Args:
        space: The Box space to get the space type for.
        is_torch: If True, return torch.Tensor type instead of np.ndarray.
    
    Returns:
        The space type for the Box space.
    """
    if is_torch:
        if torch is None:
            raise ValueError("torch is not available")
        arr_type = torch.Tensor
    else:
        if np is None:
            raise ValueError("numpy is not available")
        arr_type = np.ndarray
    
    dtype = space.dtype
    match dtype.kind:
        case "f":
            jaxtype_cls = Float
        case "i":
            jaxtype_cls = Int
        case "b":
            jaxtype_cls = Bool
        case _:
            raise ValueError(f"Unsupported dtype kind: {dtype.kind} for {dtype}")
    
    shape = " ".join([str(s) for s in space.shape])
    space_type = jaxtype_cls[arr_type, shape]
    return space_type


def _typed_dict_for_space_dict(d: gym.spaces.Dict, *, name: str, is_torch: bool = False) -> Type[TypedDict]:
    """
    Create a runtime TypedDict class for a spaces.Dict.

    Caveat: type checkers generally won't infer key/value types from a runtime
    factory; you can still use this for runtime validation/docs.
    
    Args:
        d: The Dict space to create a TypedDict for.
        name: The name of the TypedDict class.
        is_torch: If True, nested Box spaces will use torch.Tensor types.
    """
    fields: dict[str, type[Any]] = {}
    for k, subspace in d.spaces.items():
        fields[k] = _runtime_space_type(subspace, name=f"{name}_{k}", is_torch=is_torch)
    return TypedDict(name, fields)  # type: ignore[return-value]


def _runtime_space_type(space: gym.Space[Any], *, name: str, is_torch: bool = False) -> type[Any]:
    if isinstance(space, gym.spaces.Box):
        return _box_space_type(space, is_torch=is_torch)
    if isinstance(space, gym.spaces.Dict):
        return _typed_dict_for_space_dict(space, name=f"{name}Dict", is_torch=is_torch)
    raise NotImplementedError(f"Unsupported space type: {type(space).__name__}")


# ---- example ----
if __name__ == "__main__":
    box = gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype="float32")
    spec_box = make_observation_spec(box, name="vec8", is_torch=True)
    print(spec_box.obs_type)  # <class 'numpy.ndarray'> (if numpy available)

    d = gym.spaces.Dict(
        {
            "key1": gym.spaces.Box(low=0, high=10, shape=(), dtype="int32"),
            "key2": gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype="float32"),
        }
    )
    spec_dict = make_observation_spec(d, name="obs")
    print(spec_dict.obs_type)  # <class '__main__.obsDict'> (a runtime TypedDict class)