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

from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Iterator,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    Sequence,
    Type,
    TypeAlias,
    TypeVar,
    overload,
    runtime_checkable,
)
from typing_extensions import TypedDict

import gymnasium as gym
import numpy as np

import torch
# try:
# except Exception:  # torch not available / typing-only env
#     class _FakeTorch:
#         tensor = Any
#     torch = _FakeTorch()

from jaxtyping import Float, Int, Bool, Shaped

# =============================================
# Space specifications
# =============================================
Scalar: TypeAlias = int | float | bool
"""A scalar value."""
ListLike: TypeAlias = Sequence[Scalar]
"""A list-like sequence of scalars."""
@runtime_checkable
class ArrayLike(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    # dtype is tricky across libs; torch dtype isn't str.
    # If you don't need it statically, omit it.
    # Otherwise use `object` or a union of known dtype types.
    @property
    def dtype(self) -> object: ...

    def __add__(self, other: object) -> ArrayLike: ...
    def __mul__(self, other: object) -> ArrayLike: ...
    def astype(self, dtype: object) -> ArrayLike: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[ArrayLike]: ...
    def __or__(self, other: ArrayLike) -> ArrayLike: ...
    def __and__(self, other: ArrayLike) -> ArrayLike: ...
    def __bool__(self) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __le__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    
SpaceItem: TypeAlias = Scalar | ArrayLike
"""A scalar or list-like value that can be stored in a space."""
SpaceItemNP: TypeAlias = Scalar | np.ndarray[Scalar]
"""A scalar or numpy array of scalars that can be stored in a space."""
SpaceItemTorch: TypeAlias = Scalar | Shaped[torch.Tensor, "..."]
"""A scalar or PyTorch tensor of scalars that can be stored in a space."""
SpaceValue: TypeAlias = SpaceItem | Mapping[str, SpaceItem]
"""A scalar or list-like value or dictionary of scalar or list-like values that can be stored in a space."""
SpaceValueNP: TypeAlias = SpaceItemNP | Mapping[str, SpaceItemNP]
"""A scalar or numpy array of scalars or dictionary of scalar or numpy array of scalars that can be stored in a space."""
SpaceValueTorch: TypeAlias = SpaceItemTorch | Mapping[str, SpaceItemTorch]
"""A scalar or PyTorch tensor of scalars or dictionary of scalar or PyTorch tensor of scalars that can be stored in a space."""
BatchedSpaceItem: TypeAlias = Shaped[ArrayLike, "b ..."]
"""A batched list-like sequence of scalar or list-like values that can be stored in a space."""
BatchedSpaceItemNP: TypeAlias = Shaped[np.ndarray, "b ..."]
"""A batched list-like sequence of numpy arrays of scalars that can be stored in a space."""
BatchedSpaceItemTorch: TypeAlias = Shaped[torch.Tensor, "b ..."]
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
    """The number of environments in the batched space. 
    
    By default (None), the batch size is inferred from the first dimension of the space. 
        Be careful to pass a pre-batched space.
    If -1, the batch size is unknown. Assumes the space is not batched.
    """
    device: torch.device | None = None
    """The device of the space. If None, the space is on the default device."""

    def __post_init__(self):
        # Infer the batch size from the space if it is not specified.
        if self.is_batched and self.n_envs is None:
            space_shape = self.space.shape
            if space_shape is None:
                if isinstance(self.space, gym.spaces.Dict):
                    space = next(iter(self.space.spaces.values()))
                    space_shape = space.shape
                    if space_shape is None:
                        raise ValueError(f"Cannot infer batch size from dict space {self.space} because the shape of the first subspace is unknown.")
            if len(space_shape) == 0:
                raise ValueError(f"Cannot infer batch size from space {self.space} because the shape is empty. The space is not batched.")
            object.__setattr__(self, "n_envs", space_shape[0])

        # Set default device if not specified and is torch
        if self.is_torch and self.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            object.__setattr__(self, "device", device)

    def index(self, value: TSpace, env_ids: Sequence[int] | Sequence[bool]) -> TSpace:
        """Index the space value for the given environment ids.
        
        Args:
            value: The space value to index.
            env_ids: The environment ids to index the space value for.
                If a sequence of booleans, use boolean mask indexing to select the elements.
                If a sequence of integers, select the elements at the given indices.
        """
        if not self.is_batched:
            raise ValueError("Cannot index a non-batched space.")
        if self.is_torch:
            env_ids = torch.as_tensor(env_ids, device=self.device)
        else:
            env_ids = np.asarray(env_ids)
        if env_ids.ndim == 0:
            raise ValueError("env_ids must be a sequence of booleans or integers.")
        if isinstance(value, Mapping):
            return {key: self.index(v, env_ids) for key, v in value.items()}
        if self.is_torch:
            value = torch.as_tensor(value, device=self.device)
        else:
            value = np.asarray(value)
        if value.ndim == 0:
            raise ValueError(f"value {value} cannot be a scalar.")
        return value[env_ids]

    @overload
    def zeros(self) -> TSpace: ...
    @overload
    def zeros(self, shape: tuple[int, ...]) -> ArrayLike: ...
    @overload
    def zeros(self, shape: tuple[int, ...], dtype: torch.dtype | np.dtype) -> ArrayLike: ...

    def zeros(self, shape: Any = None, dtype: Any = None) -> Any:
        """Create a zero-filled tensor or array conforming to the Gym space (including handling Dict spaces).
        
        Optionally specify the shape of the tensor or array.
        """

        if shape is not None:
            dtype = dtype or self.space.dtype
            if shape[0] == -1:
                if self.n_envs == -1:
                    raise ValueError("n_envs not specified. Cannot infer shape with first dimension -1. Use with_n_envs() to set the batch size.")
                shape = (self.n_envs, *shape[1:])
            if self.is_torch:
                return torch.zeros(shape, dtype=as_torch_dtype(dtype), device=self.device)
            else:
                return np.zeros(shape, dtype=dtype)

        if self.is_batched and self.n_envs == -1:
            raise ValueError("Cannot create zeros for a variable batch size space. Use with_n_envs() to set the batch size.")

        def zeros_for_space(space: gym.Space[Any]) -> TSpace:
            if isinstance(space, gym.spaces.Box):
                if self.is_torch:
                    return torch.zeros(space.shape, dtype=as_torch_dtype(space.dtype), device=self.device)
                else:
                    return np.zeros(space.shape, dtype=space.dtype)
            elif isinstance(space, gym.spaces.Dict):
                # Recursively fill dict space
                return {key: zeros_for_space(subspace) for key, subspace in space.spaces.items()}
            elif isinstance(space, gym.spaces.Discrete):
                if self.is_torch:
                    return torch.zeros((), dtype=torch.long, device=self.device)
                else:
                    return np.zeros((), dtype=int)
            elif isinstance(space, gym.spaces.MultiDiscrete):
                if self.is_torch:
                    return torch.zeros(space.nvec.shape, dtype=torch.long, device=self.device)
                else:
                    return np.zeros(space.nvec.shape, dtype=int)
            elif isinstance(space, gym.spaces.MultiBinary):
                if self.is_torch:
                    return torch.zeros(space.n, dtype=torch.long, device=self.device)
                else:
                    return np.zeros(space.n, dtype=int)
            else:
                raise NotImplementedError(f"zeros() not implemented for space type: {type(space)}")

        return zeros_for_space(self.space)

    def sample(self) -> TSpace:
        """Sample a random value from the space."""
        sampled = self.space.sample()
        if self.is_torch:
            return torch.tensor(sampled, device=self.device)
        return sampled

    def with_n_envs(self, n_envs: int) -> SpaceSpecification[TSpace]:
        """Return a new space specification with the given number of environments."""
        if not self.is_batched:
            raise ValueError("Cannot set n_envs for a non-batched space.")
        if self.n_envs != -1 and self.n_envs != n_envs:
            raise ValueError(f"Cannot set n_envs to a different value {n_envs} than the current value {self.n_envs}.")
        if self.n_envs == -1:
            space = gym.vector.utils.space_utils.batch_space(self.space, n_envs)
        else:
            space = self.space
        return replace(self, space=space, n_envs=n_envs)

    def n_envs_from(self, value: TSpace) -> int:
        """Return the number of environments from the given value."""
        if not self.is_batched:
            raise ValueError("Cannot get n_envs from a non-batched space.")
        if isinstance(value, Mapping):
            return next(iter(value.values())).shape[0]
        return value.shape[0]

# =============================================
# Actions
# =============================================
Action: TypeAlias = SpaceItem
"""Represents an action in the environment."""
ActionTorch: TypeAlias = Float[torch.Tensor, "n"]
"""Represents an action in the environment as a PyTorch tensor."""
BatchedAction: TypeAlias = BatchedSpaceItem
"""A batched action in the environment."""
BatchedActionTorch: TypeAlias = Float[torch.Tensor, "b n"]
"""A batched action in the environment as a PyTorch tensor."""
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
ObservationTorch: TypeAlias = SpaceValueTorch
BatchedObservation: TypeAlias = BatchedSpaceValue
"""A batched scalar or list-like value or dictionary of batched scalar or list-like values that can be stored in a space."""
BatchedObservationTorch: TypeAlias = BatchedSpaceValueTorch
"""A batched scalar or list-like value or dictionary of batched scalar or list-like values that can be stored in a space as a PyTorch tensor."""
State: TypeAlias = Observation
"""The full state of the environment (full observability)."""

TObs = TypeVar("TObs", bound=Observation)
class ObservationSpec(SpaceSpecification[TObs], Generic[TObs]):
    """The specification of an observation space."""
    pass

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

TSkillParams = TypeVar("TSkillParams", bound=SkillParams)
class SkillParamsSpec(SpaceSpecification[TSkillParams], Generic[TSkillParams]):
    """The specification of a skill parameter space."""
    pass

TBSkillParams = TypeVar("TBSkillParams", bound=BatchedSkillParams)
class BatchedSkillParamsSpec(SpaceSpecification[TBSkillParams], Generic[TBSkillParams]):
    """The specification of a batched skill parameter space."""
    pass

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
    ArrayEmpty: ClassVar[SpaceSpecification[ArrayEmpty]] = SpaceSpecification[ArrayEmpty](
        space=gym.spaces.Box(low=0.0, high=1.0, shape=(0,), dtype="float32"),
        name="ArrayEmpty", is_torch=False, is_batched=False,
    )
    BatchedArrayEmpty: ClassVar[SpaceSpecification[BatchedArrayEmpty]] = SpaceSpecification[BatchedArrayEmpty](
        space=gym.spaces.Box(low=0.0, high=1.0, shape=(0,), dtype="float32"),
        name="BatchedArrayEmpty", is_torch=False, is_batched=True, n_envs=-1,
    )

# =============================================
# Utility functions
# =============================================

def as_torch_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is int:
        return torch.int64
    if dtype is float:
        return torch.float32
    if dtype is bool:
        return torch.bool
    if isinstance(dtype, np.dtype):
        if dtype == np.int32:
            return torch.int32
        if dtype == np.int64:
            return torch.int64
        if dtype == np.float32:
            return torch.float32
        if dtype == np.float64:
            return torch.float64
        if dtype.kind == "b":
            return torch.bool
    raise ValueError(f"Unsupported dtype: {dtype}")

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