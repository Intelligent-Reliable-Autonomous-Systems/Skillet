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
    Any,
    ClassVar,
    Generic,
    Mapping,
    NamedTuple,
    Protocol,
    Type,
    TypeVar,
    overload,
)
from typing_extensions import TypedDict

import gymnasium as gym
import numpy as np

try:
    import torch
except Exception:  # torch not available / typing-only env
    class _FakeTorch:
        tensor = Any
    torch = _FakeTorch()

from jaxtyping import Float, Int, Bool

# =============================================
# Space specifications
# =============================================
TSpace = TypeVar("TSpace")
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

# =============================================
# Observations
# =============================================
class Observation(Protocol):
    """Represents an observation from the environment."""
    ...
    
class BatchedObservation(Observation, Protocol):
    """Represents a batched observation from the environment."""
    ...
    
class State(Observation, Protocol):
    """Represents the full state of the environment (full observability)."""
    ...

TObs = TypeVar("TObs", bound=Observation)
type ObservationSpec[TObs: Observation] = SpaceSpecification[TObs]
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

class SkillParams(Protocol):
    """Represents the parameters of a skill."""
    ...

TParams = TypeVar("TParams", bound=SkillParams)
"""The generic type variable for the parameters of a skill."""
SkillParamsSpec = SpaceSpecification[TParams]
"""The specification of a skill parameters space."""

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
type Array1D = Float[np.ndarray, "n"]
"""Represents a 1D array of floats ndarray[(n,), float]."""
type BatchedArray1D = Float[np.ndarray, "b n"]
"""Represents a batched 1D array of floats ndarray[(b, n), float]."""
type ArrayEmpty = Float[np.ndarray, "0"]
"""Represents an empty 1D array of floats ndarray[(0,), float]."""
type BatchedArrayEmpty = Float[np.ndarray, "b 0"]
"""Represents a batched empty 1D array of floats ndarray[(b, 0), float]."""
type Array3 = Float[np.ndarray, "3"]
"""Represents a 3D position or orientation in Cartesian space as a 1x3 array."""
type BatchedArray3 = Float[np.ndarray, "b 3"]
"""Represents a batched 3D position or orientation in Cartesian space as a (b, 3) array."""
type Array6 = Float[np.ndarray, "6"]
"""Represents a 6D position and orientation in Cartesian space as a 1x6 array."""
type BatchedArray6 = Float[np.ndarray, "b 6"]
"""Represents a batched 6D position and orientation in Cartesian space as a (b, 6) array."""
type Array7 = Float[np.ndarray, "7"]
"""Represents a 7D position and orientation in Cartesian space with a weight as a 1x7 array."""
type BatchedArray7 = Float[np.ndarray, "b 7"]
"""Represents a batched 7D position and orientation in Cartesian space with a weight as a (b, 7) array."""
type ParamDC = NamedTuple("ParamDC", [("discrete", Int[np.ndarray, "m"]), ("continuous", Float[np.ndarray, "n"])])
"""Represents a skill parameter set with m discrete parameters and n continuous parameters."""
type BatchedParamDC = NamedTuple("BatchedParamDC", [("discrete", Int[np.ndarray, "b m"]), ("continuous", Float[np.ndarray, "b n"])])
"""Represents a batched skill parameter set with m discrete parameters and n continuous parameters."""
type Tensor1D = Float[torch.Tensor, "n"]
"""Represents a 1D array of floats torch.Tensor[(n,), float]."""
type BatchedTensor1D = Float[torch.Tensor, "b n"]
"""Represents a batched 1D array of floats torch.Tensor[(b, n), float]."""
type TensorEmpty = Float[torch.Tensor, "0"]
"""Represents an empty 1D array of floats torch.Tensor[(0,), float]."""
type BatchedTensorEmpty = Float[torch.Tensor, "b 0"]
"""Represents a batched empty 1D array of floats torch.Tensor[(b, 0), float]."""
type Tensor3 = Float[torch.Tensor, "3"]
"""Represents a 3D position or orientation in Cartesian space as a 1x3 tensor."""
type BatchedTensor3 = Float[torch.Tensor, "b 3"]
"""Represents a batched 3D position or orientation in Cartesian space as a (b, 3) tensor."""
type Tensor6 = Float[torch.Tensor, "6"]
"""Represents a 6D position and orientation in Cartesian space as a 1x6 tensor."""
type BatchedTensor6 = Float[torch.Tensor, "b 6"]
"""Represents a batched 6D position and orientation in Cartesian space as a (b, 6) tensor."""
type Tensor7 = Float[torch.Tensor, "7"]
"""Represents a 7D position and orientation in Cartesian space with a weight as a 1x7 tensor."""
type BatchedTensor7 = Float[torch.Tensor, "b 7"]
"""Represents a batched 7D position and orientation in Cartesian space with a weight as a (b, 7) tensor."""
type ParamDC_Torch = NamedTuple("ParamDC_Torch", [("discrete", Int[torch.Tensor, "m"]), ("continuous", Float[torch.Tensor, "n"])])
"""Represents a skill parameter set with m discrete parameters and n continuous parameters as a PyTorch tensor."""
type BatchedParamDC_Torch = NamedTuple("BatchedParamDC_Torch", [("discrete", Int[torch.Tensor, "b m"]), ("continuous", Float[torch.Tensor, "b n"])])
"""Represents a batched skill parameter set with m discrete parameters and n continuous parameters as a PyTorch tensor."""

class CommonSpecs:
    # Predefined static types for common observation spaces
    ArrayEmpty: ClassVar[SpaceSpecification[ArrayEmpty]] = SpaceSpecification[ArrayEmpty](
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