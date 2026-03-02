"""Module for defining and working with space specifications.

There are many different types of spaces that do not play nicely with type hints.
Space Specifications are a way to annotate the type of a space and associate with a Gymnasium space.
Additionally, it can track whether data is batched and whether it is a PyTorch tensor or numpy array.

Observation spaces describe the data observed from the environment.
Skill parameters spaces describe the parameters of a skill.
"""

# gym observation spaces types

# utility to batch an observation spec

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import (
    Any,
    ClassVar,
    Generic,
    NamedTuple,
    Protocol,
    SupportsFloat,
    TypeAlias,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Bool, Float, Int, Shaped
from numpy.typing import NDArray
from tensordict import TensorDict
from typing_extensions import override

# =============================================
# Space specifications
# =============================================
Scalar: TypeAlias = int | float | bool
"""A scalar value."""


@runtime_checkable
class ArrayLike(Protocol):
    """A protocol for array-like objects like numpy arrays and PyTorch tensors.

    This protocol is used to duck type array-like objects like numpy arrays and PyTorch tensors.
    Can be used with jaxtyping like Shaped[ArrayLike, "..."] to type unknown array types.
    Includes common operations like shape, dtype, __len__ and arithmetic operations.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the array-like object."""
        ...

    @property
    def dtype(self) -> np.dtype | torch.dtype:
        """The dtype of the array-like object."""
        ...

    def __len__(self) -> int:
        """Get the length of the array-like object."""
        ...

    def __add__(self, other: object) -> ArrayLike: ...  # noqa: D105
    def __mul__(self, other: object) -> ArrayLike: ...  # noqa: D105

    # def astype(self, dtype: object) -> ArrayLike: ...
    def __iter__(self) -> Iterator[ArrayLike]: ...  # noqa: D105
    def __or__(self, other: ArrayLike) -> ArrayLike: ...  # noqa: D105
    def __and__(self, other: ArrayLike) -> ArrayLike: ...  # noqa: D105
    def __bool__(self) -> bool: ...  # noqa: D105
    def __eq__(self, other: object) -> bool: ...  # noqa: D105
    def __ne__(self, other: object) -> bool: ...  # noqa: D105
    def __lt__(self, other: object) -> bool: ...  # noqa: D105
    def __le__(self, other: object) -> bool: ...  # noqa: D105
    def __gt__(self, other: object) -> bool: ...  # noqa: D105
    def __ge__(self, other: object) -> bool: ...  # noqa: D105
    def __hash__(self) -> int: ...  # noqa: D105
    def __repr__(self) -> str: ...  # noqa: D105
    def __str__(self) -> str: ...  # noqa: D105


NDArrayOrTensor: TypeAlias = NDArray | torch.Tensor

SpaceItem: TypeAlias = Scalar | Shaped[NDArrayOrTensor, "..."]
"""A scalar or list-like value that can be stored in a space."""
SpaceValue: TypeAlias = SpaceItem | Mapping[str, SpaceItem]
"""A scalar or list-like value or dictionary of scalar or list-like values."""
BatchedSpaceItem: TypeAlias = Shaped[NDArrayOrTensor, "b ..."]
"""A batched list-like sequence of scalar or list-like values that can be stored in a space."""
BatchedSpaceValue: TypeAlias = BatchedSpaceItem | Mapping[str, BatchedSpaceItem]
"""A batched scalar or list-like value or dictionary of batched scalar or list-like values."""

TSpace = TypeVar("TSpace", bound=SpaceValue)
"""The generic type variable for a space type."""


@dataclass(frozen=True)
class SpaceSpecification(Generic[TSpace]):
    """Typed descriptor of the space of a data container."""

    space: gym.Space[Any]
    """The Gymnasium space that models the space of the data."""
    name: str
    """The name of the space."""
    is_torch: bool = False
    """Whether the space is a PyTorch tensor or numpy array."""
    is_batched: bool = False
    """Whether the space is batched."""
    n_envs: int | None = None
    """The number of environments in the batched space.

    By default (None), the batch size is inferred from the first dimension of the space.
        Be careful to pass a pre-batched space.
    If -1, the batch size is unknown. Assumes the space is not batched.
    """
    device: torch.device | None = None
    """The device of the space. If None, the space is on the default device."""

    def __post_init__(self) -> None:
        """Post-initialize the space specification."""
        # Infer the batch size from the space if it is not specified.
        if self.is_batched and self.n_envs is None:
            space_shape = self.space.shape
            if space_shape is None:
                if isinstance(self.space, gym.spaces.Dict):
                    space = next(iter(self.space.spaces.values()))
                    space_shape = space.shape
                if space_shape is None:
                    raise ValueError(f"Cannot infer batch size from dict space {self.space} because the shape of the \
                            first subspace is unknown.")
            if len(space_shape) == 0:
                raise ValueError(f"Cannot infer batch size from space {self.space} because the shape is empty. \
                        The space is not batched.")
            if space_shape[0] == -1 or isinstance(space_shape[0], str):
                raise ValueError(f"Cannot infer batch size from space {self.space}. shape[0]={space_shape[0]}")
            object.__setattr__(self, "n_envs", space_shape[0])

        # Set default device if not specified and is torch
        if self.is_torch and self.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            object.__setattr__(self, "device", device)

    def is_parameterized(self) -> bool:
        """Whether the space is defined with variables.

        Many operations require this to be false.
        """
        if isinstance(self.space, ParameterizedSpace):
            return True
        if isinstance(self.space, gym.spaces.Dict):
            return any(isinstance(subspace, ParameterizedSpace) for subspace in self.space.spaces.values())
        return False

    def _ensure_not_parameterized(self) -> None:
        """Ensure the space is not parameterized.

        Many operations require this to be false.
        """
        if self.is_parameterized():
            raise ValueError("Cannot perform operation on a parameterized space.")

    def index(self, value: TSpace, env_ids: Int[NDArrayOrTensor, b] | Bool[NDArrayOrTensor, b]) -> TSpace:
        """Index the space value for the given environment ids.

        Args:
            value: The space value to index.
            env_ids: The environment ids to index the space value for.
                If a sequence of booleans, use boolean mask indexing to select the elements.
                If a sequence of integers, select the elements at the given indices.

        """
        self._ensure_not_parameterized()
        return cast("TSpace", self._index(value, env_ids))

    def _index(self, value: SpaceValue, env_ids: Int[NDArrayOrTensor, b] | Bool[NDArrayOrTensor, b]) -> SpaceValue:
        if not self.is_batched:
            raise ValueError("Cannot index a non-batched space.")
        env_ids = torch.as_tensor(env_ids, device=self.device) if self.is_torch else np.asarray(env_ids)
        if env_ids.ndim == 0:
            raise ValueError("env_ids must be a sequence of booleans or integers.")
        if isinstance(value, Mapping):
            return {key: cast("SpaceItem", self._index(v, env_ids)) for key, v in value.items()}
        value = torch.as_tensor(value, device=self.device) if self.is_torch else np.asarray(value)
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
        self._ensure_not_parameterized()
        if shape is not None:
            dtype = dtype or self.space.dtype
            if shape[0] == -1:
                if self.n_envs == -1:
                    raise ValueError("n_envs not specified. Cannot infer shape with first dimension -1. \
                            Use with_n_envs() to set the batch size.")
                shape = (self.n_envs, *shape[1:])
            if self.is_torch:
                return torch.zeros(shape, dtype=as_torch_dtype(dtype), device=self.device)
            return np.zeros(shape, dtype=dtype)

        if self.is_batched and self.n_envs == -1:
            raise ValueError(
                "Cannot create zeros for a variable batch size space. Use with_n_envs() to set the batch size."
            )

        def zeros_for_space(space: gym.Space[Any]) -> SpaceValue:
            if isinstance(space, gym.spaces.Box):
                if self.is_torch:
                    return torch.zeros(
                        space.shape,
                        dtype=as_torch_dtype(space.dtype),
                        device=self.device,
                    )
                return np.zeros(space.shape, dtype=space.dtype)
            if isinstance(space, gym.spaces.Dict):
                # Recursively fill dict space
                return {key: cast("SpaceItem", zeros_for_space(subspace)) for key, subspace in space.spaces.items()}
            if isinstance(space, gym.spaces.Discrete):
                if self.is_torch:
                    return torch.zeros((), dtype=torch.long, device=self.device)
                return np.zeros((), dtype=int)
            if isinstance(space, gym.spaces.MultiDiscrete):
                if self.is_torch:
                    return torch.zeros(space.nvec.shape, dtype=torch.long, device=self.device)
                return np.zeros(space.nvec.shape, dtype=int)
            if isinstance(space, gym.spaces.MultiBinary):
                if self.is_torch:
                    return torch.zeros(space.n, dtype=torch.long, device=self.device)
                return np.zeros(space.n, dtype=int)
            raise NotImplementedError(f"zeros() not implemented for space type: {type(space)}")

        return zeros_for_space(self.space)

    @overload
    def ones(self) -> TSpace: ...
    @overload
    def ones(self, shape: tuple[int, ...]) -> ArrayLike: ...
    @overload
    def ones(self, shape: tuple[int, ...], dtype: torch.dtype | np.dtype) -> ArrayLike: ...

    def ones(self, shape: Any = None, dtype: Any = None) -> Any:
        """Create a one-filled tensor or array conforming to the Gym space (including handling Dict spaces).

        Optionally specify the shape of the tensor or array.
        """
        self._ensure_not_parameterized()
        if shape is not None:
            dtype = dtype or self.space.dtype
            if shape[0] == -1:
                if self.n_envs == -1:
                    raise ValueError("n_envs not specified. Cannot infer shape with first dimension -1. \
                            Use with_n_envs() to set the batch size.")
                shape = (self.n_envs, *shape[1:])
            if self.is_torch:
                return torch.ones(shape, dtype=as_torch_dtype(dtype), device=self.device)
            return np.ones(shape, dtype=dtype)

        if self.is_batched and self.n_envs == -1:
            raise ValueError(
                "Cannot create ones for a variable batch size space. Use with_n_envs() to set the batch size."
            )

        def ones_for_space(space: gym.Space[Any]) -> SpaceValue:
            if isinstance(space, gym.spaces.Box):
                if self.is_torch:
                    return torch.ones(
                        space.shape,
                        dtype=as_torch_dtype(space.dtype),
                        device=self.device,
                    )
                return np.ones(space.shape, dtype=space.dtype)
            if isinstance(space, gym.spaces.Dict):
                # Recursively fill dict space
                return {key: cast("SpaceItem", ones_for_space(subspace)) for key, subspace in space.spaces.items()}
            if isinstance(space, gym.spaces.Discrete):
                if self.is_torch:
                    return torch.ones((), dtype=torch.long, device=self.device)
                return np.ones((), dtype=int)
            if isinstance(space, gym.spaces.MultiDiscrete):
                if self.is_torch:
                    return torch.ones(space.nvec.shape, dtype=torch.long, device=self.device)
                return np.ones(space.nvec.shape, dtype=int)
            if isinstance(space, gym.spaces.MultiBinary):
                if self.is_torch:
                    return torch.ones(space.n, dtype=torch.long, device=self.device)
                return np.ones(space.n, dtype=int)
            raise NotImplementedError(f"ones() not implemented for space type: {type(space)}")

        return ones_for_space(self.space)

    def sample(self) -> TSpace:
        """Sample a random value from the space."""
        self._ensure_not_parameterized()
        sampled = self.space.sample()
        if self.is_torch:
            return torch.tensor(sampled, device=self.device)
        return sampled

    def cast(self, value: SpaceValue) -> TSpace:
        """Cast a value to the type of the space."""
        self._ensure_not_parameterized()

        def cast_array(v: Any, expected_shape: tuple[int, ...], dtype: Any, key: str = "") -> Any:  # noqa: ANN401
            if self.is_torch:
                arr = torch.as_tensor(v, dtype=as_torch_dtype(dtype), device=self.device)
                if arr.shape != expected_shape:
                    if self.n_envs != -1 and arr.shape == expected_shape[1:]:
                        arr = arr.unsqueeze(0)
                        arr = arr.expand((self.n_envs, *expected_shape[1:]))
                    elif self.n_envs == -1 and arr.shape[1:] == expected_shape:
                        pass  # arr is already batched
                    else:
                        raise ValueError(f"Expected shape {expected_shape} (n_envs={self.n_envs}) but got {arr.shape} \
                            for value {key}.")
                return arr
            # numpy case
            arr = np.asarray(v, dtype=dtype)
            if arr.shape != expected_shape:
                if self.n_envs != -1 and arr.shape == expected_shape[1:]:
                    arr = arr[np.newaxis, ...]
                    arr = np.broadcast_to(arr, (self.n_envs, *expected_shape[1:]))
                elif self.n_envs == -1 and arr.shape[1:] == expected_shape:
                    pass  # arr is already batched
                else:
                    raise ValueError(f"Expected shape {expected_shape} (n_envs={self.n_envs}) but got {arr.shape} \
                        for value {key}.")
            return arr

        if isinstance(self.space, gym.spaces.Dict):
            if not isinstance(value, Mapping):
                raise TypeError(f"Expected a mapping for a Dict space but got {type(value).__name__}.")
            value_dict = {
                key: cast_array(v, self.space.spaces[key].shape, self.space.spaces[key].dtype, key=key)
                for key, v in value.items()
            }
            if self.is_batched and self.is_torch:
                n_envs = next(iter(value_dict.values())).shape[0]  # interpret n_envs from the first value
                return TensorDict(value_dict, batch_size=n_envs)
            return value_dict
        return cast_array(value, self.space.shape, self.space.dtype)

    def replace(self, **kwargs: Any) -> SpaceSpecification[TSpace]:
        """Return a new space specification with the given parameters replaced."""
        return replace(self, **kwargs)

    def with_n_envs(self, n_envs: int) -> SpaceSpecification[TSpace]:
        """Return a new space specification with the given number of environments."""
        if not self.is_batched:
            raise ValueError("Cannot set n_envs for a non-batched space.")
        if self.n_envs != -1 and self.n_envs != n_envs:
            raise ValueError(f"Cannot set n_envs to a different value {n_envs} than the current value {self.n_envs}.")
        if self.n_envs >= 0:
            return self
        if isinstance(self.space, ParameterizedSpace):
            space = self.space.batch(n_envs)
        elif isinstance(self.space, gym.spaces.Dict):
            subspaces = {}
            for key, subspace in self.space.spaces.items():
                if isinstance(subspace, ParameterizedSpace):
                    subspaces[key] = subspace.batch(n_envs)
                else:
                    subspaces[key] = gym.vector.utils.space_utils.batch_space(subspace, n_envs)
            space = gym.spaces.Dict(subspaces)
        else:
            space = gym.vector.utils.space_utils.batch_space(self.space, n_envs) if self.n_envs == -1 else self.space
        return replace(self, space=space, n_envs=n_envs)

    def n_envs_from(self, value: TSpace) -> int:
        """Return the number of environments from the given value."""
        if not self.is_batched:
            raise ValueError("Cannot get n_envs from a non-batched space.")
        if isinstance(value, Mapping):
            return next(iter(value.values())).shape[0]
        return value.shape[0]

    def unbatched(self) -> SpaceSpecification[Any]:
        """Return a new space specification with the batch dimension removed."""
        if not self.is_batched:
            return self
        if self.n_envs == -1:
            return replace(self, is_batched=False, n_envs=None)
        # TODO: implement unbatched for pre-batched spaces (n_envs != -1)
        raise NotImplementedError("unbatched() not available for pre-batched spaces")

    def bind(self, **params: int) -> SpaceSpecification[TSpace]:
        """Bind the space specification to the given parameters."""
        space = self.space
        if isinstance(self.space, ParameterizedSpace):
            space = self.space.bind(**params)
        if isinstance(self.space, gym.spaces.Dict):
            subspaces = {}
            for key, subspace in self.space.spaces.items():
                if isinstance(subspace, ParameterizedSpace):
                    subspaces[key] = subspace.bind(**params)
                else:
                    subspaces[key] = subspace
            space = gym.spaces.Dict(subspaces)
        return replace(self, space=space)

    def type_of(self, other: SpaceSpecification[Any]) -> SpaceSpecification[Any]:
        """Return a new space specification that uses the same data type as the other space specification."""
        return replace(
            self,
            is_torch=other.is_torch,
            is_batched=other.is_batched,
            device=other.device,
        )


# =============================================
# Actions
# =============================================
Action: TypeAlias = SpaceItem
"""Represents an action in the environment."""
BatchedAction: TypeAlias = BatchedSpaceItem
"""A batched action in the environment."""
TAction = TypeVar("TAction", bound=Action | BatchedAction)


# ActionSpec: TypeAlias = SpaceSpecification[TAction]
# class ActionSpec(SpaceSpecification[TAction], Generic[TAction]):
#     """The specification of an action space."""

#     pass

ActionSpec: TypeAlias = SpaceSpecification[TAction]

# =============================================
# Observations
# =============================================
Observation: TypeAlias = SpaceValue
"""A scalar or list-like value or dictionary of scalar or list-like values that can be stored in a space."""
BatchedObservation: TypeAlias = BatchedSpaceValue
"""A batched scalar or list-like value or dictionary of batched scalar or array-like that can be stored in a space."""
State: TypeAlias = Observation
"""The full state of the environment (full observability)."""

TObs = TypeVar("TObs", bound=Observation | BatchedObservation)


# class ObservationSpec(SpaceSpecification[TObs], Generic[TObs]):
#     """The specification of an observation space."""

#     pass

ObservationSpec: TypeAlias = SpaceSpecification[TObs]


# =============================================
# Skill parameters
# =============================================
# Define parameter specifications that provide a structured way to describe the parameter space of a skill.

# Skill parameters are the parameters that are used to control the skill.
# Some skills may involve continuous parameterizations. Some skills may involve discrete parameterizations.

SkillParams: TypeAlias = SpaceValue
"""A scalar or list-like value or dictionary of scalar or list-like values that can be stored in a space."""
BatchedSkillParams: TypeAlias = BatchedSpaceValue
"""A batched scalar or list-like value or dictionary of batched scalar or list-like values."""

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
ArrayEmpty: TypeAlias = Float[np.ndarray | torch.Tensor, "0"]
"""Represents an empty 1D array of floats ndarray[(0,), float]."""
BatchedArrayEmpty: TypeAlias = Float[np.ndarray | torch.Tensor, "b 0"]
"""Represents a batched empty 1D array of floats ndarray[(b, 0), float]."""


class ParamDC(NamedTuple):
    """Represents a skill parameter set with m discrete parameters and n continuous parameters."""

    discrete: Int[np.ndarray, m]
    continuous: Float[np.ndarray, n]


class BatchedParamDC(NamedTuple):
    """Represents a batched skill parameter set with m discrete parameters and n continuous parameters."""

    discrete: Int[np.ndarray, "b m"]
    continuous: Float[np.ndarray, "b n"]


class CommonSpecs:
    """Predefined common space specifications."""

    ArrayEmpty: ClassVar[SpaceSpecification[ArrayEmpty]] = SpaceSpecification[ArrayEmpty](
        space=gym.spaces.Box(low=0.0, high=1.0, shape=(0,), dtype="float32"),
        name="ArrayEmpty",
        is_torch=False,
        is_batched=False,
    )
    BatchedArrayEmpty: ClassVar[SpaceSpecification[BatchedArrayEmpty]] = SpaceSpecification[BatchedArrayEmpty](
        space=gym.spaces.Box(low=0.0, high=1.0, shape=(0,), dtype="float32"),
        name="BatchedArrayEmpty",
        is_torch=False,
        is_batched=True,
        n_envs=-1,
    )


# =============================================
# Utility functions
# =============================================


def as_torch_dtype(dtype: type[int] | type[float] | type[bool] | np.dtype | torch.dtype) -> torch.dtype:
    """Convert a dtype to a PyTorch dtype.

    Args:
        dtype: The dtype to convert to a PyTorch dtype.

    Returns:
        The PyTorch dtype.

    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is int:
        return torch.int64
    if dtype is float:
        return torch.float32
    if dtype is bool:
        return torch.bool
    if isinstance(dtype, np.dtype):
        if dtype == np.uint8:
            return torch.uint8
        if dtype == np.uint16:
            return torch.uint16
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


class ParameterizedSpace(gym.spaces.Space, ABC):
    """A gymnasium space that is parameterized by variables."""

    @property
    def variables(self) -> set[str]:
        """The variables in the parameterized space."""
        return set()

    @abstractmethod
    def bind_partial(self, **params: int) -> gym.spaces.Space:
        """Bind the parameterized space to the given parameters.

        If the parameterized space is not fully bound, return a new parameterized space with the parameters bound.
        Otherwise, return a non-parameterized space.
        """
        raise NotImplementedError(
            f"bind_partial() is not available for this parameterized space {self.__class__.__name__}."
        )

    def bind(self, **params: int) -> gym.spaces.Space:
        """Bind the parameterized space to the given parameters.

        If the parameterized box space is not fully bound, raise a ValueError.
        """
        bound = self.bind_partial(**params)
        if isinstance(bound, ParameterizedSpace):
            if len(bound.variables) > 0:
                raise ValueError(
                    f"Unbound variables {bound.variables} found in space {bound}. All variables must be bound."
                )

            return bound
        return bound

    def batch(self, n_envs: int) -> gym.spaces.Space:
        """Batch the parameterized space.

        If the parameterized space is not fully bound, raise a ValueError.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support batching. Bind space first.")

    @override
    def sample(self, mask: None = None, probability: None = None) -> Any:
        raise ValueError(f"Sample is not available for this parameterized space {self.__class__.__name__}.")

    @override
    def contains(self, x: Any) -> bool:
        raise NotImplementedError(
            f"contains() is not available for this parameterized space {self.__class__.__name__}."
        )

    @override
    def to_jsonable(self, sample_n: Any) -> Any:
        raise ValueError(f"to_jsonable() is not available for this parameterized space {self.__class__.__name__}.")

    @override
    def from_jsonable(self, sample_n: Any) -> Any:
        raise ValueError(f"from_jsonable() is not available for this parameterized space {self.__class__.__name__}.")


class ParameterizedBox(gym.spaces.Box, ParameterizedSpace):
    """A parameterized box space."""

    def __init__(
        self,
        low: SupportsFloat | NDArray[Any] | str,
        high: SupportsFloat | NDArray[Any] | str,
        shape: Sequence[int | str],
        dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32,
        seed: int | np.random.Generator | None = None,
    ):
        self._low = low
        self._high = high
        self._shape_with_params = shape
        self._dtype = dtype
        self._seed = seed

        self._variables = {v for v in shape if isinstance(v, str)}
        if isinstance(low, str):
            self._variables.add(low)
        if isinstance(high, str):
            self._variables.add(high)
        if len(self.variables) == 0:
            raise ValueError("No parameters found in shape or bounds. Use gym.spaces.Box instead.")

    @property
    def variables(self) -> set[str]:
        """The variables in the parameterized space."""
        return self._variables

    @override
    def bind_partial(self, **params: int) -> ParameterizedBox | gym.spaces.Box:
        """Bind the parameterized box space to the given parameters.

        If the parameterized box space is not fully bound, return a new parameterized box space with the parameters bound.
        Otherwise, return a gym.spaces.Box.
        """
        low = params.get(self._low, self._low) if isinstance(self._low, str) else self._low
        high = params.get(self._high, self._high) if isinstance(self._high, str) else self._high
        shape = [
            params[v] if v in params else self._shape_with_params[i] for i, v in enumerate(self._shape_with_params)
        ]
        vars_left = [v for v in shape if isinstance(v, str)]
        if isinstance(low, str):
            vars_left.append(low)
        if isinstance(high, str):
            vars_left.append(high)
        if len(vars_left) > 0:
            return ParameterizedBox(
                low=low,
                high=high,
                shape=shape,
                dtype=self._dtype,
                seed=self._seed,
            )
        low = cast("SupportsFloat | NDArray[Any]", low)
        high = cast("SupportsFloat | NDArray[Any]", high)
        shape = cast("Sequence[int]", shape)
        return gym.spaces.Box(
            low=low,
            high=high,
            shape=shape,
            dtype=self._dtype,
            seed=self._seed,
        )

    @override
    def batch(self, n_envs: int) -> ParameterizedBox | gym.spaces.Box:
        """Batch the parameterized box space.

        Checks for special variables B and n_envs in first shape index
        If found, bind the variable to the batch size.
        Otherwise, create a new parameterized box space with the batch size added to the first dimension.
        """
        first_var = next(iter(self._variables), None)
        if first_var is not None and first_var == "B":
            return self.bind_partial(B=n_envs)
        if first_var is not None and first_var == "n_envs":
            return self.bind_partial(n_envs=n_envs)
        shape = (n_envs, *self._shape_with_params)
        return ParameterizedBox(
            low=self._low,
            high=self._high,
            shape=shape,
            dtype=self._dtype,
            seed=self._seed,
        )

    @property
    def shape(self) -> tuple[int, ...]:  # noqa: D102
        return self._shape

    @override
    def contains(self, x: Any) -> bool:
        if not isinstance(x, np.ndarray):
            gym.logger.warn("Casting input x to numpy array.")
            try:
                x = np.asarray(x, dtype=self.dtype)
            except (ValueError, TypeError):
                return False

        if not np.can_cast(x.dtype, self.dtype):
            return False

        for x_dim, space_dim in zip(x.shape, self.shape, strict=True):
            if isinstance(space_dim, str):
                continue
            if x_dim != space_dim:
                return False
        if np.isscalar(self.low) and not np.all(x == self.low):
            return False

        if np.isscalar(self.high) and not np.all(x == self.high):
            return False

        raise ValueError(f"Cannot establish containment for parameterized box with shape {self.shape}.")


class ParameterizedDiscrete(gym.spaces.Discrete, ParameterizedSpace):
    """A parameterized discrete space.

    Discrete is implemented as an integer Box space with the number of options for each discrete parameter.
    """

    def __init__(self, n: int | str, start: int | str | None = None):
        self._n = n
        self._start = start
        self._variables = set[str]()
        if isinstance(n, str):
            self._variables.add(n)
        if isinstance(start, str):
            self._variables.add(start)
        if len(self._variables) == 0:
            raise ValueError("No parameters found in n or start. Use gym.spaces.Discrete instead.")

    @override
    def bind_partial(self, **params: int) -> ParameterizedDiscrete | gym.spaces.Discrete:
        """Bind the parameterized box space to the given parameters.

        If the parameterized box space is not fully bound, return a new parameterized box space with the parameters bound.
        Otherwise, return a gym.spaces.Box.
        """
        n = params.get(self._n, self._n) if isinstance(self._n, str) else self._n
        start = params.get(self._start, self._start) if isinstance(self._start, str) else self._start
        vars_left = [v for v in self.variables if v not in params]
        if isinstance(n, str):
            vars_left.append(n)
        if isinstance(start, str):
            vars_left.append(start)
        if len(vars_left) > 0:
            return ParameterizedDiscrete(n=n, start=start)
        n = cast("int", n)
        start = cast("int", start) if start is not None else 0
        return gym.spaces.Discrete(n=n, start=start)


if __name__ == "__main__":
    space = ParameterizedDiscrete(n="n_options", start=0)
    bound = space.bind(n_options=10)
    print(bound)
    for i in range(10):
        print(bound.sample())
