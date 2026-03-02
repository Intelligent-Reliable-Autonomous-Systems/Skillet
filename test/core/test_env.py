"""Tests for skillet.core.env — environment wrappers and adapters."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch

from skillet.core.env import (
    AsGymVectorEnv,
    BasicBatchedEnvironment,
    BasicEnvironment,
    BatchedEnvironment,
    Environment,
    _EnvironmentBase,
)
from skillet.core.spaces import ActionSpec, ObservationSpec

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

OBS_SHAPE = (4,)
ACTION_SHAPE = ()


def _make_box_env(obs_shape: tuple[int, ...] = OBS_SHAPE, act_shape: tuple[int, ...] = ACTION_SHAPE) -> gym.Env:
    """Create a simple gym.Env with Box obs and action spaces."""
    env = gym.make(
        "gymnasium.envs.classic_control.cartpole:CartPole-v1",
    )
    assert env.observation_space.shape == obs_shape
    assert env.action_space.shape == act_shape
    return env


def _make_vector_env(num_envs: int = 3) -> gym.vector.VectorEnv:
    """Create a simple vectorised CartPole environment."""
    vec_env = gym.make_vec("CartPole-v1", num_envs=num_envs)
    assert vec_env.num_envs == num_envs
    assert vec_env.single_observation_space.shape == OBS_SHAPE
    assert vec_env.single_action_space.shape == ACTION_SHAPE
    assert vec_env.observation_space.shape == (num_envs, *OBS_SHAPE)
    assert vec_env.action_space.shape == (num_envs, *ACTION_SHAPE)
    return vec_env


@pytest.fixture
def box_env() -> gym.Env:
    """Create a simple gym.Env with Box obs and action spaces."""
    env = _make_box_env()
    yield env
    env.close()


@pytest.fixture
def basic_env(box_env: gym.Env) -> BasicEnvironment:
    """Create a BasicEnvironment wrapper around a gym.Env."""
    return BasicEnvironment(box_env)


@pytest.fixture
def basic_env_torch(box_env: gym.Env) -> BasicEnvironment:
    """Create a BasicEnvironment wrapper around a gym.Env with PyTorch tensors."""
    return BasicEnvironment(box_env, is_torch=True, device=torch.device("cpu"))


@pytest.fixture
def vec_env() -> gym.vector.VectorEnv:
    """Create a simple vectorised CartPole environment."""
    env = _make_vector_env(num_envs=3)
    yield env
    env.close()


@pytest.fixture
def basic_batched_env(vec_env: gym.vector.VectorEnv) -> BasicBatchedEnvironment:
    """Create a BasicBatchedEnvironment wrapper around a gym.vector.VectorEnv."""
    return BasicBatchedEnvironment(vec_env, device=torch.device("cpu"))


@pytest.fixture
def basic_batched_env_torch(vec_env: gym.vector.VectorEnv) -> BasicBatchedEnvironment:
    """Create a BasicBatchedEnvironment wrapper around a gym.vector.VectorEnv with PyTorch tensors."""
    return BasicBatchedEnvironment(vec_env, is_torch=True, device=torch.device("cpu"))


# ===========================================================================
# _EnvironmentBase — static interface
# ===========================================================================


class TestEnvironmentBaseInterface:
    """Verify the abstract/default behaviour of _EnvironmentBase."""

    def test_is_abstract(self) -> None:
        """_EnvironmentBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            _EnvironmentBase()

    def test_obs_spec_raises_not_implemented(self) -> None:
        """Default obs_spec property raises NotImplementedError."""

        class Stub(_EnvironmentBase):
            def supports_observation_spec(self, obs_spec):
                return True  # noqa: ANN001, ANN202

            def supports_action_spec(self, action_spec):
                return True  # noqa: ANN001, ANN202

            def get_observation(self, obs_spec=None):
                return None  # noqa: ANN001, ANN202

            def reset(self, *, seed=None, options=None):
                return None, {}  # noqa: ANN001, ANN202

            def step(self, action, action_spec=None):
                return None, 0, False, False, {}  # noqa: ANN001, ANN202

        stub = Stub()
        with pytest.raises(NotImplementedError):
            _ = stub.obs_spec

    def test_action_spec_raises_not_implemented(self):
        """Default action_spec property raises NotImplementedError."""

        class Stub(_EnvironmentBase):
            def supports_observation_spec(self, obs_spec):
                return True

            def supports_action_spec(self, action_spec):
                return True

            def get_observation(self, obs_spec=None):
                return None

            def reset(self, *, seed=None, options=None):
                return None, {}

            def step(self, action, action_spec=None):
                return None, 0, False, False, {}

        stub = Stub()
        with pytest.raises(NotImplementedError):
            _ = stub.action_spec

    def test_get_state_raises_not_implemented(self):
        """Default get_state() raises NotImplementedError."""

        class Stub(_EnvironmentBase):
            def supports_observation_spec(self, obs_spec):
                return True

            def supports_action_spec(self, action_spec):
                return True

            def get_observation(self, obs_spec=None):
                return None

            def reset(self, *, seed=None, options=None):
                return None, {}

            def step(self, action, action_spec=None):
                return None, 0, False, False, {}

        stub = Stub()
        with pytest.raises(NotImplementedError):
            stub.get_state()

    def test_abstract_methods_required(self):
        """All required abstract methods must be overridden."""
        required = {"supports_observation_spec", "supports_action_spec", "get_observation", "reset", "step"}
        assert required <= set(_EnvironmentBase.__abstractmethods__)

    def test_environment_is_gym_env(self):
        """Environment should be both _EnvironmentBase and gym.Env."""
        assert issubclass(Environment, _EnvironmentBase)
        assert issubclass(Environment, gym.Env)

    def test_batched_environment_is_vector_env(self):
        """BatchedEnvironment should be both _EnvironmentBase and gym.vector.VectorEnv."""
        assert issubclass(BatchedEnvironment, _EnvironmentBase)
        assert issubclass(BatchedEnvironment, gym.vector.VectorEnv)


# ===========================================================================
# BasicEnvironment
# ===========================================================================


class TestBasicEnvironment:
    """Tests for BasicEnvironment wrapping a gym.Env."""

    def test_obs_spec_properties(self, basic_env: BasicEnvironment, box_env: gym.Env):
        spec = basic_env.obs_spec
        assert isinstance(spec, ObservationSpec)
        assert spec.name == "obs"
        assert spec.is_torch is False
        assert spec.is_batched is False
        assert spec.space == box_env.observation_space

    def test_action_spec_properties(self, basic_env: BasicEnvironment, box_env: gym.Env):
        spec = basic_env.action_spec
        assert isinstance(spec, ActionSpec)
        assert spec.name == "action"
        assert spec.is_torch is False
        assert spec.is_batched is False
        assert spec.space == box_env.action_space

    def test_torch_flag_propagates(self, basic_env_torch: BasicEnvironment):
        assert basic_env_torch.obs_spec.is_torch is True
        assert basic_env_torch.action_spec.is_torch is True

    def test_supports_matching_obs_spec(self, basic_env: BasicEnvironment):
        assert basic_env.supports_observation_spec(basic_env.obs_spec) is True

    def test_rejects_different_obs_spec(self, basic_env: BasicEnvironment):
        other = ObservationSpec(
            space=gym.spaces.Box(0, 1, shape=(99,)),
            name="other",
            is_torch=False,
            is_batched=False,
        )
        assert basic_env.supports_observation_spec(other) is False

    def test_supports_matching_action_spec(self, basic_env: BasicEnvironment):
        assert basic_env.supports_action_spec(basic_env.action_spec) is True

    def test_rejects_different_action_spec(self, basic_env: BasicEnvironment):
        other = ActionSpec(
            space=gym.spaces.Box(0, 1, shape=(99,)),
            name="other",
            is_torch=False,
            is_batched=False,
        )
        assert basic_env.supports_action_spec(other) is False

    # -- reset / step / observation flow --

    def test_reset_returns_obs_and_info(self, basic_env: BasicEnvironment):
        obs, info = basic_env.reset()
        assert obs is not None
        assert isinstance(info, dict)

    def test_reset_stores_last_obs(self, basic_env: BasicEnvironment):
        obs, _ = basic_env.reset()
        assert basic_env.last_obs is not None
        np.testing.assert_array_equal(basic_env.last_obs, obs)

    def test_step_returns_five_tuple(self, basic_env: BasicEnvironment):
        basic_env.reset()
        action = basic_env.action_spec.space.sample()
        result = basic_env.step(action)
        assert len(result) == 5
        obs, reward, term, trunc, info = result
        assert obs is not None
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(term, (bool, np.bool_))
        assert isinstance(trunc, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_step_updates_last_obs(self, basic_env: BasicEnvironment):
        basic_env.reset()
        action = basic_env.action_spec.space.sample()
        obs, *_ = basic_env.step(action)
        np.testing.assert_array_equal(basic_env.last_obs, obs)

    def test_step_rejects_unsupported_action_spec(self, basic_env: BasicEnvironment):
        basic_env.reset()
        bad_spec = ActionSpec(
            space=gym.spaces.Box(0, 1, shape=(99,)),
            name="wrong",
            is_torch=False,
            is_batched=False,
        )
        action = basic_env.action_spec.space.sample()
        with pytest.raises(ValueError, match="not supported"):
            basic_env.step(action, action_spec=bad_spec)

    def test_step_accepts_matching_action_spec(self, basic_env: BasicEnvironment):
        basic_env.reset()
        action = basic_env.action_spec.space.sample()
        result = basic_env.step(action, action_spec=basic_env.action_spec)
        assert len(result) == 5

    # -- get_observation --

    def test_get_observation_before_reset_raises(self, basic_env: BasicEnvironment):
        with pytest.raises(ValueError, match="reset"):
            basic_env.get_observation()

    def test_get_observation_returns_last_obs(self, basic_env: BasicEnvironment):
        obs, _ = basic_env.reset()
        np.testing.assert_array_equal(basic_env.get_observation(), obs)

    def test_get_observation_with_matching_spec(self, basic_env: BasicEnvironment):
        obs, _ = basic_env.reset()
        result = basic_env.get_observation(basic_env.obs_spec)
        np.testing.assert_array_equal(result, obs)

    def test_get_observation_rejects_unsupported_spec(self, basic_env: BasicEnvironment):
        basic_env.reset()
        other = ObservationSpec(
            space=gym.spaces.Box(0, 1, shape=(99,)),
            name="nope",
            is_torch=False,
            is_batched=False,
        )
        with pytest.raises(ValueError, match="not supported"):
            basic_env.get_observation(other)

    # -- get_state --

    def test_get_state_delegates_to_get_observation(self, basic_env: BasicEnvironment):
        obs, _ = basic_env.reset()
        np.testing.assert_array_equal(basic_env.get_state(), obs)

    # -- isinstance checks --

    def test_is_environment_subclass(self, basic_env: BasicEnvironment):
        assert isinstance(basic_env, Environment)
        assert isinstance(basic_env, _EnvironmentBase)
        assert isinstance(basic_env, gym.Env)


# ===========================================================================
# BasicBatchedEnvironment
# ===========================================================================


class TestBasicBatchedEnvironment:
    """Tests for BasicBatchedEnvironment wrapping a gym.vector.VectorEnv."""

    def test_obs_spec_properties(self, basic_batched_env: BasicBatchedEnvironment):
        spec = basic_batched_env.obs_spec
        assert isinstance(spec, ObservationSpec)
        assert spec.name == "obs"
        assert spec.is_batched is True
        assert spec.n_envs == -1  # variable batch

    def test_action_spec_properties(self, basic_batched_env: BasicBatchedEnvironment):
        spec = basic_batched_env.action_spec
        assert isinstance(spec, ActionSpec)
        assert spec.name == "action"
        assert spec.is_batched is True
        assert spec.n_envs == -1

    def test_torch_flag_propagates(self, basic_batched_env_torch: BasicBatchedEnvironment):
        assert basic_batched_env_torch.obs_spec.is_torch is True
        assert basic_batched_env_torch.action_spec.is_torch is True

    def test_uses_single_spaces(self, basic_batched_env: BasicBatchedEnvironment, vec_env: gym.vector.VectorEnv):
        assert basic_batched_env.obs_spec.space == vec_env.single_observation_space
        assert basic_batched_env.action_spec.space == vec_env.single_action_space

    def test_supports_matching_obs_spec(self, basic_batched_env: BasicBatchedEnvironment):
        assert basic_batched_env.supports_observation_spec(basic_batched_env.obs_spec)

    def test_rejects_different_obs_spec(self, basic_batched_env: BasicBatchedEnvironment):
        other = ObservationSpec(
            space=gym.spaces.Box(0, 1, shape=(99,)),
            name="other",
            is_torch=False,
            is_batched=True,
            n_envs=-1,
        )
        assert not basic_batched_env.supports_observation_spec(other)

    def test_supports_matching_action_spec(self, basic_batched_env: BasicBatchedEnvironment):
        assert basic_batched_env.supports_action_spec(basic_batched_env.action_spec)

    def test_rejects_different_action_spec(self, basic_batched_env: BasicBatchedEnvironment):
        other = ActionSpec(
            space=gym.spaces.Box(0, 1, shape=(99,)),
            name="other",
            is_torch=False,
            is_batched=True,
            n_envs=-1,
        )
        assert not basic_batched_env.supports_action_spec(other)

    # -- reset / step --

    def test_reset_returns_obs_and_info(self, basic_batched_env: BasicBatchedEnvironment):
        obs, info = basic_batched_env.reset()
        assert obs is not None
        assert isinstance(info, dict)

    def test_reset_stores_last_obs(self, basic_batched_env: BasicBatchedEnvironment):
        obs, _ = basic_batched_env.reset()
        assert basic_batched_env.last_obs is not None
        np.testing.assert_array_equal(basic_batched_env.last_obs, obs)

    def test_step_returns_five_tuple(self, basic_batched_env: BasicBatchedEnvironment):
        basic_batched_env.reset()
        actions = np.array([basic_batched_env.action_spec.space.sample() for _ in range(3)])
        obs, reward, term, trunc, info = basic_batched_env.step(actions)
        assert obs is not None
        assert reward is not None
        assert term is not None
        assert trunc is not None
        assert isinstance(info, dict)

    def test_step_updates_last_obs(self, basic_batched_env: BasicBatchedEnvironment):
        basic_batched_env.reset()
        actions = np.array([basic_batched_env.action_spec.space.sample() for _ in range(3)])
        obs, *_ = basic_batched_env.step(actions)
        np.testing.assert_array_equal(basic_batched_env.last_obs, obs)

    # -- get_observation --

    def test_get_observation_before_reset_raises(self, basic_batched_env: BasicBatchedEnvironment):
        with pytest.raises(ValueError, match="reset"):
            basic_batched_env.get_observation()

    def test_get_observation_returns_last_obs(self, basic_batched_env: BasicBatchedEnvironment):
        obs, _ = basic_batched_env.reset()
        np.testing.assert_array_equal(basic_batched_env.get_observation(), obs)

    def test_get_observation_with_matching_spec(self, basic_batched_env: BasicBatchedEnvironment):
        obs, _ = basic_batched_env.reset()
        result = basic_batched_env.get_observation(basic_batched_env.obs_spec)
        np.testing.assert_array_equal(result, obs)

    def test_get_observation_rejects_unsupported_spec(self, basic_batched_env: BasicBatchedEnvironment):
        basic_batched_env.reset()
        other = ObservationSpec(
            space=gym.spaces.Box(0, 1, shape=(99,)),
            name="nope",
            is_torch=False,
            is_batched=True,
            n_envs=-1,
        )
        with pytest.raises(ValueError, match="not supported"):
            basic_batched_env.get_observation(other)

    # -- get_state --

    def test_get_state_delegates_to_get_observation(self, basic_batched_env: BasicBatchedEnvironment):
        obs, _ = basic_batched_env.reset()
        np.testing.assert_array_equal(basic_batched_env.get_state(), obs)

    # -- isinstance checks --

    def test_is_batched_environment_subclass(self, basic_batched_env: BasicBatchedEnvironment):
        assert isinstance(basic_batched_env, BatchedEnvironment)
        assert isinstance(basic_batched_env, _EnvironmentBase)
        assert isinstance(basic_batched_env, gym.vector.VectorEnv)
