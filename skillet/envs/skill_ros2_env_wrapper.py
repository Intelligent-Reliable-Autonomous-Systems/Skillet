"""skill_isaac_env_wrapper.py.

A wrapper around IsaacLab Gym compatible with skills

Written by Will Solow and Jeff Jewett, 2026
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeVar

import gymnasium as gym
import torch
from jaxtyping import Bool, Float

from skillet.core.skill_controller import SkillController
from skillet.envs.ros2_env_wrapper import ROS2EnvWrapper

if TYPE_CHECKING:
    from skillet.envs.ros2.ros2_rl_env import ROS2RLEnv

TBatchedObsTorch = TypeVar(
    "TBatchedObsTorch", bound=Float[torch.Tensor, "b ..."] | Mapping[str, Float[torch.Tensor, "b ..."]]
)
"""A generic type of the batched observation tensor returned by the environment.

Can be a batched observation tensor or a dictionary of batched observation tensors.

torch.Tensor[(b, ...), float] | Mapping[str, torch.Tensor[(b, ...), float]]"""
TBatchedActionTorch = TypeVar("TBatchedActionTorch", bound=Float[torch.Tensor, "b n"])
"""A generic type of the batched action tensor expected by the environment.

torch.Tensor[(b, n), float]
"""


class SkillROS2EnvWrapper(
    ROS2EnvWrapper[TBatchedObsTorch, TBatchedActionTorch],
):
    """Wrapper for IsaacLab Environments.

    This assumes that the environment is a ROS2RLEnv.
    """

    def __init__(self, env: "ROS2RLEnv") -> None:
        """Initialize the environment.

        Args:
            env: IsaacLab Gymnasium environment

        Returns:
            None

        """
        super().__init__(env)
        if hasattr(env.unwrapped.cfg, "skills"):
            assert (
                env.unwrapped.cfg.skills is not None
            ), "`env.cfg.skills` must not be None. Configure to list of skills."
        else:
            raise ValueError(
                f"Cannot use `SkillIsaacWrapper` when `{type(env.unwrapped.cfg)}` does not contain the `skills` attribute."
            )

        self.sc = SkillController(
            env.unwrapped.cfg.skills,
            num_envs=env.unwrapped.num_envs,
            env=self,
            device=env.unwrapped.device,
        )
        # Update action space based on skill controller
        self.unwrapped.single_action_space = gym.spaces.Box(float("-inf"), float("inf"), shape=(self.sc.action_dim,))
        self.unwrapped.action_space = gym.spaces.Box(
            float("-inf"),
            float("inf"),
            shape=(
                env.unwrapped.num_envs,
                self.sc.action_dim,
            ),
        )

    def step(self, action: TBatchedActionTorch) -> tuple[
        TBatchedObsTorch,
        Float[torch.Tensor, "b"],  # noqa: F821
        Bool[torch.Tensor, "b"],  # noqa: F821
        Bool[torch.Tensor, "b"],  # noqa: F821
        Mapping[str, torch.Tensor],
    ]:
        """Step through the environment.

        Args:
            action: The action tensor of shape (N, num_actions)

        Returns:
            A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

        """
        self.sc.reset(action)  # Reset skills and parse
        _rewards = torch.zeros((self.num_envs,), device=self.device)
        _skill_length = torch.ones((self.num_envs,), device=self.device)
        _dones = self.sc.dones
        i = 0
        while not _dones.all():
            ll_action = self.sc.get_action(self.get_observation())

            obs_dict, reward, term, trunc, info = super().step(ll_action)

            _rewards[~_dones] += reward[~_dones]
            _skill_length += _dones
            _dones = self.sc.dones
            i += 1
        # TODO Process reward according to skill shaping in reward
        self.last_obs = obs_dict

        return obs_dict, reward, term, trunc, info
