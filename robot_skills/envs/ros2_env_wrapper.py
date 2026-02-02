"""isaac_env_wrapper.py.

A wrapper around IsaacLab Gym environments

Written by Will Solow and Jeff Jewett, 2026
"""

from typing import TYPE_CHECKING, Any, Generic, Mapping, TypeVar, overload

from jaxtyping import Bool, Float
import gymnasium as gym
import torch

from robot_skills.core.env import BatchedEnvironment
from robot_skills.core.spaces import ActionSpec
from robot_skills.env.utils import AsGymVectorEnv
if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

from robot_skills.core import Environment, TObs, TAction, ObservationSpec

TBatchedObsTorch = TypeVar("TBatchedObsTorch", bound=Float[torch.Tensor, "b ..."] | Mapping[str, Float[torch.Tensor, "b ..."]])
"""A generic type of the batched observation tensor returned by the environment.

Can be a batched observation tensor or a dictionary of batched observation tensors.

torch.Tensor[(b, ...), float] | Mapping[str, torch.Tensor[(b, ...), float]]"""
TBatchedActionTorch = TypeVar("TBatchedActionTorch", bound=Float[torch.Tensor, "b n"])
"""A generic type of the batched action tensor expected by the environment.

torch.Tensor[(b, n), float]
"""

class ROS2EnvWrapper(BatchedEnvironment[TBatchedObsTorch, TBatchedActionTorch], Generic[TBatchedObsTorch, TBatchedActionTorch]):
    """Wrapper for ROS2 Environments.

    This assumes that the environment is either a gym.Env and interfaces directly with ROS2.
    """

    def __init__(self, env: gym.Env) -> None:
        """Initialize the environment.

        Args:
            env: Gymnasium environment that interfaces with ROS2

        Returns:
            None

        """
        # self._isaac_env = env
        # self._n_envs = env.unwrapped.cfg.scene.num_envs
        # device = env.unwrapped.device
        # vector_env = AsGymVectorEnv(env, num_envs=self._n_envs)
        # super().__init__(vector_env)
        # self._obs_spec_policy = ObservationSpec[Float[torch.Tensor, "b ..."]](
        #     space=vector_env.single_observation_space["policy"],
        #     name="policy",
        #     is_torch=True,
        #     is_batched=True,
        #     n_envs=-1,
        #     device=device
        # )
        # self._obs_spec_state = ObservationSpec[Mapping[str, Float[torch.Tensor, "b ..."]]](
        #     space=vector_env.single_observation_space,
        #     name="state",
        #     is_torch=True,
        #     is_batched=True,
        #     n_envs=-1,
        #     device=device
        # )
        # self._action_spec = ActionSpec[TBatchedActionTorch](
        #     space=vector_env.single_action_space,
        #     name="action",
        #     is_torch=True,
        #     is_batched=True,
        #     n_envs=-1,
        #     device=device
        # )

    @property
    def obs_spec(self):
        return self._obs_spec_policy

    @property
    def action_spec(self):
        return self._action_spec

    @property
    def n_envs(self) -> int:
        return self._n_envs

    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:
        return obs_spec.name in ["policy", "state"]

# class ROS2EnvWrapper(SkillEnvWrapper):
#     """Wrapper for ROS2 Environments.

#     This assumes that the environment is either a gym.Env and interfaces directly with ROS2.
#     """

#     def __init__(self, env: gym.Env) -> None:
#         """Initialize the environment.

#         Args:
#             env: IsaacLab Gymnasium environment

#         Returns:
#             None

#         """
#         super().__init__(env)

#     def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
#         """Step through the environment.

#         Args:
#             action: The action tensor of shape (N, num_actions)

#         Returns:
#             A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

#         """
#         action = action.cpu().numpy()
#         obs, reward, term, trunc, info = self.env.step(action)

#         obs = (
#             torch.cat([torch.as_tensor(obs["positions"]), torch.as_tensor(obs["velocities"])], dim=0)
#             .to(self.device)
#             .unsqueeze(0)
#         )
#         reward = torch.as_tensor(reward, device=self.device).unsqueeze(0)
#         term = torch.as_tensor([term], device=self.device).unsqueeze(0)
#         trunc = torch.as_tensor([trunc], device=self.device).unsqueeze(0)

#         return obs, reward, term, trunc, info

#     def reset(self) -> tuple[torch.Tensor, dict]:
#         """Reset the environment.

#         Args:
#             None

#         Returns:
#             A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

#         """
#         obs, info = self.env.reset()
#         obs = (
#             torch.cat([torch.as_tensor(obs["positions"]), torch.as_tensor(obs["velocities"])], dim=0)
#             .to(self.device)
#             .unsqueeze(0)
#         )

#         return obs, info
