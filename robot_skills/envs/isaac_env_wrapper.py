"""isaac_env_wrapper.py.

A wrapper around IsaacLab Gym environments

Written by Will Solow and Jeff Jewett, 2026
"""

from typing import TYPE_CHECKING, Any, Generic, Mapping, TypeVar, overload

from jaxtyping import Bool, Float
import torch

from robot_skills.core.env import BatchedEnvironment
from robot_skills.core.spaces import ActionSpec
from robot_skills.envs.utils import AsGymVectorEnv
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

class IsaacEnvWrapper(BatchedEnvironment[TBatchedObsTorch, TBatchedActionTorch], Generic[TBatchedObsTorch, TBatchedActionTorch]):
    """Wrapper for IsaacLab Environments.

    This assumes that the environment is either a DirectRLEnv or ManagerBasedRLEnv.
    """

    def __init__(self, env: "ManagerBasedRLEnv | DirectRLEnv") -> None:
        """Initialize the environment.

        Args:
            env: IsaacLab Gymnasium environment

        Returns:
            None

        """
        self._isaac_env = env
        self._n_envs = env.unwrapped.cfg.scene.num_envs
        device = env.unwrapped.device
        vector_env = AsGymVectorEnv(env, num_envs=self._n_envs)
        super().__init__(vector_env)
        self._obs_spec_policy = ObservationSpec[Float[torch.Tensor, "b ..."]](
            space=vector_env.single_observation_space["policy"],
            name="policy",
            is_torch=True,
            is_batched=True,
            n_envs=-1,
            device=device
        )
        self._obs_spec_state = ObservationSpec[Mapping[str, Float[torch.Tensor, "b ..."]]](
            space=vector_env.single_observation_space,
            name="state",
            is_torch=True,
            is_batched=True,
            n_envs=-1,
            device=device
        )
        self._action_spec = ActionSpec[TBatchedActionTorch](
            space=vector_env.single_action_space,
            name="action",
            is_torch=True,
            is_batched=True,
            n_envs=-1,
            device=device
        )

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

    def reset(self) -> tuple[TBatchedObsTorch, dict]:
        """Reset the environment.

        Args:
            None

        Returns:
            A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

        """
        obs_dict, info = self.env.reset()
        self.last_obs = obs_dict
        obs = obs_dict["policy"]
        if isinstance(obs, dict):
            obs = torch.cat(list(obs.values()), dim=1)

        return obs, info
    
    def get_observation(self, obs_spec = None):
        if self.last_obs is None:
            raise ValueError("No observation has been received yet. Call reset() first.")
        if obs_spec.name == "policy" or obs_spec is None:
            return self.last_obs["policy"]
        if obs_spec.name == "state":
            return self.last_obs
        raise ValueError(f"Observation spec {obs_spec} not supported by environment.")

    def get_state(self) -> TObs:
        return self.get_observation(self.STATE_OBS_SPEC)

    def step(self, action: TBatchedActionTorch) -> tuple[TBatchedObsTorch, Float[torch.Tensor, "b"], Bool[torch.Tensor, "b"], \
            Bool[torch.Tensor, "b"], Mapping[str, torch.Tensor]]:
        """Step through the environment.

        Args:
            action: The action tensor of shape (N, num_actions)

        Returns:
            A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

        """
        obs_dict, reward, term, trunc, info = self.env.step(action)
        self.last_obs = obs_dict
        obs = obs_dict["policy"]
        if isinstance(obs, dict):
            obs = torch.cat(list(obs.values()), dim=1)

        return obs, reward, term, trunc, info
