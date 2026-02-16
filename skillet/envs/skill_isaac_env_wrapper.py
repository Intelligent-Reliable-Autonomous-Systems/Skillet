"""skill_isaac_env_wrapper.py.

A wrapper around IsaacLab Gym compatible with skills

Written by Will Solow and Jeff Jewett, 2026
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeVar

import torch
from jaxtyping import Bool, Float

from skillet.envs.isaac_env_wrapper import IsaacEnvWrapper

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

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


class SkillIsaacEnvWrapper(IsaacEnvWrapper):
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
        super().__init__(env)

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
        obs_dict, reward, term, trunc, info = self.env.step(action)
        self.last_obs = obs_dict

        return obs_dict, reward, term, trunc, info
