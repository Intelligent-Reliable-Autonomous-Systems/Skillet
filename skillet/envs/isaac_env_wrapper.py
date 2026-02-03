"""isaac_env_wrapper.py.

A wrapper around IsaacLab Gym environments

Written by Will Solow and Jeff Jewett, 2026
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from jaxtyping import Bool, Float

from skillet.core.env import BatchedEnvironment
from skillet.core.math import matrix_from_quat, quat_inv
from skillet.core.spaces import ActionSpec
from skillet.envs.utils import AsGymVectorEnv

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

from skillet.core import ObservationSpec

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


class IsaacEnvWrapper(
    BatchedEnvironment[TBatchedObsTorch, TBatchedActionTorch], Generic[TBatchedObsTorch, TBatchedActionTorch]
):
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
            device=device,
        )
        self._obs_spec_state = ObservationSpec[Mapping[str, Float[torch.Tensor, "b ..."]]](
            space=vector_env.single_observation_space,
            name="state",
            is_torch=True,
            is_batched=True,
            n_envs=-1,
            device=device,
        )
        self._action_spec = ActionSpec[TBatchedActionTorch](
            space=vector_env.single_action_space,
            name="action",
            is_torch=True,
            is_batched=True,
            n_envs=-1,
            device=device,
        )

    @property
    def obs_spec(self):  # noqa: ANN201, D102
        return self._obs_spec_policy

    @property
    def action_spec(self):  # noqa: ANN201, D102
        return self._action_spec

    @property
    def n_envs(self) -> int:  # noqa: D102
        return self._n_envs

    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:  # noqa: D102
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

    def get_observation(self, obs_spec=None):  # noqa: ANN001, ANN201, D102
        if self.last_obs is None:
            raise ValueError("No observation has been received yet. Call reset() first.")
        if obs_spec is None or obs_spec.name == "policy":
            return self.last_obs["policy"]
        if obs_spec.name == "state":
            return self.last_obs
        raise ValueError(f"Observation spec {obs_spec} not supported by environment.")

    def get_state(self) -> TBatchedObsTorch:  # noqa: D102
        return self.get_observation(self._obs_spec_state)

    def step(
        self, action: TBatchedActionTorch
    ) -> tuple[
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
        obs = obs_dict["policy"]
        if isinstance(obs, dict):
            obs = torch.cat(list(obs.values()), dim=1)

        return obs, reward, term, trunc, info

    """
    Helper functions
    """

    def _get_jacobians(self) -> None:
        """Return the jacobians"""
        robot_base_pose_w = self._robot.data.body_pose_w[:, self.cfg.base_link_idx]
        base_rot_matrix = matrix_from_quat(quat_inv(robot_base_pose_w[:, 3:7]))
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self.cfg.ee_jacobi_idx, :, self.cfg.arm_joint_ids]
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
