"""Define simple joint position policies."""

from typing import Any, Generic

import torch

from skillet.core.policy import BatchedPPolicy, TBAction, TBPolicyObs
from skillet.core.spaces import ActionSpec, ObservationSpec


class GripperPolicy(BatchedPPolicy[TBPolicyObs, torch.Tensor, TBAction], Generic[TBPolicyObs, TBAction]):
    """Base class for Inverse Kinematics End Effector Policy."""

    _params: torch.Tensor

    def __init__(self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction]) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.

        """
        self._obs_spec = obs_spec
        self._action_spec = action_spec

    @property
    def obs_spec(self) -> ObservationSpec[TBPolicyObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:  # noqa: D102
        return self._action_spec

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:
        """Get the next joint positions by computing differential inverse kinematics."""
        gripper_lim = obs["gripper_lim"]
        goal_gripper_pos = (params[:, 0] - gripper_lim[:, 0]) / (gripper_lim[:, 1] - gripper_lim[:, 0])
        return torch.cat((obs["joint_pos"][:, :-1], goal_gripper_pos.unsqueeze(1)), dim=1)

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        pass
