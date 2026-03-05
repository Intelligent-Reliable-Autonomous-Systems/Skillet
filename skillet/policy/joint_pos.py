"""Define simple joint position policies."""

from typing import Any, Generic

import torch

from skillet.core.policy import BatchedPPolicy, TBAction, TBPolicyObs
from skillet.core.spaces import ActionSpec, ObservationSpec


class GripperPolicy(BatchedPPolicy[TBPolicyObs, torch.Tensor, TBAction], Generic[TBPolicyObs, TBAction]):
    """Policy for controlling gripper motion."""

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

    @property
    def params_spec(self) -> None: ...

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:
        """Get the next gripper position."""
        return torch.cat((obs["joint_pos"][:, :7], self._goal_gripper_pos), dim=1)

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        self._goal_gripper_pos = params[:, :1]


class JointPosPolicy(BatchedPPolicy[TBPolicyObs, torch.Tensor, TBAction], Generic[TBPolicyObs, TBAction]):
    """Policy for controlling joint motion."""

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

    @property
    def params_spec(self) -> None: ...

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:
        """Get the next gripper position."""
        return self._goal_joint_pos

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        self._goal_joint_pos = params[:, : self.action_spec.space.shape[0]]
