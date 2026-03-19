"""Define MoveIt policies (ie just parameter pass throughs) for sending commands via moveit.

Written by Will Solow, 2026.
"""

import torch

from skillet.core import SkillParamsSpec
from skillet.core.policy import BatchedPolicy, TBAction, TBPolicyObs
from skillet.core.spaces import ActionSpec, ObservationSpec
from skillet.envs.specs import MOVEIT_TCP_Obs, TCP_QUAT_Action
from skillet.skill.specs import XYZ_QUAT_Params, XYZ_QUAT_Params_Spec


class MoveItTcpQuatPolicy(BatchedPolicy[MOVEIT_TCP_Obs, TCP_QUAT_Action, XYZ_QUAT_Params]):
    """Policy for tcp positions via MoveIt."""

    _params: torch.Tensor

    def __init__(self, obs_spec: ObservationSpec[MOVEIT_TCP_Obs], action_spec: ActionSpec[TCP_QUAT_Action]) -> None:
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
    def params_spec(self) -> SkillParamsSpec[XYZ_QUAT_Params]:
        """The parameter specification for XYZ + Quat target poses."""
        return XYZ_QUAT_Params_Spec

    def get_action(self, obs: TBPolicyObs, params: XYZ_QUAT_Params) -> TBAction:
        """Get the next gripper position."""
        return torch.cat((self._params[:, :7], self.start_gripper_pos), dim=-1)

    def reset(self, obs: TBPolicyObs, params: XYZ_QUAT_Params, env_ids: torch.Tensor = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        self._params = params
        gripper_lim = obs["gripper_lim"]
        self.start_gripper_pos = (obs["gripper"] - gripper_lim[:, :1]) / (gripper_lim[:, 1:] - gripper_lim[:, :1])
