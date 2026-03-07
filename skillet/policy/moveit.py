"""Define MoveIt policies (ie just parameter pass throughs) for sending commands via moveit

Written by Will Solow, 2026.
"""

from typing import Any, Generic

import torch

from skillet.core.policy import BatchedPPolicy, TBAction, TBPolicyObs
from skillet.core.spaces import ActionSpec, ObservationSpec
from skillet.skill.specs import XYZ_QUAT_Params, XYZ_QUAT_Params_Spec
from skillet.envs.specs import MOVEIT_TCP_Obs
from skillet.core import SkillParamsSpec


class MoveItTcpQuatPolicy(BatchedPPolicy[MOVEIT_TCP_Obs, TBAction, XYZ_QUAT_Params], Generic[TBPolicyObs, TBAction]):
    """Policy for tcp positions via MoveIt"""

    _params: torch.Tensor

    def __init__(self, obs_spec: MOVEIT_TCP_Obs, action_spec: ActionSpec[TBAction]) -> None:
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

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:
        """Get the next gripper position."""
        return torch.cat((self._params[:, :7], self.start_gripper_pos), dim=-1)

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        self._params = params
        gripper_lim = obs["gripper_lim"]
        self.start_gripper_pos = (obs["gripper"] - gripper_lim[:, :1]) / (gripper_lim[:, 1:] - gripper_lim[:, :1])
