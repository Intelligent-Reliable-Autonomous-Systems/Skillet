"""Define simple joint position policies."""

from typing import Any, Generic

import torch

from skillet.core.policy import BatchedPolicy, TBAction, TBPolicyObs
from skillet.core.skill import JOINT_Params, JOINT_Params_Spec, SkillParamsSpec
from skillet.core.spaces import ActionSpec, ObservationSpec


class GripperPolicy(BatchedPolicy[TBPolicyObs, torch.Tensor, TBAction], Generic[TBPolicyObs, TBAction]):
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


class JointPosPolicy(BatchedPolicy[TBPolicyObs, torch.Tensor, TBAction], Generic[TBPolicyObs, TBAction]):
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


class JointPosPidPosePolicy(BatchedPolicy[TBPolicyObs, torch.Tensor, TBAction], Generic[TBPolicyObs, TBAction]):
    """Policy for joint positions using PID control."""

    _params: torch.Tensor

    def __init__(
        self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction], frame: str = "base"
    ) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.

        """
        self._obs_spec = obs_spec
        self._action_spec = action_spec
        self.num_envs = obs_spec.n_envs if obs_spec.n_envs > 0 else 1
        self._device = obs_spec.device
        self._frame = frame

        # Max velocities
        self.joint_sensitivity = 0.06

        # PID gains
        self.Kp_joints = 1.0
        self.Ki_joints = 0.0
        self.Kd_joints = 0.1

        # PID integrals
        self.integral_joints = self._selected_skill.params_spec.zeros()

        # Last errors for derivative
        self.last_error_joints = self._selected_skill.params_spec.zeros()
        self.i = 0

    @property
    def obs_spec(self) -> ObservationSpec[TBPolicyObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:  # noqa: D102
        return self._action_spec

    @property
    def params_spec(self) -> SkillParamsSpec[JOINT_Params]:
        """The parameter specification for joint parameters."""
        return JOINT_Params_Spec

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:
        """Get the next joint position."""
        joint_pos = obs["joint_pos"]
        dt = obs["dt"]

        error_joints = joint_pos - self._joint_pos_des

        self.integral_joints += error_joints * dt

        # Compute derivative terms
        derivative_joints = (error_joints - self.last_error_joints) * dt

        # PID control
        delta_joints = (
            self.Kp_joints * error_joints + self.Ki_joints * self.integral_joints + self.Kd_joints * derivative_joints
        )
        self._delta_joints = torch.clip(delta_joints, -self.joint_sensitivity, self.joint_sensitivity)

        # Save last errors
        self.last_error_joints = error_joints
        return torch.cat((self._delta_joints, self.start_gripper_pos), dim=-1)

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        self._params = params
        self._joint_pos_des = params

        # PID integrals
        self.integral_joints = self._selected_skill.params_spec.zeros()

        # Last errors for derivative
        self.last_error_joints = self._selected_skill.params_spec.zeros()

        gripper_lim = obs["gripper_lim"]
        self.start_gripper_pos = (obs["gripper"] - gripper_lim[:, :1]) / (gripper_lim[:, 1:] - gripper_lim[:, :1])
