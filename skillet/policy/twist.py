"""Define twist policies for sending twist commands.

Written by Will Solow, 2026.
"""

from typing import Any, Generic

import torch

from skillet.core.math import base_to_tcp_twist, euler_xyz_from_quat, euler_xyz_to_rotvec
from skillet.core.policy import BatchedPolicy, TBAction, TBPolicyObs
from skillet.core.spaces import ActionSpec, ObservationSpec
from skillet.core import SkillParamsSpec
from skillet.skill.specs import (
    XYZ_RPY_Params,
    XYZ_RPY_Params_Spec,
)
from skillet.envs.specs import MOVEIT_TCP_Obs


class TwistTcpFramePolicy(BatchedPolicy[MOVEIT_TCP_Obs, TBAction, XYZ_RPY_Params], Generic[TBPolicyObs, TBAction]):
    """Policy for twist velocities of the end effector in the end effector frame."""

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
    def params_spec(self) -> SkillParamsSpec[XYZ_RPY_Params]:
        """The parameter specification for XYZ + Quat target poses."""
        return XYZ_RPY_Params_Spec

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:
        """Get the next gripper position."""
        return torch.cat((self._params[:, :6], self.start_gripper_pos), dim=-1)

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        self._params = params
        gripper_lim = obs["gripper_lim"]
        self.start_gripper_pos = (obs["gripper"] - gripper_lim[:, :1]) / (gripper_lim[:, 1:] - gripper_lim[:, :1])


class TwistTcpBasePolicy(BatchedPolicy[TBPolicyObs, torch.Tensor, TBAction], Generic[TBPolicyObs, TBAction]):
    """Policy for twist velocities of the end effector in the base (world) frame."""

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
        """Get the next gripper position."""
        return torch.cat(self.twist_cmd, self.start_gripper_pos)

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        self._params = params
        lin_vel_b, ang_vel_b = base_to_tcp_twist(params[:, 0:3], params[:, 3:6], obs["tcp_pose_b"][:, 3:7])
        self.twist_cmd = torch.cat((lin_vel_b, ang_vel_b), dim=-1)
        gripper_lim = obs["gripper_lim"]
        self.start_gripper_pos = (obs["gripper"] - gripper_lim[:, :1]) / (gripper_lim[:, 1:] - gripper_lim[:, :1])


class TwistPIDPosePolicy(BatchedPolicy[TBPolicyObs, torch.Tensor, TBAction], Generic[TBPolicyObs, TBAction]):
    """Policy for end effector position and orientation using PID control."""

    _params: torch.Tensor

    def __init__(self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction]) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.

        """
        self._obs_spec = obs_spec
        self._action_spec = action_spec
        self.num_envs = obs_spec.n_envs
        self._device = obs_spec.device

        # Max velocities
        self.rot_sensitivity = 0.25
        self.pos_sensitivity = 10

        # PID gains
        self.Kp_pos = 1.0
        self.Ki_pos = 0.0
        self.Kd_pos = 0.1
        self.Kp_rot = 1.0
        self.Ki_rot = 0.0
        self.Kd_rot = 0.1

        # PID integrals
        self.integral_pos = torch.zeros(
            (self.num_envs, 3),
            device=self._device,
        )
        self.integral_rot = torch.zeros(
            (self.num_envs, 3),
            device=self._device,
        )

        # Last errors for derivative
        self.last_error_pos = torch.zeros(
            (self.num_envs, 3),
            device=self._device,
        )
        self.last_error_rot = torch.zeros(
            (self.num_envs, 3),
            self._device,
        )

    @property
    def obs_spec(self) -> ObservationSpec[TBPolicyObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:  # noqa: D102
        return self._action_spec

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:
        """Get the next gripper position."""
        tcp_pose_b = obs["tcp_pose_b"]
        dt = obs["dt"]
        r, p, y = euler_xyz_from_quat(tcp_pose_b[:, 3:7])
        robot_xyz_b = torch.cat((tcp_pose_b[:, 0:3].squeeze(), r, p, y), dim=-1).squeeze()

        # Compute errors
        error_pos = robot_xyz_b[:, 0:3] - self._tcp_xyz_des_b[:, 0:3]
        error_rot = robot_xyz_b[:, 3:6] - self._tcp_xyz_des_b[:, 3:6]

        # Update integral terms
        self.integral_pos += error_pos * dt
        self.integral_rot += error_rot * dt

        # Compute derivative terms
        derivative_pos = (error_pos - self.last_error_pos) / dt
        derivative_rot = (error_rot - self.last_error_rot) / dt

        # PID control for translation
        delta_pos = self.Kp_pos * error_pos + self.Ki_pos * self.integral_pos + self.Kd_pos * derivative_pos
        self._delta_pos = torch.clip(delta_pos, -self.pos_sensitivity, self.pos_sensitivity)
        self._delta_pos[:, 1] = -self._delta_pos[:, 1]

        # PID control for rotation (Euler -> rotation vector)
        delta_rot = self.Kp_rot * error_rot + self.Ki_rot * self.integral_rot + self.Kd_rot * derivative_rot
        self._delta_rot = torch.clip(delta_rot, -self.rot_sensitivity, self.rot_sensitivity)
        rot_vec = euler_xyz_to_rotvec(self._delta_rot)

        # Combine translation + rotation for twist command
        command = torch.cat((self._delta_pos, rot_vec), dim=0)

        # Save last errors
        self.last_error_pos = error_pos
        self.last_error_rot = error_rot

        return torch.cat((command, self.start_gripper_pos), dim=-1)

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        self._params = params
        self._tcp_xyz_des_b = params[:, :6]

        # PID integrals
        self.integral_pos = torch.zeros(
            (self.num_envs, 3),
            device=self._device,
        )
        self.integral_rot = torch.zeros(
            (self.num_envs, 3),
            device=self._device,
        )

        # Last errors for derivative
        self.last_error_pos = torch.zeros(
            (self.num_envs, 3),
            device=self._device,
        )
        self.last_error_rot = torch.zeros(
            (self.num_envs, 3),
            self._device,
        )

        gripper_lim = obs["gripper_lim"]
        self.start_gripper_pos = (obs["gripper"] - gripper_lim[:, :1]) / (gripper_lim[:, 1:] - gripper_lim[:, :1])
