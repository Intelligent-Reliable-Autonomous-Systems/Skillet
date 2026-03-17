"""Define twist policies for sending twist commands.

Written by Will Solow, 2026.
"""

from typing import Any, Generic

import torch

from skillet.core.math import (
    base_to_tcp_twist,
    euler_xyz_from_quat,
    euler_xyz_to_rotvec,
    quat_mul,
    quat_inv,
    quat_apply,
    axis_angle_from_quat,
    quat_from_euler_xyz,
    convert_quat,
    compute_pose_error,
    quat_apply_yaw,
    yaw_quat,
)
from skillet.core.policy import BatchedPolicy, TBAction, TBPolicyObs
from skillet.core.spaces import ActionSpec, ObservationSpec
from skillet.core import SkillParamsSpec
from skillet.skill.specs import (
    XYZ_QUAT_Params,
    XYZ_QUAT_Params_Spec,
    XYZ_RPY_Params,
    XYZ_RPY_Params_Spec
)
from skillet.envs.specs import MOVEIT_TCP_Obs


class TwistFramePolicy(BatchedPolicy[MOVEIT_TCP_Obs, TBAction, XYZ_RPY_Params], Generic[TBPolicyObs, TBAction]):
    """Policy for twist velocities of the end effector in specified frame"""

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
        self._frame = frame

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
        return torch.cat((self._twist_cmd, self.start_gripper_pos), dim=-1)

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        self._params = params
        if self._frame == "base":
            lin_vel_b, ang_vel_b = base_to_tcp_twist(params[:, 0:3], params[:, 3:6], obs["tcp_pose_b"][:, 3:7])
            self._twist_cmd = torch.cat((lin_vel_b, ang_vel_b), dim=-1)
        else:
            self._twist_cmd = self._params[:, :6]
        gripper_lim = obs["gripper_lim"]
        self.start_gripper_pos = (obs["gripper"] - gripper_lim[:, :1]) / (gripper_lim[:, 1:] - gripper_lim[:, :1])


class TwistPIDPosePolicy(BatchedPolicy[TBPolicyObs, torch.Tensor, TBAction], Generic[TBPolicyObs, TBAction]):
    """Policy for end effector position and orientation using PID control."""

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
        self.num_envs = obs_spec.n_envs if obs_spec.n_envs > 0 else 1  # NOTE this won't work for batched envs
        self._device = obs_spec.device
        self._frame = frame

        # Max velocities
        self.rot_sensitivity = 10.0
        self.pos_sensitivity = 0.06

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
            device=self._device,
        )
        self.i = 0

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
        tcp_pose_b = obs["tcp_pose_b"]
        dt = obs["dt"]

        error_pos, error_rot = compute_pose_error(
            tcp_pose_b[:, 0:3], tcp_pose_b[:, 3:7], self._tcp_quat_des_b[:, 0:3], self._tcp_quat_des_b[:,3:7]
        )

        # Required offset rotation about yaw for the twist controller
        twist_rot = torch.tensor([[0.7071, 0.0, 0.0, -0.7071]], device="cuda")
        # Rotate position and rotation error into the frame the twist controller expects
        curr_new = quat_mul(twist_rot, tcp_pose_b[:, 3:7])
        error_pos = quat_apply(quat_inv(curr_new), error_pos)
        error_rot = quat_apply(quat_inv(curr_new), error_rot) * 10  # Scale rotation error to make it move faster

        print(f"########")
        print(tcp_pose_b.squeeze()[0:3])
        print(self._tcp_quat_des_b.squeeze()[0:3])

        self.integral_pos += error_pos * dt
        self.integral_rot += error_rot * dt

        # Compute derivative terms
        derivative_pos = (error_pos - self.last_error_pos) * dt
        derivative_rot = (error_rot - self.last_error_rot) * dt

        # PID control for translation
        delta_pos = self.Kp_pos * error_pos + self.Ki_pos * self.integral_pos + self.Kd_pos * derivative_pos
        self._delta_pos = torch.clip(delta_pos, -self.pos_sensitivity, self.pos_sensitivity)
        delta_rot = self.Kp_rot * error_rot + self.Ki_rot * self.integral_rot + self.Kd_rot * derivative_rot
        self._delta_rot = torch.clip(delta_rot, -self.rot_sensitivity, self.rot_sensitivity)

        # Combine translation + rotation for twist command
        command = torch.cat((self._delta_pos, self._delta_rot), dim=1)

        # Save last errors
        self.last_error_pos = error_pos
        self.last_error_rot = error_rot

        return torch.cat((command, self.start_gripper_pos), dim=-1)

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        self._params = params
        self._tcp_quat_des_b = params[:, :7]

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
            device=self._device,
        )

        gripper_lim = obs["gripper_lim"]
        self.start_gripper_pos = (obs["gripper"] - gripper_lim[:, :1]) / (gripper_lim[:, 1:] - gripper_lim[:, :1])
