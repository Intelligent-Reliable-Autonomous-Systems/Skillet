"""Define twist policies for sending twist commands.

Written by Will Solow, 2026.
"""

import torch

from skillet.core import SkillParamsSpec
from skillet.core.math import (
    compute_pose_error,
    quat_apply,
    quat_inv,
)
from skillet.core.policy import BatchedPolicy, TBAction, TBPolicyObs
from skillet.core.spaces import ActionSpec, ObservationSpec
from skillet.envs.specs import TCP_CART_Action, TCP_Obs
from skillet.skill.specs import XYZ_QUAT_Params, XYZ_QUAT_Params_Spec


class TwistPidPosePolicy(BatchedPolicy[TCP_Obs, TCP_CART_Action, XYZ_QUAT_Params]):
    """Policy for end effector position and orientation using PID control."""

    _params: torch.Tensor

    def __init__(
        self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction], frame: str = "base"
    ) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.
            frame: frame of the world to compute PID commands in. Defaults to base.

        """
        self._obs_spec = obs_spec
        self._action_spec = action_spec
        self.num_envs = obs_spec.n_envs if obs_spec.n_envs > 0 else 1
        self._device = obs_spec.device
        self._frame = frame

        # Max velocities
        self.rot_sensitivity = 20.0
        self.pos_sensitivity = 0.08

        # PID gains
        self.Kp_pos = 1.0
        self.Ki_pos = 0.0
        self.Kd_pos = -0.1
        self.Kp_rot = 1.0
        self.Ki_rot = 0.0
        self.Kd_rot = -0.1

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

    def get_action(self, obs: TBPolicyObs, params: XYZ_QUAT_Params = None) -> TBAction:
        """Get the next gripper position."""
        tcp_pose_b = obs["tcp_pose_b"]
        dt = obs["dt"]

        error_pos, error_rot = compute_pose_error(
            tcp_pose_b[:, 0:3], tcp_pose_b[:, 3:7], self._tcp_quat_des_b[:, 0:3], self._tcp_quat_des_b[:, 3:7]
        )

        # Rotate position and rotation error into the frame the twist controller expects
        error_pos = quat_apply(quat_inv(tcp_pose_b[:, 3:7]), error_pos)
        error_rot = quat_apply(quat_inv(tcp_pose_b[:, 3:7]), error_rot) * 10

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

    def reset(self, obs: TBPolicyObs, params: XYZ_QUAT_Params = None, env_ids: torch.Tensor = None) -> None:
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
