"""Class for a PID controller."""

import torch

from skillet.core.math import compute_pose_error, quat_from_euler_xyz, euler_xyz_from_quat


class PidJointController:
    """PID Controller base class."""

    def __init__(
        self,
        num_envs: int = 1,
        kp: float = 1.0,
        kd: float = 0.1,
        ki: float = 0.0,
        dt: float = 1 / 60,
        max_vel: float = 0.15,  # radians per sec
        device: str = "cuda",
    ) -> None:
        """Initialize the PID controller class."""
        self.num_envs = num_envs
        self.Kp = kp
        self.Kd = kd
        self.Ki = ki
        self.dt = dt
        self.max_vel = max_vel
        self._device = device
        self._desired_position = None

    def get_action(self, position: torch.Tensor) -> torch.Tensor:
        """Get the velocity command from the current and desired position."""
        # error = position - self._desired_position if self._desired_position is not None else position
        error = (
            self._angular_error(self._desired_position, position) if self._desired_position is not None else -position
        )

        # Compute integral terms
        self.integral += error * self.dt

        # Compute derivative terms
        derivative = (error - self.last_error) * self.dt

        # PID control
        delta = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self._delta = torch.clip(delta, -self.max_vel, self.max_vel)

        # Save last errors
        self.last_error = error
        return self._delta

    def reset(self, desired_position: torch.Tensor, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Reset the PID controller."""
        if self._desired_position is None or env_ids is None:
            self.integral = torch.zeros_like(desired_position, device=desired_position.device)
            self.last_error = torch.zeros_like(desired_position, device=desired_position.device)
        else:
            self.integral[env_ids] = torch.zeros_like(desired_position, device=desired_position.device)[env_ids]
            self.last_error[env_ids] = torch.zeros_like(desired_position, device=desired_position.device)[env_ids]

        self._desired_position = desired_position

    def _angular_error(self, target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """Compute the angular error.

        Note that this is not currently aware of joint limits.
        """
        error = target - current
        return (error + torch.pi) % (2 * torch.pi) - torch.pi


class PidTwistController:
    """PID Controller for twist actions class."""

    def __init__(
        self,
        num_envs: int = 1,
        kp: float = 1.0,
        kd: float = 0.1,
        ki: float = 0.0,
        dt: float = 1 / 60,
        max_pos_vel: float = 0.08,  # m/s
        max_ang_vel: float = 20,  # degrees/sec
        device: str = "cuda",
    ) -> None:
        """Initialize the PID controller class."""
        self.num_envs = num_envs
        self.Kp_pos, self.Kp_rot = kp, kp
        self.Kd_pos, self.Kd_rot = kd, kd
        self.Ki_pos, self.Ki_rot = ki, ki
        self.dt = dt
        self.max_pos_vel = max_pos_vel
        self.max_ang_vel = max_ang_vel
        self._device = device
        self._desired_pose_xyz = None
        self._desired_quat = None

    def get_action(self, pose: torch.Tensor) -> torch.Tensor:
        """Get the velocity command from the current and desired pose."""
        if self._desired_pose_xyz is None:
            error_pos = pose[:, 0:3]
            error_rot = torch.stack(euler_xyz_from_quat(pose), dim=-1)
        else:
            error_pos, error_rot = compute_pose_error(
                pose[:, 0:3], pose[:, 3:7], self._desired_pose_xyz[:, 0:3], self._desired_quat
            )

        self.integral_pos += error_pos * self.dt
        self.integral_rot += error_rot * self.dt

        # Compute derivative terms
        derivative_pos = (error_pos - self.last_error_pos) * self.dt
        derivative_rot = (error_rot - self.last_error_rot) * self.dt

        # PID control for translation
        delta_pos = self.Kp_pos * error_pos + self.Ki_pos * self.integral_pos + self.Kd_pos * derivative_pos
        self._delta_pos = torch.clip(delta_pos, -self.max_pos_vel, self.max_pos_vel)
        delta_rot = self.Kp_rot * error_rot + self.Ki_rot * self.integral_rot + self.Kd_rot * derivative_rot
        self._delta_rot = torch.clip(delta_rot, -self.max_ang_vel, self.max_ang_vel)

        # Combine translation + rotation for twist command
        command = torch.cat((self._delta_pos, self._delta_rot), dim=1)

        # Save last errors
        self.last_error_pos = error_pos
        self.last_error_rot = error_rot
        return command

    def reset(self, desired_pose_xyz: torch.Tensor, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Reset the PID controller."""
        num_envs = desired_pose_xyz.shape[0]
        if self._desired_pose_xyz is None or env_ids is None:
            self.integral_pos = torch.zeros((num_envs, 3), device=desired_pose_xyz.device)
            self.integral_rot = torch.zeros((num_envs, 3), device=desired_pose_xyz.device)
            self.last_error_pos = torch.zeros((num_envs, 3), device=desired_pose_xyz.device)
            self.last_error_rot = torch.zeros((num_envs, 3), device=desired_pose_xyz.device)
        else:
            self.integral_pos[env_ids] = torch.zeros((num_envs, 3), device=desired_pose_xyz.device)[env_ids]
            self.integral_rot[env_ids] = torch.zeros((num_envs, 3), device=desired_pose_xyz.device)[env_ids]
            self.last_error_pos[env_ids] = torch.zeros((num_envs, 3), device=desired_pose_xyz.device)[env_ids]
            self.last_error_rot[env_ids] = torch.zeros((num_envs, 3), device=desired_pose_xyz.device)[env_ids]

        self._desired_pose_xyz = desired_pose_xyz
        self._desired_quat = quat_from_euler_xyz(desired_pose_xyz[:, 3], desired_pose_xyz[:, 4], desired_pose_xyz[:, 5])
