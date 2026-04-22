"""Class for a PID controller."""

import torch


class PidController:
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
        """Compute the angular error."""
        error = target - current
        return (error + torch.pi) % (2 * torch.pi) - torch.pi
