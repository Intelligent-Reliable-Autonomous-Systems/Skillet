
import torch

from skillet.core.math import quat_error_magnitude


class TargetReachManager:

    def __init__(self,
        min_pose_threshold: float,
        quat_threshold: float,
        window_size: int = 120,
        max_pose_threshold: float | None = None,
        stopped_velocity_threshold: float = 0.001
    ):
        if max_pose_threshold is None:
            max_pose_threshold = min_pose_threshold
        self._min_pose_threshold = min_pose_threshold
        self._max_pose_threshold = max_pose_threshold
        self._quat_threshold = quat_threshold
        self._window_size = window_size
        self._stopped_velocity_threshold = stopped_velocity_threshold
        self._head_idx = 0
        self._stopped_steps = None
        self._is_full = False

    def reset(self, pos: torch.Tensor, quat: torch.Tensor) -> None:
        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
            quat = quat.unsqueeze(0)
        num_envs = pos.shape[0]
        self._pos_window = torch.zeros((self._window_size, num_envs, 3), device=pos.device)
        self._quat_window = torch.zeros((self._window_size, num_envs, 4), device=quat.device)
        self._head_idx = -1
        self._stopped_steps = torch.zeros(num_envs, dtype=torch.long, device=pos.device)
        self._is_full = False
        self.add_pose(pos, quat)

    def add_pose(self, pos: torch.Tensor, quat: torch.Tensor) -> None:
        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
            quat = quat.unsqueeze(0)
        self._head_idx = (self._head_idx + 1) % self._window_size
        self._pos_window[self._head_idx] = pos
        self._quat_window[self._head_idx] = quat
        if self._head_idx + 1 == self._window_size:
            self._is_full = True
        stopped = self._is_stopped()
        self._stopped_steps = torch.where(
            stopped,
            self._stopped_steps + 1,
            torch.zeros_like(self._stopped_steps),
        )

    def _is_stopped(self) -> torch.Tensor:
        num_poses = self._window_size if self._is_full else self._head_idx + 1
        if num_poses == 0:
            return torch.zeros(self._stopped_steps.shape[0], dtype=torch.bool, device=self._pos_window.device)
        samples = self._pos_window[:num_poses]
        span = samples.max(dim=0).values - samples.min(dim=0).values
        return torch.linalg.vector_norm(span / num_poses, dim=-1) < self._stopped_velocity_threshold

    def reached_pos(self, target_pos: torch.Tensor) -> torch.Tensor:
        if target_pos.ndim == 1:
            target_pos = target_pos.unsqueeze(0)
        frac = (self._stopped_steps.float() / self._window_size).clamp(max=1.0)
        pos_threshold = self._min_pose_threshold * (1 - frac) + self._max_pose_threshold * frac
        return (
            torch.linalg.vector_norm(self._pos_window[self._head_idx] - target_pos, dim=-1)
            < pos_threshold
        )

    def reached_quat(self, target_quat: torch.Tensor) -> torch.Tensor:
        if target_quat.ndim == 1:
            target_quat = target_quat.unsqueeze(0)
        return quat_error_magnitude(self._quat_window[self._head_idx], target_quat) < self._quat_threshold

    def is_stuck(self) -> torch.Tensor:
        return self._stopped_steps >= self._window_size
