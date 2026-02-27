"""kinova_reach_ros2_rl.py.

Kinova Arm class for ROS2 RL

Written by Will Solow, 2026

"""

from typing import Any

import numpy as np
from roslibpy import Ros

from skillet.envs.ros2 import (
    ROS2RLEnvCfg,
)

from .kinova_ros2 import KinovaROS2Env


class KinovaROS2ReachRLEnv(KinovaROS2Env):
    def __init__(self, cfg: ROS2RLEnvCfg, ros: Ros, render_mode: str | None = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(cfg, ros, render_mode=render_mode, **kwargs)

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Return the observations from the robot."""
        return {
            "policy": np.concatenate(
                (
                    self._current_joint_positions,
                    self._current_joint_velocities,
                    np.array([0.5, -0.1, 0.3, 0.0, 1.57, 1.57]),
                    self.prev_action,
                ),
                axis=0,
                dtype=np.float32,
            )
        }
