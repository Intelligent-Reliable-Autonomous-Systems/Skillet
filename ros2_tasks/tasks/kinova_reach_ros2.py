"""kinova_reach_ros2.py.

Kinova Arm class for ROS2 RL

Written by Will Solow, 2026

"""

import numpy as np

from ros2_tasks.envs import ROS2RLEnv, ROS2RLEnvCfg
from ros2_tasks.ros2_nodes import JointMananger


class KinovaROS2ReachEnv(ROS2RLEnv):
    """Kinova Gen3 7DoF ROS2 implementation."""

    def __init__(self, cfg: ROS2RLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize Kinova Arm ROS2."""
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_manager = JointMananger

    def _publish_action_to_robot(self, action: np.ndarray) -> None:
        """Publish the robot action.

        Args:
            action: NDArray of joint positions

        """

        self.joint_manager.send_robot_cmd(action)
