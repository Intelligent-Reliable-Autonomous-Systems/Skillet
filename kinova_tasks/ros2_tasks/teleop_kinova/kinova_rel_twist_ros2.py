from typing import Any

import gymnasium as gym
import numpy as np
import torch
from roslibpy import ActionClient, Ros

from skillet.envs.ros2 import (
    wait_for_action_server,
)

from ..kinova.kinova_ros2 import KinovaROS2Env
from .kinova_ik_rel_ros2 import TeleOpKinovaROS2EnvCfg


class KinovaROS2TwistRelEnv(KinovaROS2Env):
    """Relative twist control assuming a 7 DoF action space (delta xyz, delta rpy + gripper).

    Joint positions published to a twist controller
    """

    def __init__(
        self, cfg: TeleOpKinovaROS2EnvCfg, ros: Ros, render_mode: str | None = None, **kwargs: dict[str, Any]
    ) -> None:
        cfg.episode_length_s = 10e12  # Basically make it so no resets
        cfg.decimation = 1.0
        cfg.dt = 1 / 60

        super().__init__(cfg, ros, render_mode=render_mode, **kwargs)

        self.single_action_space = gym.spaces.Box(float("-inf"), float("inf"), shape=(7,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _pre_process_action(self, actions: torch.Tensor) -> np.ndarray:
        """Pre process the robot action.

        This function is responsible preprocessing the robot action (ie checking joint limits, etc).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, num_joints).

        """
        self.actions = actions.squeeze()

        return self.actions.cpu().numpy()

    def _reset_idx(self) -> None:
        """Reset environment based on specified indices to default position."""
        super()._reset_idx()

    def _publish_action_to_robot(self, cartesian_pos: np.ndarray, duration: float = 3) -> None:
        """Publish the robot action through the twist controller

        Args:
            joint_pos: NDArray of xyz rpy velocities
            duration: Duration of trajectory

        """
        cartesian_pos = cartesian_pos.tolist()
        twist_cmd = {
            "linear": {"x": cartesian_pos[0], "y": cartesian_pos[1], "z": cartesian_pos[2]},
            "angular": {"x": cartesian_pos[3], "y": cartesian_pos[4], "z": cartesian_pos[5]},
        }

        gripper_val = float(cartesian_pos[-1])
        gripper_val = max(0, min(gripper_val, 1)) * 0.8
        # if self.cfg.use_fake_hardware == "true":
        #     gripper_goal = {"command": {"position": gripper_val, "max_effort": 100.0}}
        # else:
        gripper_goal = {"command": {"name": self.cfg.gripper_joint_names, "position": [gripper_val]}}

        self.twist_vel_pub.publish(twist_cmd)

        if gripper_goal != self.curr_gripper_goal:
            _ = self.gripper_client.send_goal(
                gripper_goal, self._gripper_result_cb, self._gripper_feedback_cb, self._gripper_error_cb
            )
            self.curr_gripper_goal = gripper_goal
