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


class KinovaROS2IKRelMoveItEnv(KinovaROS2Env):
    """Relative inverse kinematics control assuming a 7 DoF action space (delta xyz, delta rpy + gripper).

    Joint positions published to MoveIt to resolve collisions.
    """

    def __init__(
        self, cfg: TeleOpKinovaROS2EnvCfg, ros: Ros, render_mode: str | None = None, **kwargs: dict[str, Any]
    ) -> None:
        cfg.episode_length_s = 10e12  # Basically make it so no resets
        cfg.decimation = 60.0
        cfg.dt = 1 / 60

        super().__init__(cfg, ros, render_mode=render_mode, **kwargs)

        self.single_action_space = gym.spaces.Box(float("-inf"), float("inf"), shape=(7,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        self.moveit_cmd_topic = "/move_action"
        self.moveit_cmd_topic_type = "moveit_msgs/action/MoveGroup"

        wait_for_action_server(self.ros, self.moveit_cmd_topic, self.moveit_cmd_topic_type)

        self.moveit_client = ActionClient(self.ros, self.moveit_cmd_topic, self.moveit_cmd_topic_type)

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

    def _publish_action_to_robot(self, joint_pos: np.ndarray, duration: float = 3) -> None:
        """Publish the robot action through the moveit configuration.

        Args:
            joint_pos: NDArray of joint positions
            duration: Duration of trajectory

        """
        moveit_goal = {
            "request": {
                "group_name": "manipulator",
                "goal_constraints": [
                    {
                        "joint_constraints": [
                            {
                                "joint_name": j,
                                "position": float(joint_pos[i]),
                                "tolerance_above": 0.01,
                                "tolerance_below": 0.01,
                                "weight": 1.0,
                            }
                            for i, j in enumerate(self.joint_names[:-1])
                        ]
                    }
                ],
                "num_planning_attempts": 10,
                "allowed_planning_time": 5.0,
                "max_velocity_scaling_factor": 0.1,
                "max_acceleration_scaling_factor": 0.1,
            }
        }

        gripper_val = float(joint_pos[-1])
        gripper_val = max(0, min(gripper_val, 1)) * 0.8
        if self.cfg.use_fake_hardware == "true":
            gripper_goal = {"command": {"position": gripper_val, "max_effort": 100.0}}
        else:
            gripper_goal = {"command": {"name": [self.cfg.gripper_joint_name], "position": [gripper_val]}}

        _ = self.moveit_client.send_goal(
            moveit_goal, self._moveit_result_cb, self._moveit_feedback_cb, self._moveit_error_cb
        )

        _ = self.gripper_client.send_goal(
            gripper_goal, self._gripper_result_cb, self._gripper_feedback_cb, self._gripper_error_cb
        )

    def _moveit_result_cb(self, result: dict[str, Any]) -> None:
        """MoveIt action result callback."""
        status = result.get("status")
        message = result.get("message", "")
        if status.name == "SUCCEEDED":
            # print(f"[INFO] MoveIt succeeded: {message}")
            self.moveit_ok = True

        elif status.name == "ABORTED":
            print(f"[INFO] MoveIt aborted: {message}")
            self.moveit_ok = False

        elif status.name == "CANCELED":
            print(f"[INFO] MoveIt canceled: {message}")
            self.moveit_ok = False
        else:
            print(f"[INFO] Unknown MoveIt result: {result}")
            self.moveit_ok = False
        pass

    def _moveit_feedback_cb(self, feedback: dict[str, Any]) -> None:
        """MoveIt action feedback callback."""
        stalled = feedback.get("stalled", False)
        if stalled:
            print("[INFO] MoveIt stalled before reaching goal")

        pass

    def _moveit_error_cb(self, err: dict[str, Any]) -> None:
        """MoveIt action error callback."""
        code = err.get("code")
        message = err.get("message", "Unknown error")
        details = err.get("details")

        print(f"[INFO] MoveIt error! code={code}, message={message}, details={details}")

        # Set internal state so higher-level logic can react
        self.moveit_ok = False
        self.last_gripper_error = err
        pass
