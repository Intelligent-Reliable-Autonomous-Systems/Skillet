import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from roslibpy import ActionClient, Ros

from skillet.core.math import (
    convert_quat,
    matrix_from_quat,
    quat_inv,
    subtract_frame_transforms,
)
from skillet.envs.ros2 import (
    ROS2RLEnvCfg,
    wait_for_action_server,
)

from ..kinova.kinova_ros2 import KinovaROS2Env


class KinovaROS2IKRelMoveItEnv(KinovaROS2Env):
    """Relative inverse kinematics control assuming a 7 DoF action space (delta xyz, delta rpy + gripper).

    Joint positions published to MoveIt to resolve collisions.
    """

    def __init__(self, cfg: ROS2RLEnvCfg, ros: Ros, render_mode: str | None = None, **kwargs: dict[str, Any]) -> None:
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
        ee_pose_b = self._get_ee_pose_b()
        self.ik_controller.set_command(actions[:, :6], ee_pos=ee_pose_b[:, 0:3], ee_quat=ee_pose_b[:, 3:7])
        joint_pos = self.ik_controller.compute(
            ee_pos=ee_pose_b[:, 0:3],
            ee_quat=ee_pose_b[:, 3:7],
            jacobian=self._get_jacobians(),
            joint_pos=torch.as_tensor(
                self._current_joint_positions[self.cfg.joint_ids[:-1]], device=self.device
            ).unsqueeze(0),
        )
        self.actions = torch.cat((joint_pos, actions[:, -1:]), dim=-1).squeeze()

        return self.actions.cpu().numpy()

    def _reset_idx(self) -> None:
        """Reset environment based on specified indices to default position."""
        super()._reset_idx()
        self._publish_action_to_robot(self.default_joint_positions, duration=8)
        time.sleep(8)
        self.ik_controller.reset(n_envs=self.num_envs)

    def _pre_process_action(self, actions: torch.Tensor) -> np.ndarray:
        """Pre process the robot action.

        This function is responsible preprocessing the robot action (ie checking joint limits, etc).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, num_joints).

        """
        ee_pose_b = self._get_ee_pose_b()
        self.ik_controller.set_command(actions[:, :6], ee_pos=ee_pose_b[:, 0:3], ee_quat=ee_pose_b[:, 3:7])
        joint_pos = self.ik_controller.compute(
            ee_pos=ee_pose_b[:, 0:3],
            ee_quat=ee_pose_b[:, 3:7],
            jacobian=self._get_jacobians(),
            joint_pos=torch.as_tensor(
                self._current_joint_positions[self.cfg.joint_ids[:-1]], device=self.device
            ).unsqueeze(0),
        )
        self.actions = torch.cat((joint_pos, actions[:, -1:]), dim=-1).squeeze()

        return self.actions.cpu().numpy()

    def _get_jacobians(
        self,
    ) -> torch.Tensor:
        """Return the jacobians.

        Args:
            env_ids: environment ids to compute jacobian
            ee_link: string for the name of the end effector link
            base_link: string for the name of the base link of the robot
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of jacobians of shape (n_envs, num_joints, 3)

        """
        arm_joint_ids = self.cfg.joint_ids[:-1]

        ee_link_idx = self._find_link_idx(self.cfg.ee_link_name)
        base_link_idx = self._find_link_idx(self.cfg.base_link_name)

        robot_base_pose_w = torch.as_tensor(self._robot_body_pose_w, device=self.device, dtype=torch.float32).unsqueeze(
            0
        )[:, base_link_idx]

        # Have to convert quaternion from ROS format (x,y,z,w) to IsaacLab format (w,x,y,z)
        robot_base_pose_w[:, 3:7] = convert_quat(robot_base_pose_w[:, 3:7], to="wxyz")
        base_rot_matrix = matrix_from_quat(quat_inv(robot_base_pose_w[:, 3:7]))

        jacobian = torch.as_tensor(self._jacobians, device=self.device, dtype=torch.float32)
        jacobian = jacobian.unsqueeze(0)[:, ee_link_idx][:, :, arm_joint_ids]

        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        return jacobian.to(torch.float32)

    def _get_ee_pose_b(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the end effector pose of the robot in the robot's base frame.

        Args:
            env_ids: environment ids to compute jacobian
            ee_link: string for the name of the end effector link
            base_link: string for the name of the base link of the robot
            arm_joint_ids: the list of joint ids that correspond to the arm

        Returns:
            The robot EE position in shape (N, 3) relative to the base of the robot
            The robot EE orientation in shape (N, 4) relative to the base of the robot

        """
        ee_link_idx = self._find_link_idx(self.cfg.ee_link_name)
        base_link_idx = self._find_link_idx(self.cfg.base_link_name)

        # Get the pose of the end effector and base in the world frame
        # (B, 7) with (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w)
        robot_ee_pose_w = torch.as_tensor(
            self._robot_body_pose_w[ee_link_idx], device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        robot_base_pose_w = torch.as_tensor(
            self._robot_body_pose_w[base_link_idx], device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        # Have to convert quaternion from ROS format (x,y,z,w) to IsaacLab format (w,x,y,z)
        robot_ee_pose_w[:, 3:7] = convert_quat(robot_ee_pose_w[:, 3:7], to="wxyz")
        robot_base_pose_w[:, 3:7] = convert_quat(robot_base_pose_w[:, 3:7], to="wxyz")

        # Compute the end effector pose in the robot base frame
        robot_ee_pos_b, robot_ee_quat_b = subtract_frame_transforms(
            robot_base_pose_w[:, :3],
            robot_base_pose_w[:, 3:7],
            robot_ee_pose_w[:, :3],
            robot_ee_pose_w[:, 3:7],
        )

        return torch.cat((robot_ee_pos_b, robot_ee_quat_b), dim=1).to(torch.float32)

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
