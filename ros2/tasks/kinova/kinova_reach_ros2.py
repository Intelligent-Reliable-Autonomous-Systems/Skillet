"""kinova_reach_ros2.py.

Kinova Arm class for ROS2 RL

Written by Will Solow, 2026

"""

import math
from typing import Any

import gymnasium as gym
import numpy as np
from roslibpy import ActionClient, Ros, Topic

from ros2.envs import (
    ROS2RLEnv,
    ROS2RLEnvCfg,
    launch_robot_hardware,
    wait_for_action_server,
    wait_for_rviz,
    wait_for_topic_publish,
    wait_for_topic_subscribe,
    wait_until_ready,
)
from ros2.envs.utils import configclass


@configclass
class KinovaROS2ReachEnvCfg(ROS2RLEnvCfg):
    """The configuration class for Kinova Gen3 Arm."""

    """Robot configuration"""

    # Whether to spin up real robot or not
    use_fake_hardware = "true"

    # IP of the robot
    robot_ip = "www.xxx.yyy.zzz"

    # Visualize
    vision = False

    # Default joint position of robot
    # NOTE: Must be in double format to be compatible with ROS2
    default_joint_positions = [0.0, 0.523599, 0.0, 1.5708, 1.0, 0.785398, 1.0, 0.0]

    """RL environment configuration"""
    num_envs = 1

    device = "cuda"

    dt = 1 / 60

    decimation = 1.0

    episode_length_s = 5.0


class KinovaROS2ReachEnv(ROS2RLEnv):
    """Kinova Gen3 7DoF ROS2 implementation."""

    def __init__(self, cfg: ROS2RLEnvCfg, ros: Ros, render_mode: str | None = None, **kwargs: dict[str, Any]) -> None:
        """Initialize Kinova Arm ROS2."""
        super().__init__(cfg, ros, render_mode, **kwargs)

        self.joint_names = np.asarray(
            [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "joint_7",
                "robotiq_85_left_knuckle_joint",
            ]
        )
        self.joint_state_topic = "/joint_states"
        self.joint_cmd_topic = "/joint_trajectory_controller/joint_trajectory"
        self.gripper_cmd_topic = "/robotiq_gripper_controller/gripper_cmd"
        self.jacobian_topic = "/jacobian"
        self.robot_description_topic = "/robot_info"
        self.body_pose_topic = "/robot_body_pose_w"

        self._ready = {
            "joint_states": False,
            "jacobians": False,
            "robot_info": False,
            "body_pose": False,
        }

        self.default_joint_positions = np.asarray(self.cfg.default_joint_positions)

        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["joints"] = gym.spaces.Box(float("-inf"), float("inf"), shape=(16,))
        self.single_action_space = gym.spaces.Box(float("-inf"), float("inf"), shape=(8,))

        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["joints"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        self._current_joint_positions = np.zeros(shape=len(self.joint_names))
        self._current_joint_velocities = np.zeros(shape=len(self.joint_names))

        # Launch robot hardware in ROS2
        if self.cfg.launch_ros:
            launch_robot_hardware(
                cfg, cfg.ros2_workspace, "gen3_py", "gen3.launch.py", default_joint_positions=cfg.default_joint_positions
            )

        # Wait for topics to be exposed before continuing
        wait_for_topic_publish(self.ros, self.joint_cmd_topic, "trajectory_msgs/msg/JointTrajectory")
        wait_for_action_server(self.ros, self.gripper_cmd_topic, "control_msgs/action/GripperCommand")
        wait_for_topic_subscribe(self.ros, self.joint_state_topic, "sensor_msgs/JointState")
        wait_for_rviz(self.ros)

        wait_for_topic_subscribe(self.ros, self.jacobian_topic, "gen3_cpp/msg/Jacobian")
        wait_for_topic_subscribe(self.ros, self.robot_description_topic, "gen3_cpp/msg/RobotInfo")
        wait_for_topic_subscribe(self.ros, self.body_pose_topic, "gen3_cpp/msg/BodyPose")

        # Subscribe to joint states
        self.joint_states_sub = Topic(self.ros, self.joint_state_topic, "sensor_msgs/JointState")

        def _update_robot_state(msg: dict[str, Any]) -> None:
            """Update the state of the robot by subscribing to robot topics."""
            self._current_joint_positions = msg["position"]
            self._current_joint_velocities = msg["velocity"]
            self._ready["joint_states"] = True

        self.joint_states_sub.subscribe(_update_robot_state)

        # Set up joint trajectory publisher
        self.joint_states_pub = Topic(
            self.ros, "/joint_trajectory_controller/joint_trajectory", "trajectory_msgs/JointTrajectory"
        )

        self.gripper_client = ActionClient(
            self.ros, "/robotiq_gripper_controller/gripper_cmd", "control_msgs/action/ParallelGripperCommand"
        )

        # Subscribe to jacobian topic
        self.jacobian_sub = Topic(self.ros, self.jacobian_topic, "gen3_cpp/msg/Jacobian")

        def _update_jacobians(msg: dict[str, Any]) -> None:
            """Update jacobians the robot by subscribing to jacobian topic."""
            self._current_jacobians = np.asarray(msg["jac_matrix"], dtype=float).reshape(msg["num_links"], msg["rows"], msg["cols"])
            self._ready["jacobians"] = True

        self.jacobian_sub.subscribe(_update_jacobians)

        # Subscribe to robot_description
        self.robot_description_sub = Topic(self.ros, self.robot_description_topic, "gen3_cpp/msg/RobotInfo")

        def _update_robot_links_and_joints(msg: dict[str, Any]) -> None:
            """Update the state of the robot by subscribing to robot topics."""
            self._robot_links = list(msg["links"])
            self._robot_joints = list(msg["joints"])
            self._current_upper_joint_limits = np.asarray(msg["upper_limits"], dtype=float)
            self._current_lower_joint_limits = np.asarray(msg["lower_limits"], dtype=float)
            self._ready["robot_info"] = True

        self.robot_description_sub.subscribe(_update_robot_links_and_joints)

        # Subscribe to robot_description
        self.body_pose_sub = Topic(self.ros, self.body_pose_topic, "gen3_cpp/msg/BodyPose")

        def _update_body_pose(msg: dict[str, Any]) -> None:
            """Update the state of the robot by subscribing to robot topics."""
            self._current_robot_body_pose_w = np.asarray(msg["body_pose_w"], dtype=float).reshape(msg["num_links"], -1)
            self._current_robot_root_pose_w = np.asarray(msg["root_pose_w"], dtype=float)
            self._ready["body_pose"] = True

        self.body_pose_sub.subscribe(_update_body_pose)

        wait_until_ready(self._ready)

    def _pre_process_action(self, actions: np.ndarray) -> np.ndarray:
        """Pre process the robot action.

        This function is responsible preprocessing the robot action (ie checking joint limits, etc).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, num_joints).

        """
        self.actions = actions.squeeze()  # Actions can only be 1 dimensional

        return self.actions

    def _publish_action_to_robot(self, joint_pos: np.ndarray, duration: float = 3) -> None:
        """Publish the robot action.

        Args:
            joint_pos: NDArray of joint positions
            duration: Duration of trajectory

        """
        joint_msg = {
            "joint_names": self.joint_names[:-1].tolist(),
            "points": [
                {
                    "positions": joint_pos[:-1].tolist(),
                    "velocities": [],
                    "accelerations": [],
                    "effort": [],
                    "time_from_start": {
                        "secs": math.floor(duration),
                        "nsecs": int((duration - math.floor(duration)) * 10e9),
                    },
                }
            ],
        }

        # gripper_goal = {"command": {"position": float(joint_pos[-1]), "max_effort": 100.0}}
        gripper_val = float(joint_pos[-1])
        gripper_val = max(0, min(gripper_val, 1)) * 0.8
        gripper_goal = {
            "command": {
                "name": ["robotiq_85_left_knuckle_joint"],
                "position": [gripper_val],
            }
        }

        self.joint_states_pub.publish(joint_msg)

        _ = self.gripper_client.send_goal(
            gripper_goal, self._gripper_result_cb, self._gripper_feedback_cb, self._gripper_error_cb
        )

    def _reset_idx(self) -> None:
        """Reset environment based on specified indices to default position."""
        super()._reset_idx()

        # self._publish_action_to_robot(self.default_joint_positions, duration=5.0)

    def _get_dones(self) -> tuple[bool, bool]:
        """Return dones if longer than max episode length."""
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return False, truncated

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Return the observations from the robot."""
        return {"joints": np.concatenate((self._current_joint_positions, self._current_joint_velocities), axis=0)}

    def _get_rewards(self) -> np.ndarray:
        """Compute the rewards."""
        return np.array([0.0])

    def _gripper_result_cb(self, result: dict[str, Any]) -> None:
        """Gripper action result callback."""
        pass

    def _gripper_feedback_cb(self, feedback: dict[str, Any]) -> None:
        """Gripper action feedback callback."""
        pass

    def _gripper_error_cb(self, err: dict[str, Any]) -> None:
        """Gripper action errpr callback."""
        pass
