"""kinova_ros2.py.

Kinova Arm class for ROS2 RL

Written by Will Solow, 2026

"""

import base64
import math
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import torch
from roslibpy import ActionClient, Ros, Service, Topic

from skillet.envs.ros2 import (
    ROS2RLEnv,
    ROS2RLEnvCfg,
    launch_robot_hardware,
    wait_for_action_server,
    wait_for_rviz,
    wait_for_topic_publish,
    wait_for_topic_subscribe,
    wait_until_ready,
)
from skillet.envs.util import configclass


@configclass
class KinovaROS2EnvCfg(ROS2RLEnvCfg):
    """The configuration class for Kinova Gen3 Arm."""

    """Robot configuration"""

    # Whether to spin up real robot or not
    use_fake_hardware = "true"

    # IP of the robot
    robot_ip = "www.xxx.yyy.zzz"

    # Visualize
    vision = False

    # Default joint position of robot
    default_joint_positions = [0.0, 0.523599, 0.0, 1.5708, 0.0, 0.785398, 0.0, 0.0]  # Double format for ROS 2

    """RL environment configuration"""
    num_envs = 1

    device = "cuda"

    dt = 1 / 60

    decimation = 6.0

    episode_length_s = 5.0

    skills = ["reach_xyz"]

    joint_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    tcp_offset = [0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0]
    ee_link_name = "robotiq_85_base_link"
    base_link_name = "base_link"
    gripper_joint_names = ["robotiq_85_left_knuckle_joint"]


class KinovaROS2Env(ROS2RLEnv):
    """Kinova Gen3 7DoF ROS2 implementation."""

    def __init__(
        self, cfg: KinovaROS2EnvCfg, ros: Ros, render_mode: str | None = None, **kwargs: dict[str, Any]
    ) -> None:
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
        self.mass_matrix_topic = "/mass_matrix"
        self.gravity_vector_topic = "/gravity_vector"
        self.robot_description_topic = "/robot_info"
        self.body_pose_topic = "/robot_body_pose_w"
        self.body_vel_topic = "/robot_body_vel_w"
        # self.gripper_topic_type = (
        #     "control_msgs/action/GripperCommand"
        #     if cfg.use_fake_hardware == "true"
        #     else "control_msgs/action/ParallelGripperCommand"
        # )
        self.gripper_topic_type = "control_msgs/action/ParallelGripperCommand"
        self.realsense_snapshot_service = "/table_camera/realsense/get_latest_frame"
        self.realsense_snapshot_service_type = "iras_realsense_msgs/srv/GetLatestRgbd"

        self._ready = {
            "joint_states": False,
            "jacobians": False,
            "robot_info": False,  # includes joint centers
            "body_pose": False,
            "body_vel": False,
            "gravity_vector": False,
            "mass_matrices": False,
        }

        self.default_joint_positions = np.asarray(self.cfg.default_joint_positions)
        self.prev_action = np.zeros(shape=(8,))

        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(float("-inf"), float("inf"), shape=(16,))
        self.single_action_space = gym.spaces.Box(float("-inf"), float("inf"), shape=(8,))

        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        self._current_joint_positions = np.zeros(shape=len(self.joint_names))
        self._current_joint_velocities = np.zeros(shape=len(self.joint_names))
        self._current_joint_efforts = np.zeros(shape=len(self.joint_names))

        # Launch robot hardware in ROS2
        if self.cfg.launch_ros:
            launch_robot_hardware(
                cfg,
                cfg.ros2_workspace,
                "gen3_py",
                "gen3.launch.py",
                default_joint_positions=cfg.default_joint_positions,
            )

        # Wait for topics to be exposed before continuing
        wait_for_topic_publish(self.ros, self.joint_cmd_topic, "trajectory_msgs/msg/JointTrajectory")
        wait_for_action_server(self.ros, self.gripper_cmd_topic, self.gripper_topic_type)
        wait_for_topic_subscribe(self.ros, self.joint_state_topic, "sensor_msgs/JointState")
        # wait_for_rviz(self.ros)

        wait_for_topic_subscribe(self.ros, self.jacobian_topic, "gen3_cpp/msg/LinkMatrix")
        wait_for_topic_subscribe(self.ros, self.mass_matrix_topic, "gen3_cpp/msg/LinkMatrix")
        wait_for_topic_subscribe(self.ros, self.robot_description_topic, "gen3_cpp/msg/RobotInfo")
        wait_for_topic_subscribe(self.ros, self.body_pose_topic, "gen3_cpp/msg/BodyInfo")
        wait_for_topic_subscribe(self.ros, self.body_vel_topic, "gen3_cpp/msg/BodyInfo")
        wait_for_topic_subscribe(self.ros, self.gravity_vector_topic, "gen3_cpp/msg/LinkMatrix")

        # Subscribe to joint states
        self.joint_states_sub = Topic(self.ros, self.joint_state_topic, "sensor_msgs/JointState")

        def _update_robot_state(msg: dict[str, Any]) -> None:
            """Update the state of the robot by subscribing to robot topics."""
            self._current_joint_positions = np.asarray(
                [msg["position"][msg["name"].index(j)] for j in self.joint_names]
            ).astype(np.float32)
            self._current_joint_velocities = np.asarray(
                [msg["velocity"][msg["name"].index(j)] for j in self.joint_names]
            ).astype(np.float32)
            self._current_joint_efforts = np.asarray(
                [msg["effort"][msg["name"].index(j)] for j in self.joint_names]
            ).astype(np.float32)
            self._ready["joint_states"] = True

        self.joint_states_sub.subscribe(_update_robot_state)

        # Set up joint trajectory publisher
        self.joint_states_pub = Topic(self.ros, self.joint_cmd_topic, "trajectory_msgs/JointTrajectory")

        self.gripper_client = ActionClient(self.ros, self.gripper_cmd_topic, self.gripper_topic_type)
        self.realsense_service = Service(
            self.ros,
            self.realsense_snapshot_service,
            self.realsense_snapshot_service_type,
        )

        # Subscribe to jacobian topic
        self.jacobian_sub = Topic(self.ros, self.jacobian_topic, "gen3_cpp/msg/LinkMatrix")

        def _update_jacobians(msg: dict[str, Any]) -> None:
            """Update jacobians the robot by subscribing to jacobian topic."""
            self._current_jacobians = np.asarray(msg["matrix"], dtype=float).reshape(
                msg["num_links"], msg["rows"], msg["cols"]
            )
            self._ready["jacobians"] = True

        self.jacobian_sub.subscribe(_update_jacobians)

        # Subscribe to mass matrix topic
        self.mass_matrix_sub = Topic(self.ros, self.mass_matrix_topic, "gen3_cpp/msg/LinkMatrix")

        def _update_mass_matrix(msg: dict[str, Any]) -> None:
            """Update mass matrices by subscribing to mass matrix topic."""
            self._current_mass_matrices = np.asarray(msg["matrix"], dtype=float).reshape(msg["rows"], msg["cols"])
            self._ready["mass_matrices"] = True

        self.mass_matrix_sub.subscribe(_update_mass_matrix)

        # Subscribe gravity vector topic
        self.gravity_vector_sub = Topic(self.ros, self.gravity_vector_topic, "gen3_cpp/msg/LinkMatrix")

        def _update_gravity_vector(msg: dict[str, Any]) -> None:
            """Update gravity vector by subscribing to the gravity vector topic."""
            self._current_gravity_vector = np.asarray(msg["matrix"], dtype=float).reshape(msg["rows"])
            self._ready["gravity_vector"] = True

        self.gravity_vector_sub.subscribe(_update_gravity_vector)

        # Subscribe to robot_description
        self.robot_description_sub = Topic(self.ros, self.robot_description_topic, "gen3_cpp/msg/RobotInfo")

        def _update_robot_links_and_joints(msg: dict[str, Any]) -> None:
            """Update the state of the robot by subscribing to robot topics."""
            self._robot_links = list(msg["links"])
            self._robot_joints = list(msg["joints"])
            self._current_upper_joint_limits = np.asarray(msg["upper_limits"], dtype=float)
            self._current_lower_joint_limits = np.asarray(msg["lower_limits"], dtype=float)
            self._current_joint_centers = (self._current_upper_joint_limits + self._current_lower_joint_limits) / 2
            self._ready["robot_info"] = True

        self.robot_description_sub.subscribe(_update_robot_links_and_joints)

        # Subscribe to robot_description
        self.body_pose_sub = Topic(self.ros, self.body_pose_topic, "gen3_cpp/msg/BodyInfo")

        def _update_body_pose(msg: dict[str, Any]) -> None:
            """Update the state of the robot by subscribing to robot topics."""
            self._current_robot_body_pose_w = np.asarray(msg["body_w"], dtype=float).reshape(msg["num_links"], -1)
            self._current_robot_root_pose_w = np.asarray(msg["root_w"], dtype=float)
            self._ready["body_pose"] = True

        self.body_pose_sub.subscribe(_update_body_pose)

        # Subscribe to body vel topic
        self.body_vel_sub = Topic(self.ros, self.body_vel_topic, "gen3_cpp/msg/BodyInfo")

        def _update_body_vel(msg: dict[str, Any]) -> None:
            """Update the velocity of the robot by subscribing to robot topics."""
            self._current_robot_body_vel_w = np.asarray(msg["body_w"], dtype=float).reshape(msg["num_links"], -1)
            self._ready["body_vel"] = True

        self.body_vel_sub.subscribe(_update_body_vel)
        wait_until_ready(self._ready)

    def _pre_process_action(self, actions: torch.Tensor) -> np.ndarray:
        """Pre process the robot action.

        This function is responsible preprocessing the robot action (ie checking joint limits, etc).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, num_joints).

        """
        self.actions = actions.cpu().numpy().squeeze()  # Actions can only be 1 dimensional

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

        gripper_val = float(joint_pos[-1])
        gripper_val = max(0, min(gripper_val, 1)) * 0.8
        # if self.cfg.use_fake_hardware == "true":
        #     gripper_goal = {"command": {"position": gripper_val, "max_effort": 100.0}}
        # else:
        gripper_goal = {"command": {"name": self.cfg.gripper_joint_names, "position": [gripper_val]}}

        self.joint_states_pub.publish(joint_msg)

        _ = self.gripper_client.send_goal(
            gripper_goal, self._gripper_result_cb, self._gripper_feedback_cb, self._gripper_error_cb
        )

    def _reset_idx(self) -> None:
        """Reset environment based on specified indices to default position."""
        super()._reset_idx()

    def _get_dones(self) -> tuple[bool, bool]:
        """Return dones if longer than max episode length."""
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return False, truncated

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Return the observations from the robot."""
        return {
            "policy": np.concatenate(
                (self._current_joint_positions, self._current_joint_velocities), axis=0, dtype=np.float32
            )
        }

    def _get_rewards(self) -> np.ndarray:
        """Compute the rewards."""
        return np.array([0.0])

    def _gripper_result_cb(self, result: dict[str, Any]) -> None:
        """Gripper action result callback."""
        status = result.get("status")
        if hasattr(status, "name"):
            status = status.name
        message = result.get("message", "")
        if status.name == "SUCCEEDED":
            # print(f"[INFO] Gripper succeeded: {message}")
            self.gripper_ok = True

        elif status.name == "ABORTED":
            print(f"[INFO] Gripper aborted: {message}")
            self.gripper_ok = False

        elif status.name == "CANCELED":
            print(f"[INFO] Gripper canceled: {message}")
            self.gripper_ok = False
        else:
            print(f"[INFO] Unknown gripper result: {result}")
            self.gripper_ok = False
        pass

    def _gripper_feedback_cb(self, feedback: dict[str, Any]) -> None:
        """Gripper action feedback callback."""
        pos = feedback.get("position")
        effort = feedback.get("effort")
        stalled = feedback.get("stalled", False)

        if stalled:
            print(f"[INFO] Gripper feedback: pos={pos}, effort={effort}, stalled={stalled}")

    def _gripper_error_cb(self, err: dict[str, Any]) -> None:
        """Gripper action error callback."""
        code = err.get("code")
        message = err.get("message", "Unknown error")
        details = err.get("details")

        print(f"[INFO] Gripper error! code={code}, message={message}, details={details}")

        # Set internal state so higher-level logic can react
        self.gripper_ok = False
        self.last_gripper_error = err

    def _get_latest_rgbd(self) -> dict[str, Any]:
        """Request and decode the latest RGB-D snapshot synchronously.

        Returns:
            A dictionary containing:
                - rgb: HxWx3 uint8 RGB image
                - depth: HxW float32 depth image in meters
                - intrinsic_k: 3x3 float64 camera intrinsic matrix
                - camera_pose: 7D float64 array of camera pose in world frame (x, y, z, qx, qy, qz, qw)
                - timestamp: float timestamp of the RGB-D capture

        """
        result = self.realsense_service.call({})

        data = result.data
        if not data.get("success", False):
            raise RuntimeError(f"RGB-D snapshot request failed: {data.get('message', 'unknown error')}")

        rgb_jpeg = self._decode_uint8_payload(data["rgb_jpeg"], field_name="rgb_jpeg")
        if rgb_jpeg.size < 2 or not (rgb_jpeg[0] == 0xFF and rgb_jpeg[1] == 0xD8):
            raise RuntimeError("rgb_jpeg payload does not have a JPEG SOI header (0xFFD8)")
        rgb_bgr = cv2.imdecode(rgb_jpeg, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError("Failed to decode RGB JPEG from service response")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        depth_png = self._decode_uint8_payload(data["depth_png"], field_name="depth_png")
        if depth_png.size < 8 or not np.array_equal(depth_png[:8], np.asarray([137, 80, 78, 71, 13, 10, 26, 10])):
            raise RuntimeError("depth_png payload does not have a PNG signature")
        depth = cv2.imdecode(depth_png, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError("Failed to decode depth PNG from service response")
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16, copy=False)

        camera_info = data["camera_info"]
        k = np.asarray(camera_info["k"], dtype=np.float64).reshape(3, 3)

        tf_msg = data["t_world_cam"]
        t = tf_msg["transform"]["translation"]
        q = tf_msg["transform"]["rotation"]
        translation = np.asarray([t["x"], t["y"], t["z"]], dtype=np.float64)
        # quaternion is in xyzw ROS format, to be consistent with joint states
        # wrapper must convert to wxyz format
        quat_xyzw = np.asarray([q["x"], q["y"], q["z"], q["w"]], dtype=np.float64)
        camera_pos_quat = np.concatenate((translation, quat_xyzw), axis=0)

        stamp = data.get("stamp", {"sec": 0, "nanosec": 0})
        timestamp = float(stamp.get("sec", 0)) + float(stamp.get("nanosec", 0)) * 1e-9

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsic_k": k,
            "camera_pose": camera_pos_quat,
            "timestamp": timestamp,
        }

    @staticmethod
    def _decode_uint8_payload(payload: Any, field_name: str) -> np.ndarray:
        """Decode rosbridge-serialized uint8[] payload into np.uint8."""
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return np.frombuffer(payload, dtype=np.uint8)

        if isinstance(payload, list):
            return np.asarray(payload, dtype=np.uint8)

        if isinstance(payload, dict):
            if "bytes" in payload:
                return KinovaROS2Env._decode_uint8_payload(payload["bytes"], field_name=field_name)
            if "data" in payload:
                return KinovaROS2Env._decode_uint8_payload(payload["data"], field_name=field_name)

        if isinstance(payload, str):
            try:
                # rosbridge commonly serializes uint8[] as base64 strings.
                return np.frombuffer(base64.b64decode(payload, validate=True), dtype=np.uint8)
            except Exception:
                # Fallback for raw byte-preserving strings.
                return np.frombuffer(payload.encode("latin-1"), dtype=np.uint8)

        raise TypeError(f"Unsupported payload type for {field_name}: {type(payload).__name__}")
