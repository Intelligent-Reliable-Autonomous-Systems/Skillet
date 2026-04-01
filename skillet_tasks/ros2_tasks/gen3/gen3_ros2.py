"""gen3_ros2.py.

Gen3 Arm class for ROS2 RL

Written by Will Solow, 2026

"""

import base64
import math
import threading
import time
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import torch
from roslibpy import ActionClient, Ros, Service, Topic

from skillet.core.spaces import ActionSpec
from skillet.envs.ros2 import (
    ROS2Env,
    ROS2EnvCfg,
    launch_robot_hardware,
    wait_for_action_server,
    wait_for_rviz,
    wait_for_topic_publish,
    wait_for_topic_subscribe,
    wait_until_ready,
)
from skillet.envs.util import configclass
from skillet.perception.localization import CameraLocalizer
from skillet.policy.specs import JOINTS_SPEC


@configclass
class Gen3ROS2EnvCfg(ROS2EnvCfg):
    """The configuration class for Kinova Gen3 Arm."""

    """Robot configuration"""

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

    decimation = 1.0

    episode_length_s = 1e9

    skills = ["reach_xyz"]

    joint_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    tcp_offset = [0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0]
    ee_link_name = "end_effector_link"
    base_link_name = "base_link"
    gripper_joint_names = ["robotiq_85_left_knuckle_joint"]

    arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]

    gripper_cmd_topic = "/robotiq_gripper_controller/gripper_cmd"

    move_group_name = "manipulator"

    tool_frame_name = "tcp"

    base_apriltag_id = 3


class Gen3ROS2Env(ROS2Env):
    """Kinova Gen3 7DoF ROS2 implementation."""

    def __init__(self, cfg: ROS2EnvCfg, ros: Ros, render_mode: str | None = None, **kwargs: dict[str, Any]) -> None:
        """Initialize Gen3 Arm ROS2."""
        super().__init__(cfg, ros, render_mode, **kwargs)

        self.joint_names = np.asarray(self.cfg.arm_joint_names + self.cfg.gripper_joint_names)
        self.arm_joint_names = np.asarray(self.cfg.arm_joint_names)
        self.gripper_joint_names = np.asarray(self.cfg.gripper_joint_names)

        self.joint_state_topic = "/joint_states"
        self.joint_cmd_topic = "/safety/joint_trajectory_controller/joint_trajectory"
        self.twist_vel_topic = "/safety/twist_controller/commands"
        self.gripper_cmd_topic = self.cfg.gripper_cmd_topic
        self.jacobian_topic = "/jacobian"
        self.mass_matrix_topic = "/mass_matrix"
        self.gravity_vector_topic = "/gravity_vector"
        self.robot_description_topic = "/robot_info"
        self.body_pose_topic = "/robot_body_pose_w"
        self.body_vel_topic = "/robot_body_vel_w"
        self.gripper_topic_type = (
            "control_msgs/action/GripperCommand"
            if self.cfg.use_fake_hardware
            else "control_msgs/action/ParallelGripperCommand"
        )
        self.moveit_cmd_topic = "/move_action"
        self.moveit_cmd_topic_type = "moveit_msgs/action/MoveGroup"
        self.realsense_snapshot_service = "/table_camera/realsense/get_latest_frame"
        self.realsense_snapshot_service_type = "iras_realsense_msgs/srv/GetLatestRgbd"

        self.active_controller = "joint_trajectory_controller"

        self._ready = {
            "joint_states": False,
            "jacobians": False,
            "robot_info": False,
            "body_pose": False,
            "body_vel": False,
            "gravity_vector": False,
            "mass_matrices": False,
        }

        self.default_joint_positions = np.asarray(self.cfg.default_joint_positions)
        self.prev_action = np.zeros(shape=(len(self.joint_names),))

        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(
            float("-inf"), float("inf"), shape=(2 * len(self.joint_names),)
        )
        self.single_action_space = gym.spaces.Box(float("-inf"), float("inf"), shape=(len(self.cfg.joint_ids),))

        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        self._action_specs = [
            JOINTS_SPEC.bind(
                n_joints=len(self.cfg.joint_ids),
            ).replace(device=self.device),
            ActionSpec(
                name="twist_tcp",
                space=gym.spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(6 + len(self.cfg.gripper_joint_names),)
                ),
                is_torch=True,
                is_batched=True,
                n_envs=-1,
                device=self.device,
            ),
            ActionSpec(
                name="moveit_tcp_quat",
                space=gym.spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(7 + len(self.cfg.gripper_joint_names),)
                ),
                is_torch=True,
                is_batched=True,
                n_envs=-1,
                device=self.device,
            ),
            ActionSpec(
                name="moveit_tcp_vel",
                space=gym.spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(7 + len(self.cfg.gripper_joint_names),)
                ),
                is_torch=True,
                is_batched=True,
                n_envs=-1,
                device=self.device,
            ),
        ]

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
        wait_for_topic_publish(self.ros, self.twist_vel_topic, "geometry_msgs/msg/Twist")
        wait_for_action_server(self.ros, self.gripper_cmd_topic, self.gripper_topic_type)
        wait_for_action_server(self.ros, self.moveit_cmd_topic, self.moveit_cmd_topic_type)
        wait_for_topic_subscribe(self.ros, self.joint_state_topic, "sensor_msgs/msg/JointState")
        wait_for_rviz(self.ros)

        wait_for_topic_subscribe(self.ros, self.jacobian_topic, "gen3_cpp/msg/LinkMatrix")
        wait_for_topic_subscribe(self.ros, self.mass_matrix_topic, "gen3_cpp/msg/LinkMatrix")
        wait_for_topic_subscribe(self.ros, self.robot_description_topic, "gen3_cpp/msg/RobotInfo")
        wait_for_topic_subscribe(self.ros, self.body_pose_topic, "gen3_cpp/msg/BodyInfo")
        wait_for_topic_subscribe(self.ros, self.body_vel_topic, "gen3_cpp/msg/BodyInfo")
        wait_for_topic_subscribe(self.ros, self.gravity_vector_topic, "gen3_cpp/msg/LinkMatrix")

        # Set up controller interfaces
        self.moveit_client = ActionClient(self.ros, self.moveit_cmd_topic, self.moveit_cmd_topic_type)
        self.joint_states_pub = Topic(self.ros, self.joint_cmd_topic, "trajectory_msgs/msg/JointTrajectory")
        self.twist_vel_pub = Topic(self.ros, self.twist_vel_topic, "geometry_msgs/msg/Twist")
        self.gripper_client = ActionClient(self.ros, self.gripper_cmd_topic, self.gripper_topic_type)
        self.controller_client = Service(
            self.ros, "/controller_manager/switch_controller", "controller_manager_msgs/srv/SwitchController"
        )
        self.curr_gripper_goal = None
        self.curr_moveit_goal = None

        # Subscribe to ROS2 topics
        self.joint_states_sub = Topic(self.ros, self.joint_state_topic, "sensor_msgs/msg/JointState")
        self.joint_states_sub.subscribe(self._update_robot_state)

        self.jacobian_sub = Topic(self.ros, self.jacobian_topic, "gen3_cpp/msg/LinkMatrix")
        self.jacobian_sub.subscribe(self._update_jacobians)

        self.mass_matrix_sub = Topic(self.ros, self.mass_matrix_topic, "gen3_cpp/msg/LinkMatrix")
        self.mass_matrix_sub.subscribe(self._update_mass_matrix)

        self.gravity_vector_sub = Topic(self.ros, self.gravity_vector_topic, "gen3_cpp/msg/LinkMatrix")
        self.gravity_vector_sub.subscribe(self._update_gravity_vector)

        self.robot_description_sub = Topic(self.ros, self.robot_description_topic, "gen3_cpp/msg/RobotInfo")
        self.robot_description_sub.subscribe(self._update_robot_links_and_joints)

        self.body_pose_sub = Topic(self.ros, self.body_pose_topic, "gen3_cpp/msg/BodyInfo")
        self.body_pose_sub.subscribe(self._update_body_pose)

        self.body_vel_sub = Topic(self.ros, self.body_vel_topic, "gen3_cpp/msg/BodyInfo")
        self.body_vel_sub.subscribe(self._update_body_vel)

        wait_until_ready(self._ready)

        self.realsense_service = Service(
            self.ros,
            self.realsense_snapshot_service,
            self.realsense_snapshot_service_type,
        )

        self._camera_localizer = CameraLocalizer(apriltag_size_m=0.1, apriltag_id=self.cfg.base_apriltag_id)

    def _pre_process_action(self, actions: torch.Tensor, action_spec: ActionSpec[Any] | None = None) -> np.ndarray:
        """Pre process the robot action.

        This function is responsible preprocessing the robot action (ie checking joint limits, etc).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, num_joints).
            action_spec: The action specification of the batch of actions. Note: Currently assumes all actions are
                of same spec.

        """
        self.actions = actions.cpu().numpy().squeeze()  # Actions can only be 1 dimensional

        return self.actions

    def _publish_action_to_ros(
        self, action: np.ndarray, action_spec: ActionSpec[Any] | None = None, duration: float = 3, timeout: float = 30
    ) -> None:
        """Publish the robot action.

        Args:
            action: NDArray of joint positions
            action_spec: Action specficiation of each of the actions
            duration: Duration of trajectory
            timeout: duration of time out

        """
        # Publish BLOCKING gripper command. To keep the gripper stationary
        # Assumes we can either move joints or close gripper, not both
        if self._publish_gripper(action, action_spec, close_time=4.0):
            return

        if action_spec is None or action_spec.name == "joints":
            self._publish_joint_spec(action, duration)
        elif action_spec.name == "twist_tcp":
            self._publish_twist_tcp_spec(action)
        elif "moveit" in action_spec.name:
            self._publish_moveit_spec(action, action_spec=action_spec)
        else:
            raise ValueError(f"Unknown Action Specification `{action_spec.name}`")

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

    def _supports_action_spec(self, action_spec: ActionSpec[Any] | None = None) -> bool:
        if action_spec is None:
            return True
        return action_spec.name in [s.name for s in self._action_specs]

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
        # TODO decide how we want to toggle between ROS2 vs Skillet side camera localization
        # camera_pos_quat = self._camera_localizer.get_camera_pose(rgb=rgb, intrinsic_k=k)

        # TODO I feel like we should localize the camera on the skillet side, not ROS side
        # if q["x"] == 0.0 and q["y"] == 0.0 and q["z"] == 0.0:
        #     # TODO: This is a hack for fake hardware
        #     if not hasattr(self, "_camera_localizer"):
        #         self._camera_localizer = CameraLocalizer(apriltag_size_m=0.1, apriltag_id=0)

        stamp = data.get("stamp", {"sec": 0, "nanosec": 0})
        timestamp = float(stamp.get("sec", 0)) + float(stamp.get("nanosec", 0)) * 1e-9

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsic_k": k,
            "camera_pose": camera_pos_quat,
            "timestamp": timestamp,
        }

    def _publish_gripper(
        self,
        joint_pos: np.ndarray,
        action_spec: ActionSpec,
        timeout: int = 30,
        duration: int = 3,
        close_time: float = 0.5,
    ) -> None:
        """Publish a joint position to the joint trajectory controller.

        Args:
            joint_pos: Joint positions of the robot.
            action_spec: what stationary action to specify
            timeout: Duration before timeout
            duration: what duration to specify
            close_time: time the gripper takes to close

        """
        if self.cfg.use_fake_hardware:
            return False
        new_gripper_goal = False

        gripper_val = float(joint_pos[-1])
        gripper_val = max(0, min(gripper_val, 1)) * 0.8
        gripper_goal = (
            {"command": {"position": gripper_val, "max_effort": 100.0}}
            if self.cfg.use_fake_hardware
            else {"command": {"name": self.cfg.gripper_joint_names, "position": [gripper_val], "effort": [0.0]}}
        )

        if gripper_goal != self.curr_gripper_goal:
            result_container = {}
            done_event = threading.Event()

            def on_result(result: dict[str, Any]) -> None:
                result_container["result"] = result
                done_event.set()

            def on_feedback(feedback: dict[str, Any]) -> None:
                self._gripper_feedback_cb(feedback)

            def on_error(error: dict[str, Any]) -> None:
                result_container["error"] = error
                done_event.set()

            if action_spec is None or action_spec.name == "joints":
                self._publish_joint_spec(joint_pos, duration)
            elif action_spec.name == "twist_tcp":
                self._publish_twist_tcp_spec(np.zeros_like(joint_pos))
            _ = self.gripper_client.send_goal(gripper_goal, on_result, on_feedback, on_error)
            self.curr_gripper_goal = gripper_goal
            new_gripper_goal = True
            time.sleep(close_time)

            finished = done_event.wait(timeout=timeout)

            if not finished:
                raise TimeoutError(f"MoveIt goal timed out after {timeout}s")

            if "error" in result_container:
                raise RuntimeError(f"MoveIt action error: {result_container['error']}")

        return new_gripper_goal

    def _publish_joint_spec(self, joint_pos: np.ndarray, duration: float = 3) -> None:
        """Publish a joint position to the joint trajectory controller.

        Args:
            joint_pos: Joint positions of the robot.
            duration: Duration of trajectory

        """
        if self.active_controller != "joint_trajectory_controller":
            print("[INFO] Switching controller to `joint_trajectory_controller`")
            if not self.switch_controllers(activate=["joint_trajectory_controller"], deactivate=["twist_controller"]):
                print("[INFO] Unable to switch controller to `joint_trajectory_controller`. Aborting trajectory.")
                return
            self.active_controller = "joint_trajectory_controller"
            print("[INFO] Successfully switched controller to `joint_trajectory_controller`")

        joint_msg = {
            "joint_names": self.arm_joint_names.tolist(),
            "points": [
                {
                    "positions": joint_pos[: len(self.arm_joint_names)].tolist(),
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
        self.joint_states_pub.publish(joint_msg)

    def _publish_twist_tcp_spec(self, twist: np.ndarray) -> None:
        """Publish a twist in the tcp frame to the robot twist controller.

        Args:
            twist: Cartesian twist command (velocity) in XYZ (linear) and RPY (angular).

        """
        if not self.cfg.use_fake_hardware and self.active_controller != "twist_controller":
            # print("[INFO] Switching controller to `twist_controller`") # TODO uncomment
            if not self.switch_controllers(activate=["twist_controller"], deactivate=["joint_trajectory_controller"]):
                # print("[INFO] Unable to switch controller to `twist_controller`. Aborting trajectory.")
                return
            self.active_controller = "twist_controller"
            print("[INFO] Successfully switched controller to `twist_controller`")

        twist = twist.tolist()
        twist_cmd = {
            # "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "base_link"},
            # "twist": {
            "linear": {"x": twist[0], "y": twist[1], "z": twist[2]},
            "angular": {"x": twist[3], "y": twist[4], "z": twist[5]},
            # },
        }

        self.twist_vel_pub.publish(twist_cmd)

    def _publish_moveit_spec(
        self, moveit_pose: np.ndarray, action_spec: ActionSpec[Any] | None = None, timeout: int = 30
    ) -> None:
        """Publish a MoveIt trajectory to move the arm to specified joint positions.

        Args:
            moveit_pose: The desired joint positions to send to moveit
            action_spec: Action specification for moveit
            timeout: Duration before the threading event should time out

        """
        if self.active_controller != "joint_trajectory_controller":
            print("[INFO] Switching controller to `joint_trajectory_controller`")
            if not self.switch_controllers(activate=["joint_trajectory_controller"], deactivate=["twist_controller"]):
                print("[INFO] Unable to switch controller to `joint_trajectory_controller`. Aborting trajectory.")
                return
            self.active_controller = "joint_trajectory_controller"
            print("[INFO] Successfully switched controller to `joint_trajectory_controller`")

        if action_spec.name == "moveit_joint":
            moveit_goal = self._moveit_joint_msg(moveit_pose)
        elif "moveit_tcp" in action_spec.name:
            moveit_goal = self._moveit_tcp_pose_msg(moveit_pose)
        else:
            raise ValueError(f"Unsupported MoveIt ActionSpec `{action_spec.name}`")

        result_container = {}
        done_event = threading.Event()

        def on_result(result: dict[str, Any]) -> None:
            result_container["result"] = result
            done_event.set()

        def on_feedback(feedback: dict[str, Any]) -> None:
            self._moveit_feedback_cb(feedback)

        def on_error(error: dict[str, Any]) -> None:
            result_container["error"] = error
            done_event.set()

        if self.curr_moveit_goal != moveit_goal:
            _ = self.moveit_client.send_goal(moveit_goal, on_result, on_feedback, on_error)

        finished = done_event.wait(timeout=timeout)

        if not finished:
            raise TimeoutError(f"MoveIt goal timed out after {timeout}s")

        if "error" in result_container:
            raise RuntimeError(f"MoveIt action error: {result_container['error']}")

    @staticmethod
    def _decode_uint8_payload(payload: Any, field_name: str) -> np.ndarray:
        """Decode rosbridge-serialized uint8[] payload into np.uint8."""
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return np.frombuffer(payload, dtype=np.uint8)

        if isinstance(payload, list):
            return np.asarray(payload, dtype=np.uint8)

        if isinstance(payload, dict):
            if "bytes" in payload:
                return Gen3ROS2Env._decode_uint8_payload(payload["bytes"], field_name=field_name)
            if "data" in payload:
                return Gen3ROS2Env._decode_uint8_payload(payload["data"], field_name=field_name)

        if isinstance(payload, str):
            try:
                # rosbridge commonly serializes uint8[] as base64 strings.
                return np.frombuffer(base64.b64decode(payload, validate=True), dtype=np.uint8)
            except Exception:
                # Fallback for raw byte-preserving strings.
                return np.frombuffer(payload.encode("latin-1"), dtype=np.uint8)

        raise TypeError(f"Unsupported payload type for {field_name}: {type(payload).__name__}")

    def _moveit_feedback_cb(self, feedback: dict[str, Any]) -> None:
        """MoveIt action feedback callback."""
        pos = feedback.get("position")
        effort = feedback.get("effort")
        stalled = feedback.get("stalled", False)

        if stalled:
            print(f"[INFO] MoveIt feedback: pos={pos}, effort={effort}, stalled={stalled}")

    def _gripper_result_cb(self, result: dict[str, Any]) -> None:
        """Gripper action result callback."""
        status = result.get("status")
        if hasattr(status, "name"):
            status = status.name
        message = result.get("message", "")
        if status == "SUCCEEDED":
            # print(f"[INFO] Gripper succeeded: {message}")
            self.gripper_ok = True

        elif status == "ABORTED":
            print(f"[INFO] Gripper aborted: {message}")
            self.gripper_ok = False

        elif status == "CANCELED":
            print(f"[INFO] Gripper canceled: {message}")
            self.gripper_ok = False
        else:
            print(f"[INFO] Unknown gripper result: {result}")
            self.gripper_ok = False

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

    def _moveit_joint_msg(self, joint_pos: np.ndarray) -> dict:
        joint_pos = joint_pos.tolist()
        return {
            "request": {
                "group_name": self.cfg.move_group_name,
                "start_state": {"is_diff": True},
                "goal_constraints": [
                    {
                        "joint_constraints": [
                            {
                                "joint_name": j,
                                "position": joint_pos[i],
                                "tolerance_above": 0.01,
                                "tolerance_below": 0.01,
                                "weight": 1.0,
                            }
                            for i, j in enumerate(self.arm_joint_names)
                        ]
                    }
                ],
                "num_planning_attempts": 10,
                "allowed_planning_time": 5.0,
                "max_velocity_scaling_factor": 0.3,
                "max_acceleration_scaling_factor": 0.3,
            }
        }

    def _moveit_tcp_pose_msg(self, tcp_pose_b: np.ndarray) -> dict:
        tcp_pose_b = tcp_pose_b.tolist()

        return {
            "request": {
                "group_name": self.cfg.move_group_name,
                "start_state": {"is_diff": True},
                "goal_constraints": [
                    {
                        "name": "pose_goal",
                        "position_constraints": [
                            {
                                "header": {"frame_id": "base_link"},
                                "link_name": self.cfg.tool_frame_name,
                                "constraint_region": {
                                    "primitives": [
                                        {
                                            "type": 2,  # BOX type
                                            "dimensions": [0.03, 0.03, 0.03],  # tolerance box (meters)
                                        }
                                    ],
                                    "primitive_poses": [
                                        {
                                            "position": {"x": tcp_pose_b[0], "y": tcp_pose_b[1], "z": tcp_pose_b[2]},
                                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                                        }
                                    ],
                                },
                                "weight": 1.0,
                            }
                        ],
                        # --- ORIENTATION (RPY → quaternion) ---
                        "orientation_constraints": [
                            {
                                "header": {"frame_id": "base_link"},
                                "link_name": self.cfg.tool_frame_name,
                                "orientation": {
                                    "x": tcp_pose_b[4],
                                    "y": tcp_pose_b[5],
                                    "z": tcp_pose_b[6],
                                    "w": tcp_pose_b[3],
                                    # "x": 1.0,#tcp_pose_b[4],
                                    # "y": 0.0, #tcp_pose_b[5],
                                    # "z": 0.0, #tcp_pose_b[6],
                                    # "w": 0.0, #tcp_pose_b[3],
                                },  # Skillet quaternion is in WXYZ
                                "absolute_x_axis_tolerance": 0.1,  # radians
                                "absolute_y_axis_tolerance": 0.1,
                                "absolute_z_axis_tolerance": 0.1,
                                "weight": 1.0,
                            }
                        ],
                    }
                ],
                "num_planning_attempts": 10,
                "allowed_planning_time": 5.0,
                "max_velocity_scaling_factor": 0.2,
                "max_acceleration_scaling_factor": 0.2,
            },
            "planning_options": {
                "plan_only": False,
                "replan": True,
                "replan_attempts": 3,
            },
        }
