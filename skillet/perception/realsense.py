"""RealSense RGB-D environment.

Provides a simple environment that streams RGB-D observations from an Intel
RealSense camera in a format compatible with the ROS2 RGB-D pipeline and
`ROS2EnvWrapper`'s `"rgb-d"` observation.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import gymnasium as gym
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector as AprilTagDetector
from typing_extensions import override

from skillet.core import ActionSpec
from skillet.core.env import _EnvironmentBase
from skillet.core.spaces import ObservationSpec
from skillet.envs.specs import RGBD_SPEC_BATCHED
from skillet.perception.localization import CameraLocalizer
from skillet.scene.utils import depth_to_colormap_np

if TYPE_CHECKING:
    from jaxtyping import Float, UInt8, UInt16


@dataclass(frozen=True)
class Frame:
    """Class for storing data from a camera frame."""

    serial: str
    timestamp: float
    rgb: UInt8[np.ndarray, "h w 3"]
    intrinsics: Float[np.ndarray, "3 3"]  # Camera intrinsics matrix
    depth: Float[np.ndarray, "h w"] | None = None  # Onboard sensor depth in metres (optional)

    @property
    def bgr(self) -> UInt8[np.ndarray, "h w 3"]:
        """Convert from RGB to BGR."""
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)


@dataclass(frozen=True, kw_only=True)
class RealsenseFrame(Frame):
    """Frame from RealSense which also includes the IR stereo pair."""

    ir_left: UInt8[np.ndarray, "h w"] | None = None  # IR left uint8
    ir_right: UInt8[np.ndarray, "h w"] | None = None  # IR right uint8
    depth_raw: UInt16[np.ndarray, "h w"] | None = None  # Raw depth uint16 millimeters


@dataclass(frozen=True)
class RealsenseIntrinsics:
    """Intrinsics for RealSense camera."""

    k_color: Float[np.ndarray, "3 3"]  # Color camera matrix
    k_ir: Float[np.ndarray, "3 3"]  # IR camera matrix
    baseline_ir: float  # Meters (IR baseline)
    t_color_from_ir: Float[np.ndarray, "4 4"]  # Transform from IR to color
    distortion_color: Float[np.ndarray, 5]  # Color camera distortion coefficients


class RealsenseEnv(_EnvironmentBase):
    """Environment that streams RGB-D observations from an Intel RealSense camera.

    The raw RGB-D snapshot matches `_get_latest_rgbd` in
    `gen3_ros2.py` (RGB HxWx3, depth HxW, intrinsics, pose, timestamp).
    The public observation returned by `get_observation()` also mirrors the
    post-processing done in `ROS2EnvWrapper` for the `"rgb-d"` observation:

    - quaternion converted from ROS xyzw -> IsaacLab wxyz
    - RGB transposed from (H, W, 3) -> (3, H, W)
    - Depth expanded from (H, W) -> (1, H, W)

    Camera pose is hardcoded to the world origin with identity orientation:
    position (0, 0, 0), quaternion (0, 0, 0, 1) in ROS xyzw order.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        apriltag_pose: np.ndarray = np.array([0.12, 0.005, 0.0, 0.0, 0.0, 0.7071068, 0.7071068]),
        apriltag_size_m: float = 0.100,
        apriltag_id: int = 0,
    ) -> None:
        """Initialize the RealSense pipeline and RGB-D observation space."""
        self.width = width
        self.height = height
        self.fps = fps

        # RealSense pipeline + streams.
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self._config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self._tag_detector = AprilTagDetector()
        self._T_base_to_tag = make_T(quat_xyzw_to_R(*list(apriltag_pose[3:7])), list(apriltag_pose[:3]))
        self._roll_180 = make_T(quat_xyzw_to_R(1.0, 0.0, 0.0, 0.0), [0.0, 0.0, 0.0])
        self._T_base_to_tag = self._T_base_to_tag @ self._roll_180
        self._latest_camera_pose = T_to_xyz_quat_xyzw(self._T_base_to_tag)
        self._apriltag_size_m = apriltag_size_m
        self._apriltag_id = apriltag_id

        self._camera_localizer = CameraLocalizer(
            apriltag_pose=apriltag_pose, apriltag_size_m=apriltag_size_m, apriltag_id=apriltag_id
        )

        self._profile = self._pipeline.start(self._config)

        # Align depth to color so pixels correspond.
        self._align = rs.align(rs.stream.color)

        # Intrinsics for the color stream.
        color_stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self._intrinsic_k = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Depth scale (meters per depth unit) retained for reference; we keep depth
        # in uint16 units to match the ROS2 RGB-D observation convention.
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        # Unbatched RGB-D observation spec, numpy-based, matching ROS2 `"rgb-d"` spec
        # in `ROS2EnvWrapper` but without batching.
        self.obs_spec_rgbd: ObservationSpec[Mapping[str, Any]] = ObservationSpec[Mapping[str, Any]](
            space=gym.spaces.Dict(
                {
                    "rgb": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(3, self.height, self.width),
                        dtype=np.uint8,
                    ),
                    "depth": gym.spaces.Box(
                        low=0,
                        high=10,
                        shape=(1, self.height, self.width),
                        dtype=np.float32,
                    ),
                    "intrinsic_k": gym.spaces.Box(
                        low=0.0,
                        high=2000.0,
                        shape=(3, 3),
                        dtype=np.float32,
                    ),
                    "camera_pose": gym.spaces.Box(
                        low=-10.0,
                        high=10.0,
                        shape=(7,),
                        dtype=np.float32,
                    ),
                    "timestamp": gym.spaces.Box(
                        low=0.0,
                        high=1e10,
                        shape=(),
                        dtype=np.float64,
                    ),
                }
            ),
            name="rgb-d",
            is_torch=False,
            is_batched=False,
        )
        self.obs_spec_rgbd = (
            RGBD_SPEC_BATCHED.bind(height=self.height, width=self.width).replace(is_torch=False).unbatched()
        )

        self._closed = False

    @property
    def obs_spec(self) -> ObservationSpec[Mapping[str, Any]]:
        """Return the default (unbatched) RGB-D observation specification."""
        return self.obs_spec_rgbd

    def supports_observation_spec(self, obs_spec: ObservationSpec[Mapping[str, Any]]) -> bool:
        """Return True if the given observation spec is supported."""
        return obs_spec.name == "rgb-d"

    @override
    def supports_action_spec(self, action_spec: ActionSpec[Any]) -> bool:
        return True

    def _get_latest_rgbd_raw(self) -> dict[str, Any]:
        """Grab the latest RGB-D snapshot in the raw ROS-style format.

        Returns:
            A dictionary containing:
              - ``rgb``: (H, W, 3) uint8 RGB image
              - ``depth``: (H, W) uint16 depth image
              - ``intrinsic_k``: (3, 3) float64 camera intrinsic matrix
              - ``camera_pose``: 7D float64 array (x, y, z, qx, qy, qz, qw) in ROS xyzw
              - ``timestamp``: float timestamp in seconds

        """
        if self._closed:
            raise RuntimeError("RealsenseEnv is closed. Create a new instance to continue streaming.")

        frames = self._pipeline.wait_for_frames()
        frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to acquire both color and depth frames from RealSense.")

        # (H, W, 3) BGR uint8 -> RGB
        color_bgr = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # (H, W) uint16 depth image (no conversion to meters here).
        depth = np.asanyarray(depth_frame.get_data()).astype(np.uint16, copy=False)

        # camera_params = (self._intrinsic_k[0,0], self._intrinsic_k[1,1], self._intrinsic_k[0,2], self._intrinsic_k[1,2])
        # tag_size_m = self._apriltag_size_m
        # gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # detections = self._tag_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size_m)
        # if detections:
        #     for detection in detections:
        #         if detection.tag_id == self._apriltag_id:
        #             T_tag_cam = make_T(detection.pose_R, detection.pose_t.reshape(3))
        #             T_cam_tag = _inv_T(T_tag_cam)
        #             T_base_cam = self._T_base_to_tag @ T_cam_tag
        #             self._latest_camera_pose = T_to_xyz_quat_xyzw(T_base_cam)
        #             break

        # Wall-clock timestamp in seconds.
        timestamp = float(time.time())

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsic_k": self._intrinsic_k,
            "camera_pose": self._camera_localizer.get_camera_pose(rgb=rgb, intrinsic_k=self._intrinsic_k),
            "timestamp": timestamp,
        }

    def get_observation(
        self,
        obs_spec: ObservationSpec[Mapping[str, Any]] | None = None,
    ) -> Mapping[str, Any]:
        """Return the latest RGB-D observation, optionally cast to a spec."""
        latest = self._get_latest_rgbd_raw()

        # ROS xyzw format -> IsaacLab wxyz format
        camera_pose = latest["camera_pose"].copy()
        q = camera_pose[3:7]
        camera_pose[3:7] = q[[3, 0, 1, 2]]

        # RGB is (H, W, 3) -> (3, H, W)
        rgb = latest["rgb"].transpose((2, 0, 1))

        # Depth is (H, W) -> (1, H, W)
        depth = np.expand_dims(latest["depth"], axis=0) / 1000.0
        depth = depth.astype(np.float32)

        obs: dict[str, Any] = {
            "rgb": rgb,
            "depth": depth,
            "intrinsic_k": latest["intrinsic_k"],
            "camera_pose": camera_pose,
            "timestamp": latest["timestamp"],
        }

        if obs_spec is None:
            return obs
        if not self.supports_observation_spec(obs_spec):
            raise ValueError(f"Observation spec {obs_spec} not supported by RealsenseEnv.")
        return obs_spec.cast(obs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.get_observation(), {}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self.get_observation(), 0.0, False, False, {}

    def close(self) -> None:
        """Stop the RealSense pipeline and release resources."""
        if not self._closed:
            self._pipeline.stop()
            self._closed = True

    def __del__(self) -> None:
        """Attempt to clean up the RealSense pipeline on garbage collection."""
        try:  # noqa: SIM105
            self.close()
        except Exception:
            # Best-effort cleanup; ignore errors on interpreter shutdown.
            pass


class RealsenseCameraLocalizer:
    """Class that manages the Realsense camera streaming with AprilTag estimation.

    Used as a helper class in the Kortex API environments to avoid requiring ROS2.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        apriltag_pose: np.ndarray = np.array([0.12, 0.005, 0.0, 0.0, 0.0, 0.7071068, 0.7071068]),
        apriltag_size_m: float = 0.100,
        apriltag_id: int = 3,
    ) -> None:
        """Initialize the RealSense pipeline and RGB-D observation space."""
        self.width = width
        self.height = height
        self.fps = fps

        # RealSense pipeline + streams.
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self._config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self._tag_detector = AprilTagDetector()
        self._T_base_to_tag = make_T(quat_xyzw_to_R(*list(apriltag_pose[3:7])), list(apriltag_pose[:3]))
        self._roll_180 = make_T(quat_xyzw_to_R(1.0, 0.0, 0.0, 0.0), [0.0, 0.0, 0.0])
        self._T_base_to_tag = self._T_base_to_tag @ self._roll_180
        self._latest_camera_pose = T_to_xyz_quat_xyzw(self._T_base_to_tag)
        self._apriltag_size_m = apriltag_size_m
        self._apriltag_id = apriltag_id

        self._camera_localizer = CameraLocalizer(
            apriltag_pose=apriltag_pose, apriltag_size_m=apriltag_size_m, apriltag_id=apriltag_id
        )

        self._profile = self._pipeline.start(self._config)

        # Align depth to color so pixels correspond.
        self._align = rs.align(rs.stream.color)

        # Intrinsics for the color stream.
        color_stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self._intrinsic_k = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Depth scale (meters per depth unit) retained for reference; we keep depth
        # in uint16 units to match the ROS2 RGB-D observation convention.
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        self._closed = False

    def _get_latest_rgbd_raw(self) -> dict[str, Any]:
        """Grab the latest RGB-D snapshot in the raw ROS-style format.

        Returns:
            A dictionary containing:
              - ``rgb``: (H, W, 3) uint8 RGB image
              - ``depth``: (H, W) uint16 depth image
              - ``intrinsic_k``: (3, 3) float64 camera intrinsic matrix
              - ``camera_pose``: 7D float64 array (x, y, z, qx, qy, qz, qw) in ROS xyzw
              - ``timestamp``: float timestamp in seconds

        """
        if self._closed:
            raise RuntimeError("RealsenseEnv is closed. Create a new instance to continue streaming.")

        frames = self._pipeline.wait_for_frames()
        frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to acquire both color and depth frames from RealSense.")

        # (H, W, 3) BGR uint8 -> RGB
        color_bgr = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # (H, W) uint16 depth image (no conversion to meters here).
        depth = np.asanyarray(depth_frame.get_data()).astype(np.uint16, copy=False)

        # Wall-clock timestamp in seconds.
        timestamp = float(time.time())

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsic_k": self._intrinsic_k,
            "camera_pose": self._camera_localizer.get_camera_pose(rgb=rgb, intrinsic_k=self._intrinsic_k),
            "timestamp": timestamp,
        }


class RealsenseCamera:
    """Realsense Camera class for streaming raw camera images."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        enable_ir: bool = True,
    ) -> None:
        """Initialize the Realsense Camera.

        Args:
            width: width of image
            height: height of image
            fps: frames per second to stream at
            enable_depth: if to stream depth
            enable_ir: if to enable IR

        """
        self._enable_depth = enable_depth
        self._enable_ir = enable_ir

        # Enable streams
        self.config = rs.config()
        self.colorizer = rs.colorizer()

        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        if enable_depth:
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        if enable_ir:
            self.config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
            self.config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)

        # Start pipeline
        self.pipeline = rs.pipeline()
        self._profile = self.pipeline.start(self.config)
        for _ in range(30):
            self.pipeline.wait_for_frames()

        # Get camera serial number
        device = self._profile.get_device()
        self.serial = device.get_info(rs.camera_info.serial_number)

        # Cache the intrinsics call
        self.intrinsics = self.get_intrinsics()

    def get_intrinsics(self) -> RealsenseIntrinsics:
        """Get the intrinsics of the Realsense camera.

        Returns:
            Realsense Camera intrinsics object.

        """
        # Color intrinsics
        color_profile = self._profile.get_stream(rs.stream.color)
        color_intr = color_profile.as_video_stream_profile().get_intrinsics()
        k_color = np.array(
            [
                [color_intr.fx, 0, color_intr.ppx],
                [0, color_intr.fy, color_intr.ppy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        distortion_color = np.array(color_intr.coeffs, dtype=np.float32)

        # IR intrinsics and extrinsics
        if not self._enable_ir:
            raise ValueError("IR streams must be enabled to get intrinsics")

        ir_left_profile = self._profile.get_stream(rs.stream.infrared, 1)
        ir_right_profile = self._profile.get_stream(rs.stream.infrared, 2)
        ir_intr = ir_left_profile.as_video_stream_profile().get_intrinsics()
        k_ir = np.array(
            [
                [ir_intr.fx, 0, ir_intr.ppx],
                [0, ir_intr.fy, ir_intr.ppy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Baseline between IR cameras
        extr = ir_left_profile.get_extrinsics_to(ir_right_profile)
        baseline = np.linalg.norm(extr.translation)

        # Extrinsics from IR1 to color
        extr_color = ir_left_profile.get_extrinsics_to(color_profile)
        t_color_from_ir = np.eye(4, dtype=np.float32)
        t_color_from_ir[:3, :3] = np.array(extr_color.rotation).reshape(3, 3).T
        t_color_from_ir[:3, 3] = np.array(extr_color.translation)

        return RealsenseIntrinsics(
            k_color=k_color,
            k_ir=k_ir,
            baseline_ir=baseline,
            t_color_from_ir=t_color_from_ir,
            distortion_color=distortion_color,
        )

    def read_camera(self) -> RealsenseFrame:
        """Read the camera frame.

        Returns:
            A Realsense Frame object with the current camera image information.

        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        rgb = np.asanyarray(color_frame.get_data())
        timestamp = frames.get_timestamp()

        # IR streams required for RealsenseFrame
        if not self._enable_ir:
            raise ValueError("IR streams must be enabled for RealsenseFrame")

        ir_left_frame = frames.get_infrared_frame(1)
        ir_right_frame = frames.get_infrared_frame(2)
        ir_left = np.asanyarray(ir_left_frame.get_data())
        ir_right = np.asanyarray(ir_right_frame.get_data())

        # Optional depth
        depth_float = None
        depth_raw = None
        if self._enable_depth:
            # Get raw depth
            depth_frame = frames.get_depth_frame()
            depth_raw = np.asanyarray(depth_frame.get_data())

            # Get aligned depth and convert mm to m
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth_float = (np.asanyarray(aligned_depth_frame.get_data()) / 1000.0).astype(np.float32)

        return RealsenseFrame(
            serial=self.serial,
            timestamp=timestamp,
            rgb=rgb,
            intrinsics=self.intrinsics.k_color,
            depth=depth_float,
            ir_left=ir_left,
            ir_right=ir_right,
            depth_raw=depth_raw,
        )

    def close(self) -> None:
        """Stop the camera pipeline."""
        self.pipeline.stop()


def quat_xyzw_to_R(qx, qy, qz, qw):
    # assumes unit quaternion
    x, y, z, w = qx, qy, qz, qw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def make_T(R, t_xyz):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(t_xyz, dtype=float)
    return T


def _inv_T(T):
    """Invert a 4x4 homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return make_T(R_inv, t_inv)


def T_to_xyz_quat_xyzw(T):
    xyz = T[:3, 3]
    quat = rot_to_quat_xyzw(T[:3, :3])
    return np.concatenate((xyz, quat), axis=0)


def rot_to_quat_xyzw(R):
    q = np.empty(4)
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s

    return q


if __name__ == "__main__":
    env = RealsenseEnv()

    while True:
        obs = env.get_observation()

        color = obs["rgb"].transpose((1, 2, 0))
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        depth = depth_to_colormap_np(obs["depth"][0])
        combined = np.concatenate([color, depth], axis=1)
        cv2.imshow("combined", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    env.close()
