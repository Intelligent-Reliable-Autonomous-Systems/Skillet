"""RealSense RGB-D environment.

Provides a simple environment that streams RGB-D observations from an Intel
RealSense camera in a format compatible with the ROS2 RGB-D pipeline and
`Ros2WebEnvWrapper`'s `"rgb-d"` observation.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, override

import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector as AprilTagDetector

from skillet.core.env import _EnvironmentBase
from skillet.envs.specs import RGBD_SPEC_BATCHED, RGBD_Obs
from skillet.perception.localization import CameraLocalizer
from skillet.perception.localization.camera_localizer import _make_T, _quat_xyzw_to_R, _T_to_xyz_quat_xyzw
from skillet.scene.utils import depth_to_colormap_np

if TYPE_CHECKING:
    from skillet.core import ActionSpec
    from skillet.core.spaces import ObservationSpec

DEFAULT_APRILTAG_POSE = np.array([0.13, 0.000, 0.0, 0.0, 0.0, 0.7071068, 0.7071068])
DEFAULT_APRILTAG_SIZE_M = 0.100
DEFAULT_APRILTAG_ID = 1


class RealsenseEnv(_EnvironmentBase[RGBD_Obs, Any]):
    """Environment that streams RGB-D observations from an Intel RealSense camera.

    The raw RGB-D snapshot matches `_get_latest_rgbd` in
    `gen3_ros2.py` (RGB HxWx3, depth HxW, intrinsics, pose, timestamp).
    The public observation returned by `get_observation()` also mirrors the
    post-processing done in `Ros2WebEnvWrapper` for the `"rgb-d"` observation:

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
        apriltag_pose: np.ndarray = DEFAULT_APRILTAG_POSE,
        apriltag_size_m: float = DEFAULT_APRILTAG_SIZE_M,
        apriltag_id: int = DEFAULT_APRILTAG_ID,
    ) -> None:
        """Initialize the RealSense pipeline and RGB-D observation space."""
        self.width = width
        self.height = height
        self.fps = fps

        # RealSense pipeline + streams.
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self._config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self._tag_detector = AprilTagDetector()
        self._T_base_to_tag = _make_T(_quat_xyzw_to_R(*list(apriltag_pose[3:7])), list(apriltag_pose[:3]))
        self._roll_180 = _make_T(_quat_xyzw_to_R(1.0, 0.0, 0.0, 0.0), [0.0, 0.0, 0.0])
        self._T_base_to_tag = self._T_base_to_tag @ self._roll_180
        self._latest_camera_pose = _T_to_xyz_quat_xyzw(self._T_base_to_tag)
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
        # in `Ros2WebEnvWrapper` but without batching.
        self.obs_spec_rgbd = (
            RGBD_SPEC_BATCHED.bind(height=self.height, width=self.width).replace(is_torch=False).unbatched()
        )

        self._closed = False

    @property
    def obs_spec(self) -> ObservationSpec[RGBD_Obs]:
        """Return the default (unbatched) RGB-D observation specification."""
        return self.obs_spec_rgbd

    def supports_observation_spec(self, obs_spec: ObservationSpec[RGBD_Obs]) -> bool:
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
              - ``camera_pose``: 7D float64 array (x, y, z, qw, qx, qy, qz) in Isaac wxyz
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

    def get_observation(
        self,
        obs_spec: ObservationSpec[RGBD_Obs] | None = None,
    ) -> RGBD_Obs:
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

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[RGBD_Obs, dict[str, Any]]:
        return self.get_observation(), {}

    @override
    def step(self, action: Any) -> tuple[RGBD_Obs, float, bool, bool, dict[str, Any]]:
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
