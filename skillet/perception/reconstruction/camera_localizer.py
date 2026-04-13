"""File for handling camera localization."""

import time
from typing import Any

import cv2
import numpy as np
import pupil_apriltags as apriltags
import pyrealsense2 as rs

DEFAULT_APRILTAG_POSE = np.array([0.12, 0.005, 0.0, 0.0, 0.0, 0.7071068, 0.7071068])
DEFAULT_APRILTAG_SIZE_M = 0.100
DEFAULT_APRILTAG_ID = 3


class CameraLocalizer:
    """Localizer for the camera based on the large table apriltag."""

    def __init__(
        self,
        apriltag_pose: np.ndarray = np.array([0.12, 0.005, 0.0, 0.0, 0.0, 0.7071068, 0.7071068]),
        apriltag_size_m: float = 0.100,
        apriltag_id: int = 3,
    ) -> None:
        """Initialize the camera localizer.

        Args:
            apriltag_pose: The pose of the AprilTag in the world frame.
            apriltag_size_m: The size of the AprilTag in meters.
            apriltag_id: The ID of the AprilTag.

        """
        self._apriltag_pose = apriltag_pose
        self._apriltag_size_m = apriltag_size_m
        self._apriltag_id = apriltag_id

        self._detector = apriltags.Detector()
        self._T_base_to_tag = _make_T(_quat_xyzw_to_R(*list(apriltag_pose[3:7])), list(apriltag_pose[:3]))
        self._roll_180 = _make_T(_quat_xyzw_to_R(1.0, 0.0, 0.0, 0.0), [0.0, 0.0, 0.0])
        self._T_base_to_tag = self._T_base_to_tag @ self._roll_180
        self._latest_camera_pose = _T_to_xyz_quat_xyzw(self._T_base_to_tag)

    def get_camera_pose(self, rgb: np.ndarray, intrinsic_k: np.ndarray) -> np.ndarray:
        """Get the camera pose from the ROS observation.

        Args:
            rgb: (H, W, 3) uint8 RGB image
            intrinsic_k: (3, 3) float64 camera intrinsic matrix

        Returns:
            The camera pose in the world frame.

        """
        # TODO: Make this rolling or something so that we filter out noise
        # TOOD: check
        camera_params = (intrinsic_k[0, 0], intrinsic_k[1, 1], intrinsic_k[0, 2], intrinsic_k[1, 2])
        tag_size_m = self._apriltag_size_m
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        detections = self._detector.detect(
            gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size_m
        )
        if detections:
            for detection in detections:
                if detection.tag_id == self._apriltag_id:
                    t_tag_cam = _make_T(detection.pose_R, detection.pose_t.reshape(3))
                    t_cam_tag = _inv_T(t_tag_cam)
                    t_base_cam = self._T_base_to_tag @ t_cam_tag
                    self._latest_camera_pose = _T_to_xyz_quat_xyzw(t_base_cam)
                    break

        return self._latest_camera_pose


class RealsenseCameraLocalizer:
    """Class that manages the Realsense camera streaming with AprilTag estimation.

    Used as a helper class in the Kortex API environments to avoid requiring ROS2.
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
        self._config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self._config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self._tag_detector = apriltags.Detector()
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


def _quat_xyzw_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:  # noqa: N802
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


def _make_T(R: np.ndarray, t_xyz: np.ndarray) -> np.ndarray:  # noqa: N802, N803
    T = np.eye(4)  # noqa: N806
    T[:3, :3] = R
    T[:3, 3] = np.array(t_xyz, dtype=float)
    return T


def _inv_T(T: np.ndarray) -> np.ndarray:  # noqa: N802, N803
    """Invert a 4x4 homogeneous transform."""
    R = T[:3, :3]  # noqa: N806
    t = T[:3, 3]
    R_inv = R.T  # noqa: N806
    t_inv = -R_inv @ t
    return _make_T(R_inv, t_inv)


def _T_to_xyz_quat_xyzw(T: np.ndarray) -> np.ndarray:  # noqa: N802, N803
    xyz = T[:3, 3]
    quat = _rot_to_quat_xyzw(T[:3, :3])
    return np.concatenate((xyz, quat), axis=0)


def _rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:  # noqa: N803
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
