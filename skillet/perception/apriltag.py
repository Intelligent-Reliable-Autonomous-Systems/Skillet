from collections import defaultdict
from typing import Literal

import cv2
import numpy as np
import pupil_apriltags as apriltags
import torch

from skillet.core.math import quat_apply, quat_from_matrix
from skillet.scene.base import Scene
from skillet.scene.cube import Cube


class ApriltagStateEstimator:
    """Parses observations for localizing objects using AprilTags."""

    def __init__(self, scene: Scene) -> None:
        """Initialize the AprilTag state estimator.

        Args:
            scene: The scene to update with the estimated poses of the AprilTags.

        """
        self._scene = scene
        self._detector = apriltags.Detector()

    def update_state(self, obs: dict[str, torch.Tensor]) -> None:
        """Update the state estimator with a new observation.

        Args:
            obs: The RGB-D observation to update the state estimator with.

        """
        device = obs["rgb"].device
        rgb = obs["rgb"].cpu().numpy().transpose(1, 2, 0)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        intrinsic_k = obs["intrinsic_k"]
        camera_params = (intrinsic_k[0, 0], intrinsic_k[1, 1], intrinsic_k[0, 2], intrinsic_k[1, 2])
        cam_pos = obs["camera_pose"][:3]
        cam_quat = obs["camera_pose"][3:7]

        id_sizes: dict[float, list[int]] = defaultdict(list)
        id_to_cube: dict[int, Cube] = {}
        id_to_cube_spec: dict[int, dict] = {}
        for obj in self._scene.objects:
            if isinstance(obj, Cube):
                for apriltag in obj.get_face_apriltags():
                    id_sizes[apriltag["size"]].append(apriltag["id"])
                    id_to_cube[apriltag["id"]] = obj
                    id_to_cube_spec[apriltag["id"]] = apriltag

        for size, ids in id_sizes.items():
            detections = self._detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=size)
            if detections:
                for detection in detections:
                    if detection.tag_id in ids and detection.tag_id in id_to_cube_spec:
                        pos = torch.from_numpy(detection.pose_t.reshape(3)).to(device, dtype=torch.float32)
                        rot = torch.from_numpy(detection.pose_R).to(device, dtype=torch.float32)
                        normal = -rot[:, 2]  # +Z of tag = inward normal (away from camera), so negate
                        up = -rot[:, 1]  # +Y of tag = down, so negate for up

                        # Transform position: world_pos = cam_pos + R_world_cam @ tag_pos_in_cam
                        pos_world = cam_pos + quat_apply(cam_quat, pos)

                        # Transform directions: just rotate, no translation
                        normal_world = quat_apply(cam_quat, normal)
                        up_world = quat_apply(cam_quat, up)

                        spec = id_to_cube_spec[detection.tag_id]
                        cube: Cube = id_to_cube[detection.tag_id]
                        cube_pose = Cube.pose_from_face_center(spec["face"], pos_world, normal_world, up_world, cube.size)
                        cube.pose = cube_pose


class CameraLocalizer:

    def __init__(
        self,
        apriltag_pose: np.ndarray = np.array([0.13, 0.0, 0.0, 0.0, 0.0, 0.7071068, 0.7071068]),
        apriltag_size_m: float = 0.100,
        apriltag_id: int = 0,
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
        self._T_base_to_tag = make_T(quat_xyzw_to_R(*list(apriltag_pose[3:7])), list(apriltag_pose[:3]))
        self._roll_180 = make_T(quat_xyzw_to_R(1.0, 0.0, 0.0, 0.0), [0.0, 0.0, 0.0])
        self._T_base_to_tag = self._T_base_to_tag @ self._roll_180
        # self._latest_camera_pose = np.array([0.33, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64) # in ROS xyzw format
        self._latest_camera_pose = T_to_xyz_quat_xyzw(self._T_base_to_tag)

    def get_camera_pose(self, rgb: np.ndarray, intrinsic_k: np.ndarray) -> np.ndarray:
        """Get the camera pose from the ROS observation.

        Args:
            - rgb: (H, W, 3) uint8 RGB image
            - depth: (H, W) uint16 depth image
            - intrinsic_k: (3, 3) float64 camera intrinsic matrix

        Returns:
            The camera pose in the world frame.
        """
        camera_params = (intrinsic_k[0, 0], intrinsic_k[1, 1], intrinsic_k[0, 2], intrinsic_k[1, 2])
        tag_size_m = self._apriltag_size_m
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        detections = self._detector.detect(
            gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size_m
        )
        if detections:
            for detection in detections:
                if detection.tag_id == self._apriltag_id:
                    T_tag_cam = make_T(detection.pose_R, detection.pose_t.reshape(3))
                    T_cam_tag = _inv_T(T_tag_cam)
                    T_base_cam = self._T_base_to_tag @ T_cam_tag
                    self._latest_camera_pose = T_to_xyz_quat_xyzw(T_base_cam)
                    break

        return self._latest_camera_pose


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
