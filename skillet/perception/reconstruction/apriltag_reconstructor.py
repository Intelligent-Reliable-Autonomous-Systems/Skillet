from collections import defaultdict

import cv2
import pupil_apriltags as apriltags
import torch

from skillet.core.math import quat_apply
from skillet.perception.reconstruction.reconstructor_base import ReconstructorBase
from skillet.scene.base import Scene
from skillet.scene.cube import Cube


class ApriltagStateReconstructor(ReconstructorBase):
    """Parses observations for reconstructing the scene using AprilTags."""

    def __init__(self, scene: Scene | None = None) -> None:
        """Initialize the AprilTag state estimator.

        Args:
            scene: The scene to update with the estimated poses of the AprilTags.

        """
        super().__init__(scene)
        self._detector = apriltags.Detector(
            families="tag36h11",  # or whatever family you're printing
            nthreads=4,
            quad_decimate=1.0,  # Don't downsample — critical for small tags
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def update_state(self, obs: dict[str, torch.Tensor], update: bool = True) -> None:
        """Update the state estimator with a new observation.

        Args:
            obs: The RGB-D observation to update the state estimator with.
            update: If to update the state or not

        """
        if not update:
            return
        device = obs["rgb"].device
        rgb = obs["rgb"].cpu().numpy().transpose(1, 2, 0)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        intrinsic_k = obs["intrinsic_k"]
        camera_params = (intrinsic_k[0, 0], intrinsic_k[1, 1], intrinsic_k[0, 2], intrinsic_k[1, 2])
        cam_pos = obs["camera_pose"][:3]
        cam_quat = obs["camera_pose"][3:7]
        cam_quat = cam_quat / cam_quat.norm()

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
                        cube_pose = Cube.pose_from_face_center(
                            spec["face"], pos_world, normal_world, up_world, cube.size
                        )
                        cube.pose = cube_pose

    def get_observation(self) -> Scene:
        """Return the scene."""
        return self._scene
