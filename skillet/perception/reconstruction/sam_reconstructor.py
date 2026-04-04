"""sam_reconstructor.py.

Reconstruct the scene from SAM bounding boxes.
"""

import time
from typing import Any, Literal

import numpy as np
import torch

from skillet.perception.realsense import RealsenseEnv
from skillet.perception.reconstruction.reconstructor_base import ReconstructorBase
from skillet.perception.reconstruction.utils import (
    filter_cube_centers,
    find_cube_centers,
    transform_cube_centers_to_world,
)
from skillet.perception.segmentation.sam import get_sam_client
from skillet.scene.base import Scene
from skillet.scene.cube import Cube


class SAMReconstructor(ReconstructorBase):
    """Main class for reconstruction with SAM Client.

    Finds the bounding boxes of the cubes, segments point cloud, and projects normal
    to find the center of the cube.

    """

    def __init__(
        self,
        scene: Scene | None = None,
        model: Literal["sam2", "sam3", "sam3_streaming"] = "sam3",
        mode: Literal["text", "bboxes"] = "text",
        device: str = "cuda",
    ) -> None:
        super().__init__(scene)
        self._model = model
        self._mode = mode
        self._sam_model = get_sam_client(model)
        self._device = device

    def update_state(self, obs: dict[str, Any], update: bool = True) -> None:
        """Update the state of the scene by finding cube centers.

        Args:
            obs: RGB-D obs spec from the environment
            update: If to update the state of the scene or not

        """
        if not update:
            return
        rgb = obs["rgb"]
        depth = obs["depth"]
        intrinsic_k = obs["intrinsic_k"]
        camera_pose = obs["camera_pose"]

        if self._mode == "text":
            concepts = ["block", "robot arm"]
            # TODO: sometimes SAM segments the tops of the cubes as well
            # This sometimes gets the apriltag as well
            # Want to filter this based on cubes being close to each other
            masks, boxes, scores, concept_indices = self._sam_model.segment_from_concepts(rgb, concepts)
        elif self._mode == "bboxes":
            # TODO: Get this from VLM?
            # If so, get in format 0-1000 regardless of image scale
            bboxes = [
                [100, 100, 150, 150],
                [200, 200, 250, 250],
                [300, 300, 350, 350],
                [200, 100, 250, 150],
                [100, 200, 150, 250],
            ]
            masks, scores = self._sam_model.segment_from_bboxes(rgb, bboxes)
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

        # Might want to filter
        dc = find_cube_centers(masks.cpu().numpy(), depth, intrinsic_k, cube_size=0.036)
        centers = transform_cube_centers_to_world(
            dc["centers"], camera_pos=camera_pose[0:3], camera_quat=camera_pose[3:7]
        )
        centers = filter_cube_centers(centers, max_cubes=4)  # TODO: change according to scene
        for i, c in enumerate(centers):
            if c[0] < 0.1:  # This is the large apriltag
                continue
            self._scene.objects[i].pose = torch.as_tensor(c, device=self._device)

        np.set_printoptions(suppress=True, precision=3)
        for c in centers:
            if c[0] < 0.1:  # This is the large apriltag
                continue
            print(c)
        print()

    def get_observation(self) -> Scene:
        """Return the scene."""
        return self._scene


def main() -> None:
    """Run live SAM benchmark with RGB/depth view and mask overlay."""
    TABLE_X0 = -0.0889
    TABLE_Y0 = -0.577
    TABLE_DX = 0.762
    TABLE_DY = 1.2446

    """Visualize RGB + depth color map from _get_latest_rgbd()."""
    cube_0 = Cube(size=0.041, face_apriltags=[{"face": "top", "size": 0.036, "id": 1}])
    cube_1 = Cube(size=0.041, face_apriltags=[{"face": "front", "size": 0.036, "id": 2}])
    cube_2 = Cube(size=0.041, face_apriltags=[{"face": "front", "size": 0.036, "id": 5}])

    world_bounds = (TABLE_X0, TABLE_Y0, 0, TABLE_X0 + TABLE_DX, TABLE_Y0 + TABLE_DY, 1)
    scene = Scene(objects=[cube_0, cube_1, cube_2], closed_set=True, bounds=world_bounds)

    env = RealsenseEnv()
    reconstructor = SAMReconstructor(scene)
    while True:
        rgbd_obs = env.get_observation()
        reconstructor.update_state(rgbd_obs)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
