import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from skillet.perception.reconstruction.reconstructor_base import ReconstructorBase
from skillet.scene.base import Scene


class VLMReconstructor(ReconstructorBase):
    """Parses observations for localizing objects depth and segmentation masks."""

    def __init__(self, scene: Scene | None = None) -> None:
        """Initialize the VLM Reconstructor.

        Args:
            scene: The scene to update with the estimated poses of the AprilTags.

        """
        super().__init__(scene)
        # self.seg_client = SegmentationClient()

        self._bboxes, self._masks, self._goal_atoms, self._scene_atoms = None, None, None, None

    def update_state(self, obs: dict[str, torch.Tensor], update: bool = True, seg_out: dict | None = None) -> None:
        """Update the state estimator with a new observation.

        Args:
            obs: The RGB-D observation to update the state estimator with
                Note: Contains tags: `rgb`, `depth`, `intrinsics`, `camera_pose`
            update: If to update the scene or not
            seg_out: The output from segmentation stored as a cache

        """
        if not update:
            return None
        if seg_out is None:
            seg_out = self.seg_client.segmentation(
                obs["rgb"], task_instruction="Move the red block onto the purple block."
            )
        self._bboxes = seg_out["bboxes"]
        self._masks = seg_out["masks"]
        self._goal_atoms = seg_out["grounded_goal_atoms"]
        self._scene_atoms = seg_out["grounded_scene_atoms"]

        self.compute_cube_centers(obs["depth"], self._masks)

        return seg_out

    def get_observation(self) -> Scene:
        """Return the scene."""
        return self._scene

    def compute_cube_centers(
        self,
        depth: np.ndarray,
        masks: np.ndarray,
        intrinsics: dict = {"fx": 615, "fy": 610, "cx": 321, "cy": 241},
        cam_pos: np.ndarray = np.ndarray([-0.7, 0.35, 0.3]),  # (3,)
        cam_quat: np.ndarray = None,  # (qx, qy, qz, qw)
        visualize: bool = True,
    ) -> np.ndarray:
        """Compute cube centers in WORLD frame."""
        masks = masks.squeeze(1)  # N x H x W
        centers_world = []

        # Build rotation matrix from quaternion
        cam_r = R.from_quat(cam_quat).as_matrix() if cam_quat is not None else None

        for i in range(masks.shape[0]):
            ys, xs = np.where(masks[i] > 0)
            if len(xs) == 0:
                centers_world.append(None)
                continue

            cube_depth = depth[ys, xs]

            # Filter invalid depth
            valid = cube_depth > 0
            if np.sum(valid) == 0:
                centers_world.append(None)
                continue

            xs = xs[valid]
            ys = ys[valid]
            cube_depth = cube_depth[valid]

            # ---- Camera frame ----
            X = (xs - intrinsics["cx"]) * cube_depth / intrinsics["fx"]
            Y = (ys - intrinsics["cy"]) * cube_depth / intrinsics["fy"]
            Z = cube_depth
            points_3d = np.stack([X, Y, Z], axis=-1)

            center_cam = points_3d.mean(axis=0)

            # ---- Transform to world ----
            center_world = cam_r @ center_cam + cam_pos if cam_r is not None and cam_pos is not None else None

            centers_world.append(center_world)

        print("World frame centers:", centers_world)

        # Visualization (still image plane)
        if visualize:
            depth_vis = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
            plt.figure(figsize=(8, 6))
            plt.imshow(depth_vis, cmap="viridis")
            plt.title("Cube Centers Overlay")
            plt.axis("off")

            for i, center_world in enumerate(centers_world):
                if center_world is not None:
                    # Recompute projection using camera-frame version
                    # (optional: store center_cam if you want cleaner separation)
                    pass

            plt.show()

        return centers_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ti", type=str, default="Move the red block onto the purple block")
    parser.add_argument("--dir", type=str, default="captures/capture_20260402_083413/")
    parser.add_argument(
        "--new",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to redo segmentation on an image or load from .pkl",
    )
    args = parser.parse_args()

    rgb = np.load(f"{args.dir}/np_color.npy")
    depth = np.load(f"{args.dir}/np_depth.npy")

    vlm = VLMReconstructor()

    if args.new:
        out = vlm.update_state({"rgb": rgb, "depth": depth})
        with pathlib.Path(f"{args.dir}/out.pkl").open("wb") as f:
            pickle.dump(out, f)
    else:
        with pathlib.Path(f"{args.dir}/out.pkl").open("rb") as f:
            out = pickle.load(f)

        vlm.update_state({"rgb": rgb, "depth": depth}, seg_out=out)


if __name__ == "__main__":
    main()
