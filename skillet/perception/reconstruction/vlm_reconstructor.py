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
