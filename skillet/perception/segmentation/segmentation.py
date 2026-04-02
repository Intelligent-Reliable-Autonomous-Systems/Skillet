import asyncio

import aiohttp
import numpy as np
from jaxtyping import Bool, Float, UInt8
from PIL import Image

from skillet.perception.realsense import Frame
from skillet.perception.segmentation.sam import SAM2Client
from skillet.perception.segmentation.utils import depth_to_xyz, get_o3d_pcd
from skillet.perception.segmentation.vlm import GeminiClient


class SegmentationClient:
    """Client for computing the depth and candidate grasps."""

    def __init__(
        self,
    ) -> None:
        self.sam_client = SAM2Client()
        self.vlm_client = GeminiClient()

    def run_perception(self) -> dict:
        """Run the full perception pipeline."""
        raise NotImplementedError

    def segmentation(self, rgb: UInt8[np.ndarray, "h w 3"], task_instruction: str) -> dict:
        """Test the segmentation and task instruction with the Gemini and SAM2 pipline.

        Args:
            rgb: RGB image to segment
            task_instruction: Instruction of the task to complete.

        Returns:
            dictionary of bounding boxes, segmentation masks, goal predicates, and scene predicates

        """
        rgb_pil = Image.fromarray(rgb)
        rgb_pil_resized = rgb_pil.resize((800, int(800 * rgb_pil.size[1] / rgb_pil.size[0])), Image.Resampling.LANCZOS)
        bboxes, grounded_goal_atoms, grounded_scene_atoms = self.vlm_client.detect_and_translate(
            rgb_pil_resized, task_instruction
        )

        for bbox in bboxes:
            bbox["label"] = bbox["label"].replace(" ", "_")
        for atom in grounded_goal_atoms:
            atom["args"] = [arg.replace(" ", "_") for arg in atom["args"]]
        for atom in grounded_scene_atoms:
            atom["args"] = [arg.replace(" ", "_") for arg in atom["args"]]

        masks = self.sam_client.segment_objects(rgb_pil, bboxes)

        return {
            "bboxes": bboxes,
            "masks": masks,
            "grounded_goal_atoms": grounded_goal_atoms,
            "grounded_scene_atoms": grounded_scene_atoms,
        }

    async def _predict_depth_and_grasps(
        self,
        session: aiohttp.ClientSession,
        frame: Frame,
        world_from_cam: Float[np.ndarray, "4 4"],
        downsample_voxel_size: float,
        gripper_mask: Bool[np.ndarray, "h w 3"] | None = None,
    ) -> dict:
        """Predict depth map using FoundationStereo and grasps using M2T2.

        Uses depth_estimator if provided, otherwise uses frame.depth.
        """
        # Get depth map — use estimator (e.g. FoundationStereo) or fall back to onboard sensor depth
        if self.depth_client is not None:
            depth_map = await self.depth_client.rs_infer_depth_async(session, frame, self.camera.get_intrinsics())
        else:
            if frame.depth is None:
                raise RuntimeError(
                    "No depth available: depth_estimator is None and frame.depth is not set. "
                    "Either provide a depth_estimator or ensure the camera captures hardware depth."
                )
            print("[WARN] No depth_estimator provided, falling back to hardware depth")
            depth_map = frame.depth

        # Convert to point cloud in world frame
        K = frame.intrinsics
        xyz_map = depth_to_xyz(depth_map, K)
        xyz_map = xyz_map @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]
        if gripper_mask is not None:
            xyz_map[gripper_mask] = 0.0
        rgb_map = frame.rgb.astype(np.float32) / 255.0  # make it float with [0, 1]

        # Create open3d point cloud and downsample
        pcd = await asyncio.to_thread(
            get_o3d_pcd,
            xyz_map,
            rgb_map,
            downsample_voxel_size,
        )
        xyz_downsampled = np.asarray(pcd.points)
        rgb_downsampled = np.asarray(pcd.colors)

        # Predict grasps using M2T2
        grasps = await self.grasp_client.generate_grasps_async(
            session, scene_xyz=xyz_downsampled, scene_rgb=rgb_downsampled
        )

        return {
            "depth_map": depth_map,
            # (h, w, 3) for xyz, rgb, and valid mask map
            "xyz_map": xyz_map,
            "rgb_map": rgb_map,
            # (n, 3) for downsampled point cloud
            "xyz_downsampled": xyz_downsampled,
            "rgb_downsampled": rgb_downsampled,
            "pcd_downsampled": pcd,
            # grasps
            "grasps": grasps,
        }
