"""Segmentation client for computing the depth and candidate grasps."""

import numpy as np
from jaxtyping import UInt8
from PIL import Image

from skillet.perception.segmentation.sam import SAM2Client
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

        masks = self.sam_client.segment_from_bboxes(rgb_pil, bboxes)

        return {
            "bboxes": bboxes,
            "masks": masks,
            "grounded_goal_atoms": grounded_goal_atoms,
            "grounded_scene_atoms": grounded_scene_atoms,
        }
