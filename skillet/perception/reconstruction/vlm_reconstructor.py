from typing import Any, Literal

import cv2
import numpy as np
import torch

from skillet.envs.realsense import RealsenseEnv
from skillet.perception.reconstruction.reconstructor_base import ReconstructorBase
from skillet.perception.segmentation.sam import get_sam_client
from skillet.perception.segmentation.vlm import GeminiClient, QwenClient
from skillet.scene.base import Scene


class VlmReconstructor(ReconstructorBase):
    """Parses observations for localizing objects depth and segmentation masks."""

    def __init__(
        self,
        scene: Scene | None = None,
        model: Literal["sam2", "sam3", "sam3_streaming"] = "sam2",
        vlm_model: Literal["gemini", "qwen"] = "qwen",
        mode: Literal["text", "bboxes"] = "bboxes",
        device: str = "cuda",
        visualize: bool = True,
    ) -> None:
        """Initialize the VLM Reconstructor.

        Args:
            scene: The scene to update with the estimated poses of the AprilTags.

        """
        super().__init__(scene, device=device)
        self._model = model
        self._mode = mode
        self._sam_model = get_sam_client(model)(use_server=False)
        self._vlm_client = GeminiClient() if vlm_model == "gemini" else QwenClient()
        self._visualize = visualize

        self._bboxes, self._masks, self._goal_atoms, self._scene_atoms = None, None, None, None

    @property
    def masks(self) -> torch.Tensor:
        return self._masks

    @property
    def segment_indices(self) -> torch.Tensor:
        return self._segment_indices

    def update_state(
        self, obs: dict[str, Any], update: bool = True, frame: Literal["world", "camera"] = "camera"
    ) -> None:
        """Update the state of the scene by finding cube centers.

        Args:
            obs: RGB-D obs spec from the environment
            update: If to update the state of the scene or not
            frame: the frame to perform the scene update from

        """
        if not update:
            return
        rgb = obs["rgb"]
        bboxes, labels = self._vlm_client.detect_bboxes(rgb, self._vlm_client.prompt)

        if self._mode == "text":
            concepts = ["block", "robot_arm"]
            masks, _, _, concept_indices = self._sam_model.segment_concepts(rgb, concepts)
        elif self._mode == "bboxes":
            for box in bboxes:
                box[0] = (box[0] / 1000) * rgb.shape[2]
                box[2] = (box[2] / 1000) * rgb.shape[2]
                box[1] = (box[1] / 1000) * rgb.shape[1]
                box[3] = (box[3] / 1000) * rgb.shape[1]
            masks, _ = self._sam_model.segment_bboxes(rgb, bboxes)
            concept_indices = np.arange(len(labels))
            concepts = labels
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

        self._masks = masks
        self._segment_indices = torch.arange(masks.shape[0], device=masks.device)

        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
            # depth = depth.cpu().numpy()
            # intrinsic_k = intrinsic_k.cpu().numpy()
            # camera_pose = camera_pose.cpu().numpy()

        # Grab only the cubes
        masks = masks.cpu().numpy()

        if self._visualize:
            self._bbox_frame = VlmReconstructor.show_bounding_boxes(
                rgb, masks, concept_indices=concept_indices, concepts=concepts
            )

    @staticmethod
    def show_bounding_boxes(
        rgb_image: np.ndarray,
        masks: np.ndarray,
        concept_indices: np.ndarray | None = None,
        concepts: list | None = None,
    ) -> np.ndarray:
        """Draw bounding boxes and semi-transparent mask overlays on an RGB image.

        Args:
            rgb_image: HxWx3 numpy array in RGB format.
            masks:     NxHxW binary (or boolean) numpy array from SAM.
            concept_indices: mask indices of each concept
            concepts: list of concepts from segmentation

        Returns:
            BGR image ready for cv2.imshow.

        """
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.40
        THICKNESS = 1
        PADDING = 4
        # cv2 works in BGR
        rgb_image = rgb_image.transpose((1, 2, 0))
        display = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR).copy()
        overlay = display.copy()

        # Generate a distinct colour per mask
        rng = np.random.default_rng(seed=42)
        colors = [tuple(int(c) for c in rng.integers(80, 230, size=3)) for _ in range(len(masks))]

        for i, mask in enumerate(masks):
            # Semi-transparent fill
            overlay[mask > 0] = colors[i]

            # Bounding box
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            cv2.rectangle(display, (x1, y1), (x2, y2), colors[i], thickness=2)

            # Concept label
            label = f"{concepts[concept_indices[i]]} {i}" if concept_indices is not None else f"Object {i}"

            (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)

            pill_x1 = x1
            pill_y1 = max(0, y1 - text_h - baseline - PADDING * 2)
            pill_x2 = x1 + text_w + PADDING * 2
            pill_y2 = max(text_h + baseline + PADDING * 2, y1)

            cv2.rectangle(display, (pill_x1, pill_y1), (pill_x2, pill_y2), colors[i], cv2.FILLED)

            text_x = pill_x1 + PADDING
            text_y = pill_y2 - PADDING - baseline
            cv2.putText(display, label, (text_x, text_y), FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA)

        # Blend fill at 35 % opacity
        cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)
        return display

    def get_observation(self, obs_spec):
        pass


def main():
    env = RealsenseEnv()
    vlm = VlmReconstructor()
    cv2.namedWindow("VLM Scene", cv2.WINDOW_NORMAL)

    while True:
        obs = env.get_observation()
        vlm.update_state(obs)
        cv2.imshow("VLM Scene", vlm._bbox_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
