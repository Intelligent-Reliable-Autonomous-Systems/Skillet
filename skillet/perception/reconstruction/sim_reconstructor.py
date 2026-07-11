"""sam_reconstructor.py.

Reconstruct the scene from SAM3 concepts and bounding boxes.
"""

from typing import Any

import cv2
import numpy as np
import torch

from skillet.perception.reconstruction.reconstructor_base import ReconstructorBase
from skillet.perception.reconstruction.utils import (
    find_block_centers_mean,
    transform_xyz_to_world,
)
from skillet.scene import CUBE_SIZE, SPILL_SIZE, SPONGE_SIZE, TARGET_SIZE
from skillet.scene.base import Scene


class SimReconstructor(ReconstructorBase):
    """Main class for reconstruction with SAM Client.

    Finds the bounding boxes of the cubes, segments point cloud, and projects normal
    to find the center of the cube.

    """

    def __init__(
        self,
        scene: Scene | None = None,
        device: str = "cuda",
        visualize: bool = True,
    ) -> None:
        """Initialize the SAM reconstructor."""
        super().__init__(scene, device=device)
        self._visualize = visualize

        self._masks = None
        self._segment_indices = None

        # Scene reconstruction
        self._vlm_bboxes = None
        self._vlm_goal_atoms = None

    @property
    def masks(self) -> torch.Tensor:
        return self._masks

    @property
    def segment_indices(self) -> torch.Tensor:
        return self._segment_indices

    def update_state(
        self,
        obs: dict[str, Any],
        update: bool = True,
    ) -> None:
        """Update the state of the scene by finding cube centers.

        Args:
            obs: RGB-D obs spec from the environment
            update: If to update the state of the scene or not

        """
        if not update:
            return
        obj_names = obs["obj_names"]
        obj_poses = obs["obj_poses"]
        for i, n in enumerate(obs["obj_names"]):
            obj = self._scene.get_objects_from_name([n])[0]
            obj.pose = obj_poses[i]

    def _get_object_centers(
        self,
        obj_masks: torch.Tensor,
        obj_types: list,
        depth: torch.Tensor,
        intrinsic_k: torch.Tensor,
        camera_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Find object centers and orientation in the camera frame
        cube_inds = np.argwhere(obj_types == "block")
        target_inds = np.argwhere(obj_types == "target")
        sponge_inds = np.argwhere(obj_types == "sponge")
        spill_inds = np.argwhere(obj_types == "spill")
        obj_inds = np.concatenate((cube_inds, target_inds, sponge_inds, spill_inds)).flatten()
        obj_sizes = np.zeros(obj_inds.shape[0])
        obj_sizes[cube_inds] = CUBE_SIZE
        obj_sizes[target_inds] = TARGET_SIZE
        obj_sizes[sponge_inds] = SPONGE_SIZE
        obj_sizes[sponge_inds] = SPILL_SIZE

        centers = torch.zeros(obj_masks.shape[0], 3, device=obj_masks.device)
        bboxes = torch.zeros(obj_masks.shape[0], 6, device=obj_masks.device)

        # Localize the cubes, targets, and sponge
        obj_centers, obj_bboxes = find_block_centers_mean(
            obj_masks[obj_inds],
            depth,
            intrinsic_k,
            obj_size=obj_sizes,
            camera_pos=camera_pose[0:3],
            camera_quat=camera_pose[3:7],
        )
        obj_centers = transform_xyz_to_world(obj_centers, camera_pos=camera_pose[0:3], camera_quat=camera_pose[3:7])

        obj_bboxes[:, 0:3] = transform_xyz_to_world(
            obj_bboxes[:, 0:3], camera_pos=camera_pose[0:3], camera_quat=camera_pose[3:7]
        )
        obj_bboxes[:, 3:6] = transform_xyz_to_world(
            obj_bboxes[:, 3:6], camera_pos=camera_pose[0:3], camera_quat=camera_pose[3:7]
        )

        centers[obj_inds] = obj_centers
        bboxes[obj_inds] = obj_bboxes

        return centers, bboxes

    def get_observation(self) -> Scene:
        """Return the scene."""
        return self._scene

    def _build_scene(
        self,
        obs: dict[str, torch.Tensor],
        call_vlm: bool = True,
    ) -> None:
        """Build the scene using an API call to a VLM by creating bounding boxes for each object.

        Args:
            obs: RGBD obs spec observation
            call_vlm: If to call VLM or load scene from defaults

        """
        if self._task_instruction is None:
            self._task_instruction = "Put the dark red block on the purple block."

        if call_vlm:
            self._vlm_goal_atoms = self._vlm_client.detect_goal(self._task_instruction, self._scene.abstract_scene)
            for atom in self._vlm_goal_atoms:
                atom["args"] = [arg.replace(" ", "_") for arg in atom["args"]]

        self._scene.goal = self._vlm_goal_atoms

        print("[INFO] Reconstructed Goal with VLM")

    @staticmethod
    def show_vlm_image_and_masks(
        image: np.ndarray,
        masks: np.ndarray,
        labels: list[str],
    ) -> np.ndarray:
        """Show a single overlay image with all masks + labels.

        Args:
            image: (H, W, 3) RGB image, uint8 or float
            masks: (N, H, W) boolean or {0,1} masks
            labels: list of length N containing labels for each mask

        """
        num_masks = masks.shape[0]
        if len(labels) != num_masks:
            raise ValueError(f"Expected {num_masks} labels, got {len(labels)}")

        # Normalize to uint8 BGR
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        rng = np.random.default_rng(0)
        colors = rng.integers(64, 255, size=(num_masks, 3)).tolist()
        alpha = 0.5

        for i in range(num_masks):
            mask = masks[i].astype(bool)
            if mask.sum() == 0:
                continue

            color = colors[i]

            # Blend color into masked region
            colored = overlay.copy()
            colored[mask] = color
            overlay = cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)

            # Label at centroid
            ys, xs = np.where(mask)
            cx = int(np.mean(xs))
            cy = max(int(np.mean(ys)) - 10, 0)

            (tw, th), baseline = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (cx - tw // 2 - 2, cy - th - 4), (cx + tw // 2 + 2, cy + baseline), (0, 0, 0), -1)
            cv2.putText(
                overlay, labels[i], (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

        return overlay

    @staticmethod
    def masks_to_bboxes(masks: torch.Tensor) -> list[tuple[int, int, int, int]]:
        """Convert binary masks of shape (N, H, W) to bounding boxes (x1, y1, x2, y2)."""
        bboxes = []
        for mask in masks:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            bboxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        return bboxes

    @staticmethod
    def show_obj_masks(
        rgb_image: np.ndarray,
        masks: np.ndarray,
        scene: Scene,
        ids: list,
        colors: list[tuple[int, int, int]],
    ) -> np.ndarray:
        """Show the masks and the corresponding labels.

        Args:
            rgb_image: RGB image from camera
            masks: masks produced by SAM
            scene: the current scene to obtain
            ids: np.ndarray of sorted object ids
            colors: color array

        """
        rgb_image = rgb_image.transpose((1, 2, 0))
        display = rgb_image.copy()

        for color_idx, ob in enumerate(scene.objects):
            if not ob.localizable:
                continue

            idx = np.argwhere(ob.object_id == ids)
            if len(idx) != 0:
                idx = idx[0]
            else:
                continue
            if idx.size > 0:
                idx = idx[0]
            else:
                continue

            mask = masks[idx]  # shape (H, W), bool or 0/1
            color = colors[color_idx]

            # Overlay colored mask with transparency
            overlay = display.copy()
            overlay[mask.astype(bool)] = color
            cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)

            # Draw contour around mask
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, color, 2)

            # Place label at centroid of mask
            padding = 3
            ys, xs = np.where(mask.astype(bool))
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(ob.name, font, font_scale, thickness)
                tx, ty = cx - text_w // 2, int(ys.min()) - padding
                tx = max(0, min(tx, display.shape[1] - text_w))
                ty = max(text_h + padding, ty)

                cv2.putText(display, ob.name, (tx, ty), font, font_scale, (255, 255, 255), thickness)

        return display

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
        display = rgb_image.copy()
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
