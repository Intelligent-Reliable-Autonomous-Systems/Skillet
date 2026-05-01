"""sam_reconstructor.py.

Reconstruct the scene from SAM3 concepts and bounding boxes.
"""

import pathlib
import pickle
from typing import Any, Literal

import cv2
import numpy as np
import torch

from skillet.perception.reconstruction.reconstructor_base import ReconstructorBase
from skillet.perception.reconstruction.utils import (
    find_cube_centers_mean,
    get_sorted_object_poses,
    transform_xyz_to_world,
)
from skillet.perception.segmentation.sam import SAMClient, get_sam_client
from skillet.perception.segmentation.vlm import GeminiClient, QwenClient
from skillet.perception.segmentation.vlm.vlm_base import VLMClient
from skillet.scene import CUBE_SIZE, Cube, Target
from skillet.scene.base import Scene


class Sam3Reconstructor(ReconstructorBase):
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
        self._sam_model: SAMClient = get_sam_client("sam3")(use_server=True)
        self._vlm_client: VLMClient = GeminiClient(prompt_name="detect_goal_qwen")
        self._visualize = visualize

        self._masks = None
        self._segment_indices = None

        # Scene reconstruction
        self._vlm_bboxes = None
        self._vlm_goal_atoms = None

        self._concepts = ["robot arm"]
        for o in self._scene.get_object_names(Cube):
            self._concepts.append(o.replace("_", " "))
        for o in self._scene.get_object_names(Target):
            self._concepts.append(o.replace("_", " "))

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
        if self._build_scene_flag:
            print("[INFO][SAM RECONSTRUCTOR] Building scene...")
            self._build_scene(obs, frame=frame)
            print("[INFO][SAM RECONSTRUCTOR] Successfully built scene.")
            self._build_scene_flag = False

        rgb = obs["rgb"]
        depth = obs["depth"]
        intrinsic_k = obs["intrinsic_k"]
        camera_pose = obs["camera_pose"]
        masks, _, _, concept_indices = self._sam_model.segment_concepts(rgb, self._concepts)

        self._masks = masks
        self._segment_indices = torch.arange(masks.shape[0], device=masks.device)

        # Grab only the cubes and combine overlapping indices
        agg_cube_masks = []
        _, mh, mw = masks.shape
        cube_inds = [j for j, item in enumerate(self._concepts) if "block" in item]
        for i in cube_inds:  # grab only cube masks
            if i not in concept_indices:
                continue
            inds = torch.argwhere(i == concept_indices)[0]
            c_mask = torch.zeros(size=(mh, mw), device=self._device)
            for j in inds:
                c_mask = torch.logical_or(c_mask, masks[j].squeeze())
            agg_cube_masks.append(c_mask)
        if len(agg_cube_masks) == 0:
            return
        cube_masks = torch.stack(agg_cube_masks, dim=0)

        # Find cube centers in the camera frame
        centers = find_cube_centers_mean(
            cube_masks,
            depth,
            intrinsic_k,
            cube_size=CUBE_SIZE,
            camera_pos=camera_pose[0:3],
            camera_quat=camera_pose[3:7],
            frame=frame,
        )

        # TODO localize target centers as well

        centers = (
            transform_xyz_to_world(centers, camera_pos=camera_pose[0:3], camera_quat=camera_pose[3:7])
            if frame == "camera"
            else centers
        )

        _, ids = get_sorted_object_poses(self._scene, Cube)
        cube_idx, det_idx = [], []
        for i, c in enumerate(torch.unique(concept_indices[concept_indices != 0]).cpu().numpy()):
            if c not in cube_inds:
                continue
            cube = self._scene.get_objects_from_name([self._concepts[c].replace(" ", "_")])[0]
            cube.pose = torch.cat((centers[i], torch.as_tensor([1, 0, 0, 0], device=centers[i].device)), dim=0)
            cube_idx.append(int(np.argwhere(cube.object_id == ids)[0][0]))
            det_idx.append(i)

        if self._visualize:
            self._bbox_frame = Sam3Reconstructor.show_bounding_boxes(
                rgb.cpu().numpy(), masks.cpu().numpy(), concept_indices=concept_indices, concepts=self._concepts
            )
            if cube_idx is not None and det_idx is not None:
                if not hasattr(self, "_colors"):
                    self._colors = [
                        (int(c[0]), int(c[1]), int(c[2]))
                        for c in np.random.randint(100, 255, size=(len(self._scene.objects), 3))
                    ]
                self._mask_frame = Sam3Reconstructor.show_cube_masks(
                    rgb.cpu().numpy(), cube_masks.cpu().numpy(), self._scene, ids, cube_idx, det_idx, self._colors
                )

    def get_observation(self) -> Scene:
        """Return the scene."""
        return self._scene

    def _build_scene(
        self,
        obs: dict[str, torch.Tensor],
        call_vlm: bool = True,
        frame: Literal["world", "camera"] = "camera",
    ) -> None:
        """Build the scene using an API call to a VLM by creating bounding boxes for each object.

        Args:
            obs: RGBD obs spec observation
            call_vlm: If to call VLM or load scene from defaults
            frame: the frame in which to compute centers in

        """
        if self._task_instruction is None:
            self._task_instruction = "Put the red block on the purple block."
        rgb = obs["rgb"]
        depth = obs["depth"]
        camera_pose = obs["camera_pose"]
        intrinsic_k = obs["intrinsic_k"]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
            depth = depth.cpu().numpy()
            camera_pose = camera_pose.cpu().numpy()
            intrinsic_k = intrinsic_k.cpu().numpy()
        if call_vlm:
            _, _, self._vlm_goal_atoms = self._vlm_client.detect_goal(self._task_instruction)
            for atom in self._vlm_goal_atoms:
                atom["args"] = [arg.replace(" ", "_") for arg in atom["args"]]

        self._scene.goal = self._vlm_goal_atoms
        print(self._scene.goal)

        pathlib.Path("data/test/").mkdir(exist_ok=True, parents=True)
        with pathlib.Path("data/test/vlm_out_multi.pkl").open("wb") as f:
            pickle.dump(self._scene, f)
        print(f"[INFO] Reconstructed Goal with VLM.\n{self._scene}")

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
    def masks_to_bboxes(masks: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Convert binary masks of shape (N, H, W) to bounding boxes (x1, y1, x2, y2)."""
        bboxes = []
        for mask in masks:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            bboxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        return bboxes

    @staticmethod
    def show_cube_masks(
        rgb_image: np.ndarray,
        masks: np.ndarray,
        scene: Scene,
        ids: np.ndarray,
        obj_idx: np.ndarray,
        det_idx: np.ndarray,
        colors: list[tuple[int, int, int]],
    ) -> np.ndarray:
        """Show the masks and the corresponding labels.

        Args:
            rgb_image: RGB image from camera
            masks: masks produced by SAM
            scene: the current scene to obtain
            ids: np.ndarray of sorted object ids
            obj_idx: Sorted indexes of object scene ids according to poses
            det_idx: The detection index of which pose to assign to which object

        """
        rgb_image = rgb_image.transpose((1, 2, 0))
        display = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR).copy()

        for color_idx, ob in enumerate(scene.objects):
            if not isinstance(ob, Cube):
                continue

            idx = np.where(ob.object_id == ids[obj_idx])[0]
            if idx.size > 0:
                idx = idx[0]
            else:
                continue

            mask = masks[det_idx[idx]]  # shape (H, W), bool or 0/1
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
            baseline = 3
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
