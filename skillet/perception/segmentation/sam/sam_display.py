import argparse
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import torch

from skillet.perception.realsense import RealsenseEnv
from skillet.perception.reconstruction.utils import (
    filter_cube_centers,
    assign_objects_to_id,
    find_cube_centers,
    transform_cube_centers_to_world,
)
from skillet.perception.segmentation.sam import get_sam_client
from skillet.scene import SkilletVisualizer
from skillet.scene.base import Scene
from skillet.scene.cube import Cube
from skillet.scene.utils import get_sorted_object_poses, assign_poses_to_objects

if TYPE_CHECKING:
    from skillet.core import ObservationSpec
    from skillet.envs.specs import RGBD_Obs

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.40
THICKNESS = 1
PADDING = 4


def masks_to_bboxes(masks: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Convert binary masks of shape (N, H, W) to bounding boxes (x1, y1, x2, y2)."""
    bboxes = []
    for mask in masks:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        bboxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    return bboxes


def draw_overlay(
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


def show_masks(
    rgb_image: np.ndarray,
    masks: np.ndarray,
    concept_indices: np.ndarray | None = None,
    concepts: list | None = None,
    window_name: str = "SAM Detections",
) -> bool:
    """Display the image + mask overlay. Call this each time new masks arrive.

    Args:
        rgb_image:   HxWx3 RGB numpy array.
        masks:       NxHxW binary numpy array (SAM output).
        concept_indices: mask indices of each concept
        concepts: list of concepts from segmentation
        window_name: OpenCV window title.

    Returns:
        False if the user pressed 'q' or closed the window, True otherwise.

    """
    frame = draw_overlay(rgb_image, masks, concept_indices=concept_indices, concepts=concepts)
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    return key != ord("q")


def main(
    model: Literal["sam2", "sam3", "sam3_streaming", "sam3_ultralytics"],
    mode: Literal["text", "bboxes"] = "text",
):

    env = RealsenseEnv()
    print("RealsenseEnv initialized")
    sam_model = get_sam_client(model)
    print("Sam model loaded")

    TABLE_X0 = -0.0889
    TABLE_Y0 = -0.577
    TABLE_DX = 0.762
    TABLE_DY = 1.2446

    cube_0 = Cube(size=0.041, init_pose=torch.as_tensor([0.26, 0.041, 0.16, 1, 0, 0, 0], device="cuda"))
    cube_1 = Cube(size=0.041, init_pose=torch.as_tensor([0.44, 0.041, 0.16, 1, 0, 0, 0], device="cuda"))
    cube_2 = Cube(size=0.041, init_pose=torch.as_tensor([0.35, 0.041, 0.16, 1, 0, 0, 0], device="cuda"))

    world_bounds = (TABLE_X0, TABLE_Y0, 0, TABLE_X0 + TABLE_DX, TABLE_Y0 + TABLE_DY, 1)
    scene = Scene(objects=[cube_0, cube_1, cube_2], closed_set=True, bounds=world_bounds)
    print(scene)
    rgbd_spec: ObservationSpec[RGBD_Obs] = env.coerce_obs_spec("rgb-d")
    visualizer = SkilletVisualizer(
        env=env,
        obs_spec=rgbd_spec,
        scene=scene,
        poll_rate=8,
        device="cuda",
    )

    # visualizer.run_thread()

    print("Press 'q' to quit.")
    while True:
        obs = env.get_observation()

        rgb = obs["rgb"]
        depth = obs["depth"]
        intrinsic_k = obs["intrinsic_k"]
        camera_pose = obs["camera_pose"]

        if mode == "text":
            concepts = ["block"]
            # TODO: sometimes SAM segments the tops of the cubes as well
            # This sometimes gets the apriltag as well
            # Want to filter this based on cubes being close to each other
            masks, boxes, scores, concept_indices = sam_model.segment_from_concepts(rgb, concepts)
        elif mode == "bboxes":
            # TODO: Get this from VLM?
            # If so, get in format 0-1000 regardless of image scale
            bboxes = [
                [100, 100, 150, 150],
                [200, 200, 250, 250],
                [300, 300, 350, 350],
                [200, 100, 250, 150],
                [100, 200, 150, 250],
            ]
            masks, scores = sam_model.segment_from_bboxes(rgb, bboxes)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Might want to filter
        dc = find_cube_centers(masks.cpu().numpy(), depth, intrinsic_k, cube_size=0.041)
        centers = transform_cube_centers_to_world(
            dc["centers"], camera_pos=camera_pose[0:3], camera_quat=camera_pose[3:7]
        )

        poses, ids = get_sorted_object_poses(scene, Cube)
        cube_idx, det_idx = assign_objects_to_id(poses, centers)

        assign_poses_to_objects(scene, Cube, centers, ids, cube_idx, det_idx)

        print(scene)
        print("\n")
        # np.set_printoptions(suppress=True, precision=3)
        # for c in centers:
        #     print(c)
        # print()
        if not show_masks(rgb, masks.cpu().numpy(), concept_indices=concept_indices, concepts=concepts):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SAM segmentation with live visualization.")
    parser.add_argument(
        "--model",
        type=str,
        default="sam3",
        choices=["sam2", "sam3", "sam3_streaming", "sam3_ultralytics"],
        help="SAM backend to use.",
    )
    parser.add_argument("--mode", type=str, default="text", choices=["text", "bboxes"], help="text vs bbox prompts.")
    args = parser.parse_args()
    main(args.model, args.mode)
