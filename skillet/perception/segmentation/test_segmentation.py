import argparse
import pathlib
import pickle
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import UInt8
from PIL import Image

from skillet.perception.segmentation.sam import SAM2Client
from skillet.perception.segmentation.vlm import GeminiClient


class SegmentationPipeline:
    """Segmentation Pipeline using Gemini and SAM3."""

    def __init__(self) -> None:
        self.sam_client = SAM2Client()
        self.gemini_client = GeminiClient()

    def _segmentation(self, rgb: UInt8[np.ndarray, "h w 3"], task_instruction: str) -> dict:
        """Test the segmentation and task instruction with the Gemini and SAM2 pipline.

        Args:
            rgb: RGB image to segment
            task_instruction: Instruction of the task to complete.

        """
        rgb_pil = Image.fromarray(rgb)
        rgb_pil_resized = rgb_pil.resize((800, int(800 * rgb_pil.size[1] / rgb_pil.size[0])), Image.Resampling.LANCZOS)
        print("[INFO] Starting Gemini object detection")
        _st = time.perf_counter()
        bboxes, grounded_goal_atoms, grounded_scene_atoms = self.gemini_client.detect_and_translate(
            rgb_pil_resized, task_instruction
        )
        _dur = time.perf_counter() - _st
        print(f"[INFO] Gemini detection took {_dur:.2f}s ({len(bboxes)} objects, {len(grounded_goal_atoms)} atoms)")

        for bbox in bboxes:
            bbox["label"] = bbox["label"].replace(" ", "_")
        for atom in grounded_goal_atoms:
            atom["args"] = [arg.replace(" ", "_") for arg in atom["args"]]
        for atom in grounded_scene_atoms:
            atom["args"] = [arg.replace(" ", "_") for arg in atom["args"]]

        print("[INFO] Starting SAM object segmentation with Gemini masks")
        _st = time.perf_counter()
        masks = self.sam_client.segment_objects(rgb_pil, bboxes)
        _dur = time.perf_counter() - _st
        print(f"[INFO] SAM segmentation took {_dur:.2f}s ({len(masks)} masks)")

        return {
            "bboxes": bboxes,
            "masks": masks,
            "grounded_goal_atoms": grounded_goal_atoms,
            "grounded_scene_atoms": grounded_scene_atoms,
        }


def display_segmentation_output(task_instruction: str, rgb: np.ndarray, out: dict, dir: str):
    """Display the output of segmentation with Gemini/SAM2.

    Args:
        task_instruction: the task instruction as a natural language prompt.
        rgb: the RGB image of the scene
        out: a dictionary containing the bounding boxes, segmentation masks, and PDDL predictates
        dir: directory to save the output of segmentation

    """
    img_h, img_w = rgb.shape[:2]

    bboxes = out["bboxes"]
    masks = out["masks"]
    grounded_goal_atoms = out["grounded_goal_atoms"]
    grounded_scene_atoms = out["grounded_scene_atoms"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Display original image with bounding boxes
    axes[0].imshow(rgb)
    axes[0].set_title("Gemini Bounding Boxes")
    colors = plt.cm.tab10.colors
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        y1, x1, y2, x2 = bbox["box_2d"]
        x1, y1, x2, y2 = (
            int(x1 / 1000 * img_w),
            int(y1 / 1000 * img_h),
            int(x2 / 1000 * img_w),
            int(y2 / 1000 * img_h),
        )

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 5, bbox["label"], color=color, fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    axes[1].set_title("SAM2 Masks Overlay")

    for i in range(masks.shape[0]):
        mask = masks[i].squeeze()
        color = np.array(colors[i % len(colors)])
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask.astype(np.bool_)] = [*color, 0.5]
        axes[1].imshow(overlay)
        # Label at mask centroid
        ys, xs = np.where(mask)
        if len(xs) > 0:
            axes[1].text(
                xs.mean(), ys.mean(), bboxes[i]["label"], color="white", fontsize=9, ha="center", fontweight="bold"
            )
    axes[1].axis("off")

    axes[2].axis("off")
    axes[2].set_title("Scene + Goal Description Predicates")
    scene_atom_text = "\n".join([f"{a['scene_predicate']}({', '.join(a['args'])})" for a in grounded_scene_atoms])
    goal_atom_text = "\n".join([f"{a['goal_predicate']}({', '.join(a['args'])})" for a in grounded_goal_atoms])
    axes[2].text(
        0.1,
        0.55,  # slightly above scene text (0.3)
        "Scene Predicates",
        transform=axes[2].transAxes,
        fontsize=13,
        fontweight="bold",
        verticalalignment="center",
    )
    axes[2].text(
        0.1,
        0.5,
        scene_atom_text if scene_atom_text else "No atoms",
        transform=axes[2].transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
    )
    axes[2].text(
        0.1,
        0.85,  # slightly above goal text (0.8)
        "Goal Predicates",
        transform=axes[2].transAxes,
        fontsize=13,
        fontweight="bold",
        verticalalignment="center",
    )
    axes[2].text(
        0.1,
        0.8,
        goal_atom_text if goal_atom_text else "No atoms",
        transform=axes[2].transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
    )

    plt.suptitle(f"Task: {task_instruction}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{dir}/segmentation_output.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ti", type=str, default="Move the red block onto the purple block")
    parser.add_argument("--dir", type=str, default="captures/capture_20260402_083413/")
    parser.add_argument(
        "--new",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to redo segmentation on an image or load from .pkl",
    )
    args = parser.parse_args()

    rgb = np.array(Image.open(f"{args.dir}/img_color.png"))
    if args.new:
        seg = SegmentationPipeline()
        out = seg._segmentation(rgb, task_instruction=args.ti)
        with pathlib.Path(f"{args.dir}/out.pkl").open("wb") as f:
            pickle.dump(out, f)
    else:
        with pathlib.Path(f"{args.dir}/out.pkl").open("rb") as f:
            out = pickle.load(f)
    display_segmentation_output(args.ti, rgb, out, args.dir)


if __name__ == "__main__":
    main()
