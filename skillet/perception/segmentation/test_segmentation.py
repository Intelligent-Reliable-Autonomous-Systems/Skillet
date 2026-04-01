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

    def segmentation(self, rgb: UInt8[np.ndarray, "h w 3"], task_instruction: str) -> dict:
        """Test the segmentation and task instruction with the Gemini and SAM2 pipline.

        Args:
            rgb: RGB image to segment
            task_instruction: Instruction of the task to complete.

        """
        rgb_pil = Image.fromarray(rgb)
        rgb_pil_resized = rgb_pil.resize((800, int(800 * rgb_pil.size[1] / rgb_pil.size[0])), Image.Resampling.LANCZOS)
        print("[INFO] Starting Gemini object detection")
        _st = time.perf_counter()
        bboxes, grounded_atoms = self.gemini_client.detect_and_translate(rgb_pil_resized, task_instruction)
        _dur = time.perf_counter() - _st
        print(f"[INFO] Gemini detection took {_dur:.2f}s ({len(bboxes)} objects, {len(grounded_atoms)} atoms)")

        for bbox in bboxes:
            bbox["label"] = bbox["label"].replace(" ", "_")
        for atom in grounded_atoms:
            atom["args"] = [arg.replace(" ", "_") for arg in atom["args"]]

        print("[INFO] Starting SAM2 object segmentation with Gemini masks")
        _st = time.perf_counter()
        masks = self.sam_client.segment_objects(rgb_pil, bboxes)
        _dur = time.perf_counter() - _st
        print(f"[INFO] SAM2 segmentation took {_dur:.2f}s ({len(masks)} masks)")

        return {"bboxes": bboxes, "masks": masks, "grounded_atoms": grounded_atoms}


def display_segmentation_output(task_instruction, rgb, out):
    img_h, img_w = rgb.shape[:2]

    bboxes = out["bboxes"]
    masks = out["masks"]
    grounded_atoms = out["grounded_atoms"]

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
    axes[2].set_title("Grounded Atoms")
    atom_text = "\n".join([f"{a['predicate']}({', '.join(a['args'])})" for a in grounded_atoms])
    axes[2].text(
        0.1,
        0.5,
        atom_text if atom_text else "No atoms",
        transform=axes[2].transAxes,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
    )

    plt.suptitle(f"Task: {task_instruction}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("segmentation_output.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to segmentation_output.png")


# In your main():


def main():
    task_instruction = "Move the apple onto the cup."
    rgb = np.array(Image.open("test_robot.jpg"))
    seg = SegmentationPipeline()
    out = seg.segmentation(rgb, task_instruction=task_instruction)
    with pathlib.Path("out.pkl").open("wb") as f:
        pickle.dump(out, f)
    # with pathlib.Path("out.pkl").open("rb") as f:
    #     out = pickle.load(f)
    print(out["grounded_atoms"])
    display_segmentation_output(task_instruction, rgb, out)


if __name__ == "__main__":
    main()
