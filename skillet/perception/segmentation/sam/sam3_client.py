"""SAM2 segmentation — local predictor and remote HTTP client."""

import pathlib
from collections.abc import Sequence
import os
import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from PIL import Image
import sam3
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from typing_extensions import override

from skillet.perception.segmentation.sam.sam_base import SAMClient
from skillet.perception.utils import get_skillet_model_cache_dir

_SAM3_BPE_URL = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"


class SAM3Client(SAMClient):
    """Main client class for the SAM3 model."""

    def __init__(
        self,
        model_name: str = "sam3.pt",
        device: str = "cuda",
    ) -> None:
        """Initialize the SAM3 client.

        Args:
            model_name: Name of the SAM3 model checkpoint
            device: Device to load the model on

        """
        model_path = get_skillet_model_cache_dir() / model_name
        super().__init__(model_path, device)
        self.sam_model = self._load_sam_model(checkpoint=model_path)
        print("[INFO][SAM3] Successfully loaded SAM3 Model")

    @override
    def segment_from_bboxes(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        bboxes: Sequence[Float[torch.Tensor | np.ndarray, "n 4"]] | None = None,
    ) -> Float[torch.Tensor, "n 1 h w"]:
        boxes = self._convert_bounding_boxes(rgb, bboxes)
        masks = []
        scores = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if isinstance(rgb, np.ndarray):
                rgb = torch.as_tensor(rgb, device=self.device)

            for idx, box in enumerate(boxes):
                state = self.sam_model.set_image(rgb)
                box_state = self.sam_model.add_geometric_prompt(box.tolist(), True, state)

                if len(box_state["masks"]) == 0:
                    masks.append(torch.zeros((0, 1, *rgb.shape[-2:]), dtype=torch.float32, device=self.device))
                    scores.append(torch.zeros((0,), dtype=torch.float32, device=self.device))
                    continue
                best_idx = box_state["scores"].argmax().item()
                masks.append(box_state["masks"][best_idx].detach())
                scores.append(box_state["scores"][best_idx].detach().item())

        if len(masks) == 0:
            return torch.zeros((0, 1, *rgb.shape[-2:]), dtype=torch.float32, device=self.device), torch.zeros(
                (0,), dtype=torch.float32, device=self.device
            )

        masks_t = torch.cat(masks, dim=0).to(dtype=torch.float32, device=self.device)
        scores_t = torch.tensor(scores, dtype=torch.float32, device=self.device)

        return masks_t, scores_t

    @override
    def segment_from_concepts(
        self, rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image, concepts: Sequence[str]
    ) -> tuple[
        Float[torch.Tensor, "n 1 h w"], Int[torch.Tensor, "n 4"], Float[torch.Tensor, " n"], Int[torch.Tensor, " n"]
    ]:
        boxes = []
        masks = []
        scores = []
        concept_indices = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if isinstance(rgb, np.ndarray):
                rgb = torch.as_tensor(rgb, device=self.device)

            state = self.sam_model.set_image(rgb)
            for idx, concept in enumerate(concepts):
                concept_state = self.sam_model.set_text_prompt(concept, state)
                for box_idx in range(len(state["boxes"])):
                    boxes.append(concept_state["boxes"][box_idx].detach())
                    masks.append(concept_state["masks"][box_idx].detach())
                    scores.append(concept_state["scores"][box_idx].detach())
                    concept_indices.append(idx)

        if len(masks) == 0:
            return (
                torch.zeros((0, 1, *rgb.shape[-2:]), dtype=torch.float32, device=self.device),
                torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                torch.zeros((0,), dtype=torch.float32, device=self.device),
                torch.zeros((0,), dtype=torch.int64, device=self.device),
            )

        masks_t = torch.cat(masks, dim=0).to(dtype=torch.float32, device=self.device)
        scores_t = torch.tensor(scores, dtype=torch.float32, device=self.device)
        concept_indices_t = torch.tensor(concept_indices, dtype=torch.int64, device=self.device)

        # Convert boxes to [ymin, xmin, ymax, xmax] format
        boxes_t = torch.stack(boxes, dim=0).to(
            dtype=torch.float32, device=self.device
        )  # in [center_x, center_y, width, height] format
        minx = boxes_t[:, 0] - boxes_t[:, 2] / 2
        miny = boxes_t[:, 1] - boxes_t[:, 3] / 2
        maxx = boxes_t[:, 0] + boxes_t[:, 2] / 2
        maxy = boxes_t[:, 1] + boxes_t[:, 3] / 2
        boxes_t = torch.stack([minx, miny, maxx, maxy], dim=1)
        return masks_t, boxes_t, scores_t, concept_indices_t

    def _convert_bounding_boxes(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        bboxes: Sequence[Float[torch.Tensor | np.ndarray, "n 4"]] | None = None,
    ) -> Float[torch.Tensor, "n 4"]:
        """Convert bounding boxes into required SAM3 format.

        Convert bbox format [ymin, xmin, ymax, xmax] pixel space to
        SAM3 [center_x, center_y, width, height] format
        and normalized in [0, 1] range.

        Args:
            rgb: RGB image to segment. Can be an rgb from an RGBD obs or a PIL image.
            bboxes: Bounding boxes in [ymin, xmin, ymax, xmax] format in pixel space.

        Returns:
            np.ndarray of bounding boxes

        """
        height, width = rgb.shape[-2:]
        return torch.tensor(
            [
                [
                    ((xmin + xmax) / 2) / width,  # center_x
                    ((ymin + ymax) / 2) / height,  # center_y
                    (xmax - xmin) / width,  # width
                    (ymax - ymin) / height,  # height
                ]
                for ymin, xmin, ymax, xmax in bboxes
            ],
            dtype=torch.float32,
        )

    def _load_sam_model(self, checkpoint: pathlib.Path | None = None, confidence: float = 0.5):  # noqa: ANN202
        """Load and cache the SAM2 image predictor."""

        if checkpoint is not None and not checkpoint.exists():
            # Let sam3 download the checkpoint if it doesn't exist
            checkpoint = None
        print(f"[INFO][SAM] Loading SAM3 with checkpoint={checkpoint or 'default'}, device={self.device}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sam_model = build_sam3_image_model(checkpoint_path=checkpoint)
        return Sam3Processor(sam_model, confidence_threshold=confidence)


if __name__ == "__main__":
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

    import torch

    image_path = f"{sam3_root}/assets/images/test_image.jpg"
    image = Image.open(image_path)
    width, height = image.size

    image.show()
    sam3_client = SAM3Client()
    masks, boxes, scores, concept_indices = sam3_client.segment_from_concepts(
        image, ["shoe", "a child", "basketball hoop"]
    )
    print(masks.shape, boxes.shape, scores.shape, concept_indices.shape)
