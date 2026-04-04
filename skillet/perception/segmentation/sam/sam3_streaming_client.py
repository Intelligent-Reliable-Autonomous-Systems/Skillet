"""SAM2 segmentation — local predictor and remote HTTP client."""

from collections.abc import Sequence
import pathlib
from functools import cache
from jaxtyping import Float, UInt8
import torch
from typing_extensions import override

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from skillet.perception.segmentation.sam.sam_base import SAMClient

_SAM3_BPE_URL = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"


class SAM3StreamingClient(SAMClient):
    """Main client class for the SAM3 model."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cuda",
        mode: str = "local",
        remote_url: str | None = None,
    ) -> None:
        super().__init__(model_name, device, mode, remote_url)

    @override
    def reset(self) -> None:
        resp = self.sam_model.handle_request({"type": "start_session"})
        print("Session start response:", resp)
        self._session_id = resp["session_id"]
        self._frame_idx = 0

    @override
    def segment_from_bboxes(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        bboxes: Sequence[Float[torch.Tensor | np.ndarray, "n 4"]] | None = None,
    ) -> Float[torch.Tensor, "n 1 h w"]:
        boxes = self._convert_bounding_boxes(rgb, bboxes)
        if isinstance(rgb, np.ndarray):
            rgb = torch.as_tensor(rgb, device=self.device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_model.handle_request({"type": "add_frame", "session_id": self._session_id, "frame": rgb})
            # Add text prompt only on first frame
            if self._frame_idx == 0:
                self.sam_model.handle_request(
                    {"type": "add_prompt", "session_id": self._session_id, "frame_index": 0, "bounding_boxes": boxes}
                )
            # Run per-frame inference
            resp = self.sam_model.handle_request(
                {"type": "run_inference", "session_id": self._session_id, "frame_index": self._frame_idx}
            )

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

    def _load_sam_model(self, checkpoint: str | None = None, confidence: float = 0.5):  # noqa: ANN202
        """Load and cache the SAM2 image predictor."""
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_stream_predictor

        print(f"[INFO][SAM] Loading SAM3 with checkpoint={checkpoint}, device={self.device}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sam_model = build_sam3_stream_predictor(checkpoint_path=checkpoint, device=self.device)
        return Sam3Processor(sam_model, confidence_threshold=confidence)
        return Sam3Processor(sam_model, confidence_threshold=confidence)
