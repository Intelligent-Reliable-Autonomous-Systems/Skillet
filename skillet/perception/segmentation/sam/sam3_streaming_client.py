"""SAM2 segmentation — local predictor and remote HTTP client."""

import pathlib
from functools import cache
from jaxtyping import Float
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
        print('Session start response:', resp)
        self._session_id = resp["session_id"]
        self._frame_idx = 0

    def _segment_local(self, image: Image.Image, boxes: Float[np.ndarray, "n 4"]) -> tuple[np.ndarray, np.ndarray]:
        """Run SAM3 segmentation locally.

        Args:
            image: PIL image to segment
            boxes: bounding boxes in xywh format (2 dimensional)

        Returns:
            Masks of segmented objects and confidence scores

        """
        self.sam_model.handle_request({"type": "add_frame", "session_id": self._session_id, "frame": image})
        # Add text prompt only on first frame
        if self._frame_idx == 0:
            self.sam_model.handle_request({"type": "add_prompt", "session_id": self._session_id, 
                "frame_index": 0, "bounding_boxes":})
        # Run per-frame inference
        resp = self.sam_model.handle_request({"type": "run_inference", "session_id": self._session_id, "frame_index": self._frame_idx})

    def _convert_bounding_boxes(self, rgb_pil: Image.Image, detection_results: list[dict]) -> np.ndarray:
        """Convert bounding boxes into required SAM3 format.

        Convert VLM bbox format [ymin, xmin, ymax, xmax] (0-1000) to SAM3 [center_x, center_y, width, height] format
        and normalized in [0, 1] range.

        Args:
            rgb_pil: RGB image to segment.
            detection_results: dictionary list of segmentation results from VLM.

        Returns:
            np.ndarray of bounding boxes

        """
        return np.array(
            [
                [
                    ((xmin + xmax) / 2) / 1000.0,  # center_x
                    ((ymin + ymax) / 2) / 1000.0,  # center_y
                    (xmax - xmin) / 1000.0,  # width
                    (ymax - ymin) / 1000.0,  # height
                ]
                for detection in detection_results
                if len(box_2d := detection.get("box_2d", [])) == 4
                for ymin, xmin, ymax, xmax in [box_2d]
            ],
            dtype=np.float32,
        )

    def _download_sam_checkpoint(self, model_name: str | None = None) -> pathlib.Path:
        """Ensure the SAM3 BPE tokenizer file exists in cache.

        Returns:
            Path to the BPE file.

        """
        from skillet.perception.utils import get_skillet_model_cache_dir

        cache_dir = get_skillet_model_cache_dir() / "sam3"

        bpe_path = cache_dir / "bpe_simple_vocab_16e6.txt.gz"

        if bpe_path.exists():
            return bpe_path

        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO][SAM3] Downloading BPE vocab: {bpe_path}")

        response = requests.get(_SAM3_BPE_URL, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with (
            pathlib.Path(bpe_path).open("wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="bpe_vocab",
            ) as pbar,
        ):
            for chunk in response.iter_content(block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print("[SAM3] BPE vocab ready.")
        return bpe_path

    @cache
    def _load_sam_model(self, checkpoint: str | None = None, confidence: float = 0.5):  # noqa: ANN202
        """Load and cache the SAM2 image predictor."""
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_stream_predictor

        print(f"[INFO][SAM] Loading SAM3 with checkpoint={checkpoint}, device={self.device}")
        return Sam3Processor(
            build_sam3_stream_predictor(checkpoint_path=checkpoint, device=self.device),
            confidence_threshold=confidence
        )
