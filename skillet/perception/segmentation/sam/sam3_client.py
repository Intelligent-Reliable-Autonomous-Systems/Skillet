"""SAM2 segmentation — local predictor and remote HTTP client."""

import base64
import io
import pathlib
from functools import cache

import numpy as np
import requests
from jaxtyping import Float
from PIL import Image
from tqdm import tqdm

_SAM3_BPE_URL = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"


class SAM3Client:
    """Main client class for the SAM3 model."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cuda",
        mode: str = "local",
        remote_url: str | None = None,
    ) -> None:
        self.device = device
        self.mode = mode
        self.remote_url = remote_url
        self.model_path = self._get_bpe_path()
        self.sam_model = self._load_sam_model(checkpoint=self.model_path)

    def segment_objects(
        self,
        rgb_pil: Image.Image,
        detection_results: list[dict],
    ) -> Float[np.ndarray, "n 1 h w"]:
        """Segment detection results from VLM with SAM3.

        Args:
            rgb_pil: PIL Image to segment.
            detection_results: List of detection dicts from VLM, each with a 'box_2d' key
                            in [ymin, xmin, ymax, xmax] format normalized to 0-1000.

        Returns:
            Segmentation masks of shape (N, 1, H, W).

        """
        # Convert VLM bbox format [ymin, xmin, ymax, xmax] (0-1000) to SAM2 [x0, y0, x1, y1] (pixels)
        img_height, img_width = rgb_pil.height, rgb_pil.width
        boxes = np.array(
            [
                [
                    (xmin / 1000.0) * img_width,
                    (ymin / 1000.0) * img_height,
                    (xmax / 1000.0) * img_width,
                    (ymax / 1000.0) * img_height,
                ]
                for detection in detection_results
                if len(box_2d := detection.get("box_2d", [])) == 4
                for ymin, xmin, ymax, xmax in [box_2d]
            ]
        )

        if self.mode == "local":
            masks, _ = self._segment_local(rgb_pil, boxes)
        else:
            masks, _ = self._segment_remote(rgb_pil, boxes, self.remote_url)

        if masks.ndim == 3:
            masks = masks[None]
        return masks

    def _segment_local(self, image: Image.Image, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run SAM3 segmentation locally.

        Args:
            image: PIL image to segment
            boxes: bounding boxes

        Returns:
            Masks of segmented objects and confidence scores

        """
        self.sam_model.set_image(image)
        masks, scores, _ = self.sam_model.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        return masks, scores

    def _segment_remote(self, image: Image.Image, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run SAM3 segmentation via remote server."""
        assert self.remote_url is not None
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = {"image_base64": base64.b64encode(buffer.getvalue()).decode(), "boxes": boxes.tolist()}

        try:
            response = requests.post(f"{self.remote_url}/segment", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            masks = np.array(
                [[np.load(io.BytesIO(base64.b64decode(m))) for m in mask_batch] for mask_batch in result["masks"]]
            )
            return masks, np.array(result["scores"])

        except Exception as e:
            print(f"[SAM][ERROR] Remote SAM2 segmentation failed: {e}")
            raise e

    def _get_bpe_path(self) -> pathlib.Path:
        """Ensure the SAM3 BPE tokenizer file exists in cache.

        Args:
            force_download: If True, re-download even if file exists.

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
        import sam3
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        sam3_root = pathlib.Path(sam3.__file__).parent
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

        print(f"[INFO][SAM] Loading SAM3 with checkpoint={checkpoint}, device={self.device}")
        return Sam3Processor(build_sam3_image_model(bpe_path=checkpoint), confidence_threshold=confidence)
