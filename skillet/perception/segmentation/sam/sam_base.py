"""Base class for all Segment Anything (SAM) clients."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import httpx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jaxtyping import Float, Int, UInt8


def get_sam_client(model: Literal["sam2", "sam3", "sam3_streaming"] = "sam3") -> SAMClient:
    """Get a SAM client."""
    if model == "sam2":
        from skillet.perception.segmentation.sam.sam2_client import SAM2Client as SAM2Client

        return SAM2Client
    if model == "sam3":
        from skillet.perception.segmentation.sam.sam3_client import SAM3Client as SAM3Client

        return SAM3Client
    raise ValueError(f"Invalid SAM model: {model}")


@dataclass
class ConceptResponse:
    """Dataclass for segmenting from concepts."""

    masks_b64: list[str]
    boxes: list[list[float]]
    scores: list[float]
    concept_indices: list[int]
    count: int


@dataclass
class BboxResponse:
    """Dataclass for segmeneting from bounding boxes."""

    masks_b64: list[str]
    scores: list[float]
    count: int


SAM_SERVER_URL = "http://localhost:8000"


class SAMClient(ABC):
    """Base class for all SAM clients."""

    model_name: str = None

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        use_server: bool = True,
        load_server: bool = False,
        server_url: str = SAM_SERVER_URL,
    ) -> None:
        """Initialize the base SAM client.

        Args:
            model_path: Path to the SAM model
            device: Device to load model on

        """
        self.device = device
        self.model_path = model_path
        self.server_url = server_url
        self.use_server = use_server

        if use_server and not load_server:
            SAMClient.ensure_server(self.server_url, self.model_name)

    def reset(self) -> None:  # noqa: B027
        """Reset the SAM session."""
        pass

    @abstractmethod
    def segment_from_bboxes(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        bboxes: Sequence[Float[torch.Tensor | np.ndarray, "n 4"]] | None = None,
    ) -> tuple[Float[torch.Tensor, "n 1 h w"], Float[torch.Tensor, " n"]]:
        """Segment detection results from bounding boxes with SAM.

        Args:
            rgb: RGB image to segment. Can be an rgb from an RGBD obs or a PIL image.
            bboxes: Bounding boxes in [ymin, xmin, ymax, xmax] format in pixel space.

        Returns:
            - masks: Segmentation masks of shape (N, 1, H, W).
            - scores: Confidence scores of the segmentation masks.

        """
        raise NotImplementedError

    @abstractmethod
    def segment_from_concepts(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        concepts: list[str],
    ) -> tuple[
        Float[torch.Tensor, "n 1 h w"], Int[torch.Tensor, "n 4"], Float[torch.Tensor, " n"], Int[torch.Tensor, " n"]
    ]:
        """Segment an image from a list of text concepts.

        This functionality was introduced in SAM3.

        Args:
            rgb: RGB image to segment. Can be an rgb from an RGBD obs or a PIL image.
            concepts: List of text concepts to segment.

        Returns:
            - masks: Segmentation masks of shape (N, 1, H, W).
            - boxes: Boxes of the segmentation masks in [ymin, xmin, ymax, xmax] format in pixel space.
            - scores: Confidence scores of the segmentation masks.
            - prompt_indices: Corresponding indices of the prompts that were used to segment the image.

        """
        raise NotImplementedError

    def segment_concepts(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        concepts: list[str],
    ) -> tuple[
        Float[torch.Tensor, "n 1 h w"], Int[torch.Tensor, "n 4"], Float[torch.Tensor, " n"], Int[torch.Tensor, " n"]
    ]:
        """Segment an image from a list of text concepts."""
        if self.use_server:
            return self.remote_segment_from_concepts(rgb, concepts)
        return self.segment_from_concepts(rgb, concepts)

    def segment_bboxes(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        bboxes: Sequence[Float[torch.Tensor | np.ndarray, "n 4"]] | None = None,
    ) -> tuple[Float[torch.Tensor, "n 1 h w"], Float[torch.Tensor, " n"]]:
        """Segment detection results from bounding boxes with SAM."""
        if self.use_server:
            return self.remote_segment_from_bboxes(rgb, bboxes)
        return self.segment_from_bboxes(rgb, bboxes)

    def remote_segment_from_concepts(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        concepts: Sequence[str],
    ) -> tuple[
        Float[torch.Tensor, "n 1 h w"],
        Int[torch.Tensor, "n 4"],
        Float[torch.Tensor, " n"],
        Int[torch.Tensor, " n"],
    ]:
        """Segment an image from a list of text concepts on the remote server.

        This functionality was introduced in SAM3.

        Args:
            rgb: RGB image to segment. Can be an rgb from an RGBD obs or a PIL image.
            concepts: List of text concepts to segment.

        Returns:
            - masks: Segmentation masks of shape (N, 1, H, W).
            - boxes: Boxes of the segmentation masks in [ymin, xmin, ymax, xmax] format in pixel space.
            - scores: Confidence scores of the segmentation masks.
            - prompt_indices: Corresponding indices of the prompts that were used to segment the image.

        """
        image_bytes = self.arr_to_bytes(rgb)
        client = self._get_http_client()

        data = self._decode_response(
            client.post(
                "/segment/concepts",
                files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                data={"concepts": ",".join(concepts)},
            )
        )

        masks = self._b64_masks_to_tensor(data["masks_b64"], self.device)
        boxes = torch.tensor(data["boxes"], dtype=torch.float32, device=self.device)
        scores = torch.tensor(data["scores"], dtype=torch.float32, device=self.device)
        concept_indices = torch.tensor(data["concept_indices"], dtype=torch.int64, device=self.device)
        return masks, boxes, scores, concept_indices

    def remote_segment_from_bboxes(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        bboxes: Sequence[Float[torch.Tensor | np.ndarray, "n 4"]] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "n 1 h w"],
        Float[torch.Tensor, " n"],
    ]:
        """Segment detection results from bounding boxes with SAM on the remote server.

        Args:
            rgb: RGB image to segment. Can be an rgb from an RGBD obs or a PIL image.
            bboxes: Bounding boxes in [ymin, xmin, ymax, xmax] format in pixel space.

        Returns:
            - masks: Segmentation masks of shape (N, 1, H, W).
            - scores: Confidence scores of the segmentation masks.

        """
        image_bytes = self.arr_to_bytes(rgb)
        client = self._get_http_client()

        # Normalise bboxes to plain Python lists for JSON serialisation
        bbox_list = [b.cpu().tolist() if isinstance(b, torch.Tensor) else b.tolist() for b in bboxes]

        data = self._decode_response(
            client.post(
                "/segment/bboxes",
                files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                data={"bboxes": json.dumps(bbox_list)},
            )
        )

        masks = self._b64_masks_to_tensor(data["masks_b64"], self.device)
        scores = torch.tensor(data["scores"], dtype=torch.float32, device=self.device)

        return masks, scores

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Load model into GPU on startup; nothing to clean up on shutdown."""
        yield

    def create_server(self) -> FastAPI:
        """Create and return the FastAPI app bound to this client instance."""
        app = FastAPI(
            title="SAM Segmentation Server",
            lifespan=lambda a: self.lifespan(a),  # bind self
        )

        def _tensor_to_list(t: torch.Tensor) -> list:
            return t.cpu().tolist()

        def _upload_to_tensor(file_bytes: bytes) -> torch.Tensor:
            """Decode uploaded image bytes → CHW uint8 tensor on CUDA."""
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            arr = np.array(img, dtype=np.uint8)  # HWC
            return torch.as_tensor(arr).permute(2, 0, 1)  # CHW

        def _masks_to_b64(masks: torch.Tensor) -> list[str]:
            """Encode Nx1xHxW float mask tensor as a list of base64 PNG strings."""
            out = []
            for mask in masks:  # 1xHxW
                m = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                buf = io.BytesIO()
                Image.fromarray(m).save(buf, format="PNG")
                out.append(base64.b64encode(buf.getvalue()).decode())
            return out

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.post("/segment/concepts", response_model=ConceptResponse)
        async def segment_concepts(
            file: Annotated[UploadFile, File(description="RGB image (JPEG/PNG)")],
            concepts: Annotated[str, Form(description="Comma-separated concept strings, e.g. 'shoe,basketball hoop'")],
        ):
            """Segment all instances matching any of the provided text concepts.

            Returns one mask per detected instance, with the concept index it matched.
            """
            concept_list = [c.strip() for c in concepts.split(",") if c.strip()]
            if not concept_list:
                raise HTTPException(status_code=422, detail="Provide at least one concept.")

            rgb = _upload_to_tensor(await file.read())

            masks, boxes, scores, concept_indices = self.segment_from_concepts(rgb, concept_list)

            return ConceptResponse(
                masks_b64=_masks_to_b64(masks),
                boxes=_tensor_to_list(boxes),
                scores=_tensor_to_list(scores),
                concept_indices=_tensor_to_list(concept_indices),
                count=len(masks),
            )

        @app.post("/segment/bboxes", response_model=BboxResponse)
        async def segment_bboxes(
            file: Annotated[UploadFile, File(description="RGB image (JPEG/PNG)")],
            bboxes: Annotated[
                str,
                Form(description="JSON array of [ymin,xmin,ymax,xmax] boxes, e.g. '[[10,20,100,200]]'"),
            ],
        ):
            """Segment objects inside provided bounding boxes.

            Boxes should be in pixel-space [ymin, xmin, ymax, xmax] format —
            the same format your SAM3Client._convert_bounding_boxes() expects.
            """
            try:
                bbox_list = json.loads(bboxes)
                bbox_tensors = [torch.tensor(b, dtype=torch.float32) for b in bbox_list]
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Invalid bboxes JSON: {exc}") from exc

            rgb = _upload_to_tensor(await file.read())

            masks, scores = self.segment_from_bboxes(rgb, bbox_tensors)

            return BboxResponse(
                masks_b64=_masks_to_b64(masks),
                scores=_tensor_to_list(scores),
                count=len(masks),
            )

        return app

    @staticmethod
    def _is_server_alive(server_url: str = SAM_SERVER_URL) -> bool:
        """Check if the server is already running."""
        try:
            r = httpx.get(f"{server_url}/health", timeout=2.0)
            return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    @staticmethod
    def _wait_for_server(server_url: str, timeout: float = 30.0) -> bool:
        """Poll until server is up or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if SAMClient._is_server_alive(server_url):
                return True
            time.sleep(0.5)
        return False

    @staticmethod
    def ensure_server(server_url: str, model_name: str | None = None, timeout: float = 30.0) -> None:
        """Start the server as a detached background process if not already running."""
        if SAMClient._is_server_alive(server_url):
            return

        print("[INFO][SAM] No server found, spawning background process…")
        subprocess.Popen(
            [sys.executable, __file__, "--serve", model_name],
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        print("[INFO][SAM] Waiting for server to come up…")
        if not SAMClient._wait_for_server(server_url, timeout=timeout):
            raise RuntimeError(f"SAM server did not start within {timeout}s.")
        print("[INFO][SAM] Server is up.")

    def _get_http_client(self) -> httpx.Client:
        """Lazily create a persistent HTTP client (reuses TCP connection)."""
        if not hasattr(self, "_http_client"):
            self._http_client = httpx.Client(base_url=self.server_url, timeout=60.0)
        return self._http_client

    def arr_to_bytes(self, rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image) -> bytes:
        """Convert any supported image type to JPEG bytes for upload."""
        if isinstance(rgb, Image.Image):
            img = rgb
        else:
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            img = Image.fromarray(rgb.transpose(1, 2, 0).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def _decode_response(self, response: httpx.Response) -> dict:
        """Parse server response to JSON."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"SAM server returned {response.status_code}: {response.text}") from e
        return response.json()

    def _b64_masks_to_tensor(self, masks_b64: list[str], device: str) -> Float[torch.Tensor, "n 1 h w"]:
        """Decode list of base64 PNG masks back to a float tensor."""
        masks = []
        for b64 in masks_b64:
            arr = np.array(Image.open(io.BytesIO(base64.b64decode(b64))))
            masks.append(torch.as_tensor(arr / 255.0, dtype=torch.float32))
        if not masks:
            return torch.zeros((0, 1, 1, 1), dtype=torch.float32, device=device)
        return torch.stack(masks).to(device)


def main() -> None:
    """Run the SAM server."""
    if "--serve" in sys.argv:
        sam_client: SAMClient = get_sam_client(model=sys.argv[2])(load_server=True)
        app = sam_client.create_server()
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, workers=1, access_log=False)
    else:
        SAMClient.ensure_server(SAM_SERVER_URL)
        print(f"[INFO][SAM3] Ready. Server at {SAM_SERVER_URL}")


if __name__ == "__main__":
    main()
