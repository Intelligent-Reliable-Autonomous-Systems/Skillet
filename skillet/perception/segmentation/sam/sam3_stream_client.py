"""SAM3 segmentation — real-time video streaming predictor (HuggingFace transformers).

Unlike `SAM3Client`, which segments every frame independently, this client keeps a
`Sam3VideoInferenceSession` alive across calls. Each call to `segment_from_concepts`
pushes one new frame into the session, so objects keep a stable track ID over time
and the tracker's memory bank conditions each frame on the recent past.

The session is (re)initialised whenever the concept list changes or `reset()` is called.
Call `reset()` between episodes or whenever the camera view jumps discontinuously.

When running behind the SAM server, the session lives in the server process, so
only one client should stream frames to a given server at a time.
"""

from collections.abc import Sequence

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.models.sam3_video.modeling_sam3_video import Sam3VideoInferenceSession
from typing_extensions import override

from skillet.perception.segmentation.sam.sam_base import SAMClient


class _StreamingInferenceSession(Sam3VideoInferenceSession):
    """Inference session whose frame count survives dropping old frames.

    Sam3Video runs its tracker in offline mode even when streaming, and the tracker caps the
    object pointers it attends to by `num_frames`. Report frames seen so far (what an untrimmed
    session reports) so `SAM3StreamClient._trim_session` can discard old pixel data.
    """

    @property
    def num_frames(self) -> int | None:
        return max(self.processed_frames) + 1 if self.processed_frames else None


class SAM3StreamClient(SAMClient):
    """SAM3 client that tracks concepts across a stream of frames."""

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        device: str = "cuda",
        use_server: bool = True,
        load_server: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize the SAM3 streaming client.

        Args:
            model_name: HuggingFace model id of the SAM3 video checkpoint
            device: Device to load the model on
            use_server: if to use a SAM server
            load_server: if this instance is the one backing the server
            dtype: dtype for the model weights and the inference session

        """
        self.model_name = "sam3_streaming"
        self.dtype = dtype
        super().__init__(model_name, device, use_server, load_server)

        self._session: _StreamingInferenceSession | None = None
        self._concepts: list[str] = []
        self._frame_idx = 0
        # Track IDs of the objects returned by the last call, aligned with its masks
        self.object_ids = torch.zeros((0,), dtype=torch.int64)

        if (load_server and use_server) or not use_server:
            self.sam_model, self.processor = self._load_sam_model(model_name)
            tracker_config = self.sam_model.tracker_model.config
            # Number of past frames the tracker's memory attention can read from
            self._history = max(tracker_config.num_maskmem, tracker_config.max_object_pointers_in_encoder)
            self._max_cond_frames = tracker_config.max_cond_frame_num
            print("[INFO][SAM3 STREAM] Successfully loaded SAM3 Video Model")

    @override
    def reset(self) -> None:
        """Drop the current tracking session; the next frame starts a fresh one."""
        if self.use_server and not self.load_server:
            self._decode_response(self._get_http_client().post("/reset"))
            return
        self._session = None
        self._concepts = []
        self._frame_idx = 0

    @override
    def segment_from_bboxes(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
        bboxes: Sequence[Float[torch.Tensor | np.ndarray, "n 4"]] | None = None,
    ) -> tuple[Float[torch.Tensor, "n 1 h w"], Float[torch.Tensor, " n"]]:
        raise NotImplementedError("SAM3 streaming only supports text concepts; use the `sam3` client for bboxes.")

    @override
    def segment_from_concepts(
        self, rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image, concepts: Sequence[str]
    ) -> tuple[
        Float[torch.Tensor, "n 1 h w"], Int[torch.Tensor, "n 4"], Float[torch.Tensor, " n"], Int[torch.Tensor, " n"]
    ]:
        concepts = list(concepts)
        if self._session is None or concepts != self._concepts:
            self._start_session(concepts)

        if isinstance(rgb, np.ndarray):
            rgb = torch.as_tensor(rgb)
        inputs = self.processor(images=rgb, device=self.device, return_tensors="pt").to(self.device)

        model_outputs = self.sam_model(
            inference_session=self._session,
            frame=inputs.pixel_values[0],
            frame_idx=self._frame_idx,
        )
        outputs = self.processor.postprocess_outputs(self._session, model_outputs, original_sizes=inputs.original_sizes)
        self._trim_session(self._frame_idx)
        self._frame_idx += 1

        object_ids = outputs["object_ids"].to(dtype=torch.int64, device=self.device)
        height, width = inputs.original_sizes[0].tolist()
        if object_ids.numel() == 0:
            self.object_ids = object_ids
            return (
                torch.zeros((0, 1, height, width), dtype=torch.float32, device=self.device),
                torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                torch.zeros((0,), dtype=torch.float32, device=self.device),
                torch.zeros((0,), dtype=torch.int64, device=self.device),
            )

        obj_id_to_concept = {
            obj_id: concepts.index(prompt)
            for prompt, obj_ids in outputs["prompt_to_obj_ids"].items()
            for obj_id in obj_ids
        }
        concept_indices_t = torch.tensor(
            [obj_id_to_concept[obj_id] for obj_id in object_ids.tolist()], dtype=torch.int64, device=self.device
        )
        masks_t = outputs["masks"].to(dtype=torch.float32, device=self.device).unsqueeze(1)
        # Boxes are already [xmin, ymin, xmax, ymax] in pixel space, matching SAM3Client
        boxes_t = outputs["boxes"].to(dtype=torch.float32, device=self.device)
        scores_t = outputs["scores"].to(dtype=torch.float32, device=self.device)

        self.object_ids = object_ids
        return masks_t, boxes_t, scores_t, concept_indices_t

    def _start_session(self, concepts: list[str]) -> None:
        """Open a new streaming session prompted with `concepts`."""
        # Equivalent to `processor.init_video_session()` without a video, but with our session class
        self._session = _StreamingInferenceSession(
            inference_device=self.device,
            inference_state_device=self.device,
            video_storage_device=self.device,
            dtype=self.dtype,
        )
        self.processor.add_text_prompt(inference_session=self._session, text=concepts)
        self._concepts = concepts
        self._frame_idx = 0

    def _trim_session(self, frame_idx: int) -> None:
        """Drop per-frame state the tracker will never read again.

        The HF session keeps every frame and every per-frame output forever, which grows GPU
        memory by ~16 MB/frame on a long-running robot stream. The tracker only reads the last
        `num_maskmem` / `max_object_pointers_in_encoder` non-conditioning frames, so older ones
        are dropped losslessly. Conditioning frames are capped at the `max_cond_frame_num` most
        recent: HF draws object pointers from *every* conditioning frame (one more per object
        each reconditioning), while the reference SAM3 implementation only uses the selected ones.
        """
        session = self._session
        oldest = frame_idx - self._history

        session.processed_frames = {frame_idx: session.processed_frames[frame_idx]}
        for output_dict in session.output_dict_per_obj.values():
            non_cond = output_dict["non_cond_frame_outputs"]
            for t in [t for t in non_cond if t < oldest]:
                del non_cond[t]
            cond = output_dict["cond_frame_outputs"]
            if self._max_cond_frames > 0 and len(cond) > self._max_cond_frames:
                for t in sorted(cond)[: -self._max_cond_frames]:
                    del cond[t]
        for frames_tracked in session.frames_tracked_per_obj.values():
            for t in [t for t in frames_tracked if t < oldest]:
                del frames_tracked[t]
        for per_frame in (session.obj_id_to_tracker_score_frame_wise, session.suppressed_obj_ids):
            for t in [t for t in per_frame if t < oldest]:
                del per_frame[t]
        # Hotstart bookkeeping is append-only and its removal logic is disabled in streaming mode
        for per_obj in (session.unmatched_frame_inds, session.overlap_pair_to_frame_inds):
            for key, frame_inds in per_obj.items():
                per_obj[key] = [t for t in frame_inds if t >= oldest]

    def _load_sam_model(self, model_name: str) -> tuple[Sam3VideoModel, Sam3VideoProcessor]:
        """Load the SAM3 video model and its processor."""
        print(f"[INFO][SAM] Loading SAM3 Video with checkpoint={model_name}, device={self.device}")
        model = Sam3VideoModel.from_pretrained(model_name, dtype=self.dtype).to(self.device).eval()
        processor = Sam3VideoProcessor.from_pretrained(model_name)
        return model, processor


if __name__ == "__main__":
    import sys
    import time

    import cv2

    # Usage: python -m skillet.perception.segmentation.sam.sam3_stream_client <video.mp4> [num_frames]
    capture = cv2.VideoCapture(sys.argv[1])
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    video_frames = []
    while len(video_frames) < num_frames and (frame := capture.read()[1]) is not None:
        video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    client = SAM3StreamClient(use_server=False)
    for i, frame in enumerate(video_frames):
        t0 = time.perf_counter()
        masks, boxes, scores, concept_indices = client.segment_concepts(frame.transpose(2, 0, 1), ["person", "bed"])
        torch.cuda.synchronize()
        print(
            f"frame {i}: {masks.shape[0]} objects, ids={client.object_ids.tolist()}, "
            f"concepts={concept_indices.tolist()}, {(time.perf_counter() - t0) * 1000:.1f} ms"
        )
