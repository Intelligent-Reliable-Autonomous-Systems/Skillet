"""Benchmark SAM segmentation latency while visualizing predictions."""

import argparse
import time
from collections import deque
from typing import Literal

import cv2
import numpy as np
import torch

from skillet.envs.realsense import RealsenseEnv
from skillet.perception.segmentation.sam import get_sam_client, SAMClient
from skillet.scene.utils import depth_to_colormap_np

# Distinct BGR colors for mask overlays (high contrast on typical scenes).
_MASK_COLORS_BGR: list[tuple[int, int, int]] = [
    (0, 165, 255),
    (255, 144, 30),
    (147, 20, 255),
    (0, 255, 127),
    (255, 0, 255),
    (0, 215, 255),
    (180, 105, 255),
    (204, 72, 63),
]


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_mem_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024**2)


def overlay_masks_bgr(
    rgb_bgr: np.ndarray,
    masks: torch.Tensor,
    scores: torch.Tensor,
    *,
    concept_indices: torch.Tensor | None = None,
    concepts: list[str] | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend segmentation masks onto a BGR image and draw short labels."""
    out = rgb_bgr.astype(np.float32).copy()
    m = masks.detach().float().cpu().numpy()
    if m.ndim == 4:
        m = np.squeeze(m, axis=1)
    s = scores.detach().cpu().numpy().reshape(-1)
    ci = concept_indices.detach().cpu().numpy().reshape(-1) if concept_indices is not None else None

    h, w = out.shape[:2]
    for i in range(m.shape[0]):
        mi = np.clip(m[i], 0.0, 1.0)
        if mi.shape != (h, w):
            mi = cv2.resize(mi, (w, h), interpolation=cv2.INTER_LINEAR)
        col = np.array(_MASK_COLORS_BGR[i % len(_MASK_COLORS_BGR)], dtype=np.float32)
        for c in range(3):
            out[:, :, c] = out[:, :, c] * (1.0 - alpha * mi) + col[c] * (alpha * mi)

        label_parts = [f"{i}"]
        if ci is not None and concepts and 0 <= int(ci[i]) < len(concepts):
            label_parts.append(concepts[int(ci[i])])
        label_parts.append(f"{float(s[i]):.2f}")
        label = " | ".join(label_parts)
        ys, xs = np.where(mi > 0.5)
        if xs.size > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(
                out,
                label,
                (max(4, cx - 80), max(16, cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return np.clip(out, 0, 255).astype(np.uint8)


def draw_stats_panel(
    img_bgr: np.ndarray,
    *,
    latency_ms: float,
    rolling_mean_ms: float,
    rolling_min_ms: float,
    rolling_max_ms: float,
    frame_idx: int,
    rolling_samples: int,
    rolling_window: int,
    gpu_mb: float | None,
    warmup_remaining: int,
) -> None:
    """Draw a semi-transparent stats strip on the top-left of `img_bgr` (in place)."""
    lines = [
        f"frame {frame_idx}",
        f"last: {latency_ms:.1f} ms",
        (
            f"mean/min/max (n={rolling_samples}/{rolling_window}): "
            f"{rolling_mean_ms:.1f} / {rolling_min_ms:.1f} / {rolling_max_ms:.1f} ms"
        ),
        f"fps (mean): {1000.0 / rolling_mean_ms:.1f}" if rolling_mean_ms > 1e-6 else "fps: —",
    ]
    if gpu_mb is not None:
        lines.append(f"GPU mem: {gpu_mb:.0f} MB")
    if warmup_remaining > 0:
        lines.append(f"warmup: {warmup_remaining} frames")

    margin = 8
    line_h = 18
    pad = 6
    w = max(len(t) for t in lines) * 8 + 2 * pad
    h = len(lines) * line_h + 2 * pad
    roi = img_bgr[margin : margin + h, margin : margin + w]
    overlay = roi.copy()
    overlay[:] = (40, 40, 40)
    cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)
    for j, text in enumerate(lines):
        y = margin + pad + (j + 1) * line_h - 4
        cv2.putText(img_bgr, text, (margin + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)


def main(
    model: Literal["sam2", "sam3"],
    mode: Literal["text", "bboxes"] = "text",
    *,
    rolling_window: int = 30,
    warmup_frames: int = 3,
) -> None:
    """Run live SAM benchmark with RGB/depth view and mask overlay."""
    env = RealsenseEnv()
    print("RealsenseEnv initialized")
    sam_model: SAMClient = get_sam_client(model)(use_server=True)
    print("Sam model loaded")

    latencies_ms: deque[float] = deque(maxlen=rolling_window)
    frame_idx = 0

    try:
        while True:
            obs = env.get_observation()
            rgb = obs["rgb"]

            _sync_cuda()
            t0 = time.perf_counter()
            if mode == "text":
                concepts = ["block", "robot arm"]
                masks, boxes, scores, concept_indices = sam_model.segment_concepts(rgb, concepts)
                _sync_cuda()
                t1 = time.perf_counter()
                vis_rgb = overlay_masks_bgr(
                    cv2.cvtColor(rgb.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR),
                    masks,
                    scores,
                    concept_indices=concept_indices,
                    concepts=concepts,
                )
            elif mode == "bboxes":
                bboxes = [
                    [100, 100, 150, 150],
                    [200, 200, 250, 250],
                    [300, 300, 350, 350],
                    [200, 100, 250, 150],
                    [100, 200, 150, 250],
                ]
                masks, scores = sam_model.segment_from_bboxes(rgb, bboxes)
                _sync_cuda()
                t1 = time.perf_counter()
                vis_rgb = overlay_masks_bgr(
                    cv2.cvtColor(rgb.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR),
                    masks,
                    scores,
                )
            else:
                raise ValueError(f"Invalid mode: {mode}")

            latency_ms = (t1 - t0) * 1000.0
            frame_idx += 1
            in_warmup = frame_idx <= warmup_frames
            if not in_warmup:
                latencies_ms.append(latency_ms)

            if latencies_ms:
                arr = np.array(latencies_ms, dtype=np.float64)
                rmean = float(arr.mean())
                rmin = float(arr.min())
                rmax = float(arr.max())
            else:
                rmean = rmin = rmax = latency_ms

            draw_stats_panel(
                vis_rgb,
                latency_ms=latency_ms,
                rolling_mean_ms=rmean,
                rolling_min_ms=rmin,
                rolling_max_ms=rmax,
                frame_idx=frame_idx,
                rolling_samples=len(latencies_ms),
                rolling_window=rolling_window,
                gpu_mb=_gpu_mem_mb(),
                warmup_remaining=max(0, warmup_frames - frame_idx),
            )

            depth = depth_to_colormap_np(obs["depth"][0])
            combined = np.concatenate([vis_rgb, depth], axis=1)
            print(f"Latency: {latency_ms:.2f} ms, FPS: {1000.0 / latency_ms:.2f}, GPU mem: {_gpu_mem_mb():.0f} MB")
            # cv2.imshow("SAM benchmark (segmentation | depth) — q to quit", combined)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
    finally:
        env.close()
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
    parser.add_argument("--rolling", type=int, default=30, help="Frames for rolling mean/min/max latency.")
    parser.add_argument("--warmup", type=int, default=3, help="Frames to skip before rolling stats fill.")
    args = parser.parse_args()
    main(args.model, args.mode, rolling_window=args.rolling, warmup_frames=args.warmup)
