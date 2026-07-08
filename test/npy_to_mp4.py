#!/usr/bin/env python3
"""
Convert a .npy file containing a stack of RGB images into an .mp4 video.

Expected input shape: (num_frames, height, width, 3), dtype uint8 (0-255).
If your array is float (0-1) it will be automatically scaled to uint8.

Usage:
    python npy_to_mp4.py input.npy output.mp4 --fps 30
"""

import argparse
import sys
import numpy as np
import cv2


def load_frames(npy_path: str) -> np.ndarray:
    frames = np.load(npy_path)
    if frames.shape[-1] != 3:
        frames = np.transpose(frames, (0, 2, 3, 1))
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected array of shape (N, H, W, 3), got {frames.shape}")

    # Scale float arrays (assumed 0-1 range) to uint8
    if np.issubdtype(frames.dtype, np.floating):
        if frames.max() <= 1.0:
            frames = frames * 255.0
        frames = frames.clip(0, 255).astype(np.uint8)
    elif frames.dtype != np.uint8:
        frames = frames.clip(0, 255).astype(np.uint8)

    return frames


def write_video(frames: np.ndarray, output_path: str, fps: int) -> None:
    num_frames, height, width, _ = frames.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    for i in range(num_frames):
        # cv2 expects BGR, input frames are RGB
        bgr_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

    writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Path to input .npy file")
    parser.add_argument("--output", help="Path to output .mp4 file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    args = parser.parse_args()

    try:
        frames = load_frames(args.input)
        print(f"Loaded array: shape={frames.shape}, dtype={frames.dtype}")

        write_video(frames, args.output, args.fps)
        print(f"Saved video to {args.output} ({frames.shape[0]} frames at {args.fps} fps)")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
