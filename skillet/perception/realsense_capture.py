"""
Capture color and depth images from an Intel RealSense camera using pyrealsense2.

Shows a live preview window — press SPACE to save a batch, press Q or ESC to quit.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import pyrealsense2 as rs


def build_pipeline(width: int, height: int, fps: int):
    """Start and return a configured RealSense pipeline + aligner."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)  # LEFT
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)  # RIGHT

    print("Starting RealSense pipeline...")
    try:
        profile = pipeline.start(config)
        for s in profile.get_streams():
            print(s)
    except RuntimeError as e:
        print(f"Failed to start pipeline: {e}")
        print("Is the camera plugged in and not in use by another process?")
        sys.exit(1)

    device = profile.get_device()
    print(f"  Device : {device.get_info(rs.camera_info.name)}")
    print(f"  Serial : {device.get_info(rs.camera_info.serial_number)}")

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    return pipeline, align, colorizer


def warmup(pipeline, n: int = 30):
    print(f"Warming up ({n} frames)...", end=" ", flush=True)
    for _ in range(n):
        pipeline.wait_for_frames()
    print("done.")


def get_frames(pipeline, align, colorizer):
    """Grab and align one frameset; return (color_image, depth_colormap, depth_frame)."""
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()

    ir_left = frames.get_infrared_frame(1)
    ir_right = frames.get_infrared_frame(2)

    if not color_frame or not depth_frame:
        return None, None, None

    color_image = np.asanyarray(color_frame.get_data())
    depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    ir_left_img = np.asanyarray(ir_left.get_data())
    ir_right_img = np.asanyarray(ir_right.get_data())

    return color_image, depth_colormap, depth_frame, ir_left_img, ir_right_img


def save_batch(
    color_image: np.ndarray,
    depth_colormap: np.ndarray,
    depth_frame: np.ndarray,
    ir_left: np.ndarray,
    ir_right: np.ndarray,
    output_dir: Path,
) -> None:
    """Save a color + depth pair with a zero-padded batch index."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"capture_{ts}"
    (output_dir / f"{stem}").mkdir(exist_ok=True, parents=True)

    color_path = output_dir / f"{stem}" / "img_color.png"
    depth_path = output_dir / f"{stem}" / "img_depth.png"

    # ---- RAW NUMPY OUTPUTS (new) ----
    color_npy_path = output_dir / f"{stem}" / "np_color.npy"
    ir_left_path = output_dir / f"{stem}" / "np_ir_left.npy"
    ir_right_path = output_dir / f"{stem}" / "np_ir_right.npy"
    depth_npy_path = output_dir / f"{stem}" / "np_depth.npy"
    depth_m_path = output_dir / f"{stem}" / "np_depth_m.npy"

    # Raw arrays
    depth_raw = np.asanyarray(depth_frame.get_data())  # uint16 (depth in sensor units)

    # Convert to meters (important for ML models)
    depth_scale = depth_frame.get_units()  # e.g. 0.001 for mm → meters
    depth_meters = depth_raw.astype(np.float32) * depth_scale

    np.save(color_npy_path, color_image)
    np.save(depth_npy_path, depth_raw)
    np.save(depth_m_path, depth_meters)
    np.save(ir_left_path, ir_left)
    np.save(ir_right_path, ir_right)

    cv2.imwrite(str(color_path), color_image)
    cv2.imwrite(str(depth_path), depth_colormap)

    print(f"[{ts}] Saved -> {color_path.name}  |  {depth_path.name}")


def run(width: int, height: int, fps: int, warmup_frames: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline, align, colorizer = build_pipeline(width, height, fps)
    warmup(pipeline, warmup_frames)

    print("\nLive preview active.")
    print("  SPACE   — save current frame (color + depth)")
    print("  Q / ESC — quit\n")

    batch_index = 1
    window = "RealSense  |  SPACE = save  |  Q/ESC = quit"

    try:
        while True:
            color_image, depth_colormap, depth_frame, ir_left, ir_right = get_frames(pipeline, align, colorizer)
            if color_image is None:
                continue

            # Side-by-side preview with on-screen instructions
            preview = np.hstack([color_image, depth_colormap])
            cv2.putText(
                preview,
                "SPACE: save  |  Q/ESC: quit",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                f"Saved: {batch_index - 1}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                save_batch(color_image, depth_colormap, depth_frame, ir_left, ir_right, output_dir)
                batch_index += 1

            elif key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                print("Exiting.")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"Pipeline stopped. {batch_index - 1} batch(es) saved to '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Live RealSense preview — press SPACE to save a frame batch.")
    parser.add_argument("--output-dir", default="captures", help="Directory to save images (default: ./captures)")
    parser.add_argument("--width", type=int, default=640, help="Stream width  (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Stream height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate    (default: 30)")
    parser.add_argument("--warmup", type=int, default=30, help="Warm-up frames (default: 30)")
    args = parser.parse_args()

    run(
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
