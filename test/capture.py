"""
Capture a single color frame from an Intel RealSense camera and save it as a PNG.

Requirements:
    pip install pyrealsense2 opencv-python numpy
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
args = parser.parse_args()
OUTPUT_PATH = f"{args.name}.png"
WARMUP_FRAMES = 30  # let auto-exposure settle before saving


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable the color stream (adjust resolution/fps if your camera doesn't support this combo)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except RuntimeError as e:
        print(f"Failed to start RealSense pipeline: {e}")
        print("Make sure the camera is plugged in and not in use by another process.")
        sys.exit(1)

    try:
        # Grab a few frames first so exposure/white balance can adjust
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            print("No color frame received.")
            sys.exit(1)

        color_image = np.asanyarray(color_frame.get_data())

        cv2.imwrite(OUTPUT_PATH, color_image)
        print(f"Saved image to {OUTPUT_PATH}")

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
