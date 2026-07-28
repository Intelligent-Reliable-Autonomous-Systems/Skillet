"""
Extract the first frame of an MP4 video and save it as a PNG.

Requirements:
    pip install opencv-python

Usage:
    python extract_first_frame.py input.mp4 output.png
"""

import sys
import cv2


def extract_first_frame(video_path: str, output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"Could not read a frame from: {video_path}")

    cv2.imwrite(output_path, frame)
    print(f"Saved first frame to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_first_frame.py <input.mp4> <output.png>")
        sys.exit(1)

    extract_first_frame(sys.argv[1], sys.argv[2])
