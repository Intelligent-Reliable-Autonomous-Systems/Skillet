import argparse
import os

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Save the first and last images from an rgb.npy file.")
    parser.add_argument("--f", help="Path to rgb.npy")
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to save the PNGs (default: current directory)",
    )
    args = parser.parse_args()

    # Load the RGB array
    rgb = np.load(args.f)

    if len(rgb) == 0:
        raise ValueError("The array is empty.")

    first = rgb[0]
    last = rgb[-1]

    first = np.transpose(rgb[0], (1, 2, 0))
    last = np.transpose(rgb[-1], (1, 2, 0))

    # Convert RGB -> BGR for OpenCV if the images have 3 color channels
    if first.ndim == 3 and first.shape[-1] == 3:
        first = cv2.cvtColor(first, cv2.COLOR_RGB2BGR)
        last = cv2.cvtColor(last, cv2.COLOR_RGB2BGR)

    os.makedirs(args.output_dir, exist_ok=True)

    first_path = os.path.join(args.output_dir, "first.png")
    last_path = os.path.join(args.output_dir, "last.png")

    cv2.imwrite(first_path, first)
    cv2.imwrite(last_path, last)

    print(f"Saved first image to: {first_path}")
    print(f"Saved last image to:  {last_path}")


if __name__ == "__main__":
    main()
