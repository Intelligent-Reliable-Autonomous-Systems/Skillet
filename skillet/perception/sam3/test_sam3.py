import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
except Exception as import_error:  # pragma: no cover
    print("Failed to import Ultralytics SAM3. Please install ultralytics:", file=sys.stderr)
    print("  pip install ultralytics", file=sys.stderr)
    raise import_error


def _default_repo_root() -> Path:
    """Repo root: skillet/perception/sam3/test_sam3.py -> skillet (or workspace root)."""
    return Path(__file__).resolve().parents[3]


def _default_model_path() -> Path:
    """Path to SAM3 weights (data/models/sam3.pt)."""
    return _default_repo_root() / "data" / "models" / "sam3.pt"


def _default_image_path() -> Path:
    """Default image: first image in data/images, or data/images/image.jpg."""
    images_dir = _default_repo_root() / "data" / "images"
    if images_dir.is_dir():
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                return p
    return images_dir / "image.jpg"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Ultralytics SAM3 model with text prompts and visualize the results."
    )
    default_model = _default_model_path()
    default_image = _default_image_path()

    parser.add_argument(
        "--source",
        type=str,
        default=str(default_image),
        help="Path to image or video file. Default: first image in data/images or data/images/image.jpg.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["bag of groceries"],
        help="One or more text prompts describing target(s).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(default_model),
        help="Path to SAM3 model (.pt). Default: data/models/sam3.pt.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use, e.g., 'cuda:0' or 'cpu'. Empty lets Ultralytics decide.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Use FP16 for faster inference when supported.",
    )
    parser.add_argument(
        "--no-half",
        dest="half",
        action="store_false",
        help="Disable FP16.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show visualization window(s).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save visualization to disk.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_default_repo_root() / "data" / "runs" / "sam3"),
        help="Output directory for saved results.",
    )
    return parser.parse_args()


def is_video_path(path: str | os.PathLike) -> bool:
    lowercase = str(path).lower()
    # Common video extensions
    return any(lowercase.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"])


def initialize_predictor(
    model_path: str,
    conf: float,
    half: bool,
    device: str,
) -> SAM3SemanticPredictor:
    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=model_path,
        half=half,
        device=device or None,
        save=False,  # We'll handle saving ourselves
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    return predictor


def results_to_image(results) -> np.ndarray | None:
    """Convert SAM3 results to a BGR image using the object's plot() method when available.
    Returns None if plotting is not supported.
    """
    try:
        # Ultralytics Results is iterable; handle list or single object
        if isinstance(results, (list, tuple)):
            # Concatenate vertically if multiple; usually single item for one image
            rendered: list[np.ndarray] = []
            for res in results:
                if hasattr(res, "plot"):
                    plot_img = res.plot()  # BGR uint8
                    if plot_img is not None:
                        rendered.append(plot_img)
            if not rendered:
                return None
            return rendered[0] if len(rendered) == 1 else np.vstack(rendered)
        # Single Results-like object
        if hasattr(results, "plot"):
            return results.plot()
        return None
    except Exception:
        return None


def visualize_image(
    predictor: SAM3SemanticPredictor,
    source_path: str,
    prompts: Iterable[str],
    show: bool,
    save: bool,
    out_dir: str,
) -> None:
    # Accept either path string or numpy image for set_image
    predictor.set_image(source_path)
    results = predictor(text=list(prompts))

    vis = results_to_image(results)
    if vis is None:
        print("Warning: Could not render results automatically. Printing raw results:")
        print(results)
        return

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if save:
        output_path = out_dir_path / (Path(source_path).stem + "_sam3.png")
        cv2.imwrite(str(output_path), vis)
        print(f"Saved visualization to: {output_path}")

    if show:
        cv2.imshow("SAM3", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualize_video(
    predictor: SAM3SemanticPredictor,
    source_path: str,
    prompts: Iterable[str],
    show: bool,
    save: bool,
    out_dir: str,
) -> None:
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_writer = None
    output_path = None
    if save:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(out_dir) / (Path(source_path).stem + "_sam3.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Set current frame as input image
            predictor.set_image(frame)
            results = predictor(text=list(prompts))
            vis = results_to_image(results)
            if vis is None:
                # Fallback to showing the original frame if plotting fails
                vis = frame

            if show:
                cv2.imshow("SAM3 (video)", vis)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break

            if out_writer is not None:
                out_writer.write(vis)
    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if show:
            cv2.destroyAllWindows()

    if save and output_path is not None:
        print(f"Saved video visualization to: {output_path}")


def main() -> None:
    args = parse_arguments()

    if not Path(args.model).exists():
        print(f"Error: Model weights not found: {args.model}", file=sys.stderr)
        print("  Place sam3.pt in data/models/ or pass --model /path/to/sam3.pt", file=sys.stderr)
        sys.exit(1)
    if not Path(args.source).exists():
        print(f"Error: Source not found: {args.source}", file=sys.stderr)
        print("  Add an image to data/images/ or pass --source /path/to/image.jpg", file=sys.stderr)
        sys.exit(1)

    source_path = args.source
    prompts = args.prompts
    predictor = initialize_predictor(
        model_path=args.model,
        conf=args.conf,
        half=args.half,
        device=args.device,
    )

    Path(args.out_dir).mkdir(exist_ok=True, parents=True)
    if is_video_path(source_path):
        visualize_video(
            predictor=predictor,
            source_path=source_path,
            prompts=prompts,
            show=args.show,
            save=args.save,
            out_dir=args.out_dir,
        )
    else:
        visualize_image(
            predictor=predictor,
            source_path=source_path,
            prompts=prompts,
            show=args.show,
            save=args.save,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    """
    Example usages:
      - Image:
        python sam3/sam3_test.py --source /home/jjewett/project/perception/Grounded-SAM-2/notebooks/images/groceries.jpg --prompts "bag of groceries"

      - Video:
        python sam3/sam3_test.py --source /home/jjewett/project/perception/Grounded-SAM-2/assets/hippopotamus.mp4 --prompts "hippopotamus"

      - Custom model and show window:
        python sam3/sam3_test.py --model /path/to/sam3.pt --source /path/to/img.jpg --prompts "a person" --show
    """
    main()
