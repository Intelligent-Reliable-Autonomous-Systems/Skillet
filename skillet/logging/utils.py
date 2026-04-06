import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def filled_rect(img: np.ndarray, x: int, y: int, w: int, h: int, color: float, radius: int = 6) -> None:
    """Return a filled rectange with curved corners.

    Draws directly on image.

    Args:
        img: np.ndarray of the image to be displayed
        x: x coordinate to start drawing rect
        y: y coordinate to start drawing rect
        w: width of rectangle
        h: height of rectangle
        color: color of rectange
        radius: minimum radius of rectangle

    """
    r = min(radius, w // 2, h // 2)
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in [(x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)]:
        cv2.circle(img, (cx, cy), r, color, -1)


def outline_rect(
    img: np.ndarray, x: int, y: int, w: int, h: int, color: float, thickness: int = 1, radius: int = 6
) -> None:
    """Outline a rectange in the image.

    Args:
        img: np.ndarray of the image to be displayed
        x: x coordinate to start drawing rect
        y: y coordinate to start drawing rect
        w: width of rectangle
        h: height of rectangle
        color: color of rectange
        thickness: thickness of line
        radius: minimum radius of rectangle

    """
    r = min(radius, w // 2, h // 2)
    pts = np.array(
        [
            [x + r, y],
            [x + w - r, y],
            [x + w, y + r],
            [x + w, y + h - r],
            [x + w - r, y + h],
            [x + r, y + h],
            [x, y + h - r],
            [x, y + r],
        ],
        np.int32,
    )
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


def tw(s: str, scale: float, thick: int = 1) -> int:
    """Get the specific text size.

    Args:
        s: string of text
        scale: scale of the text size
        thick: thickness of the text

    """
    (w, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    return w


def put(img: np.ndarray, s: str, x: int, y: int, color: float, scale: float = 0.45, thick: int = 1) -> None:
    """Put text onto the cv2 window.

    Args:
        img: np.ndarray of the image to be displayed
        s: string of text to put on the image
        x: x coordinate to start drawing text
        y: y coordinate to start drawing text
        color: color of rectange
        scale: scale to draw the text at
        thick: thickness of line

    """
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def put_c(
    img: np.ndarray, s: str, cx: int, y: int, color: tuple[int, int, int], scale: float = 0.45, thick: int = 1
) -> None:
    """Draw text horizontally centered at pixel column cx, baseline at y."""
    put(img, s, cx - tw(s, scale, thick) // 2, y, color, scale, thick)


def pill(
    img: np.ndarray,
    s: str,
    x: int,
    y: int,
    fg: tuple[int, int, int],
    bg: tuple[int, int, int],
    scale: float = 0.38,
    px: int = 8,
    py: int = 4,
) -> int:
    """Draw text s inside a filled pill badge.

    The pill's left edge starts at x; the text baseline sits at y.
    px / py control horizontal and vertical inner padding.
    Returns the x coordinate immediately after the pill's right edge,
    so callers can chain pills in a row without manual spacing.
    """
    (t_w, t_h), _ = cv2.getTextSize(s, FONT, scale, 1)
    bw, bh = t_w + px * 2, t_h + py * 2
    filled_rect(img, x, y - bh + py, bw, bh, bg, radius=4)
    cv2.putText(img, s, (x + px, y + py - 1), FONT, scale, fg, 1, cv2.LINE_AA)
    return x + bw + 6
