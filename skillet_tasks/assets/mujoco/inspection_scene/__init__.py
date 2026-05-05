"""MuJoCo assets for the inspection pick-and-place scene."""

from __future__ import annotations

from pathlib import Path

_TEXTURE_DIR = Path(__file__).parent / "textures"


def load_texture(name: str) -> bytes:
    """Return the raw bytes of a texture file stored in the textures/ directory.

    Args:
        name: Filename of the texture (e.g. ``"defect_texture.png"``).

    Returns:
        Raw file bytes.

    """
    return (_TEXTURE_DIR / name).read_bytes()
