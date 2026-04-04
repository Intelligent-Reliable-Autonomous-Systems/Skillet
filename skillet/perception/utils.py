from functools import cache
from pathlib import Path


@cache
def get_skillet_model_cache_dir() -> Path:
    """Return the cache directory within skillet."""
    import skillet

    top_level_dir = Path(skillet.__file__).parent
    cache_dir = top_level_dir / "data" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
