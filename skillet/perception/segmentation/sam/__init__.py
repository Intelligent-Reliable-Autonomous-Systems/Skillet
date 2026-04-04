"""SAM clients for segmentation."""
from typing import Literal

from skillet.perception.segmentation.sam.sam_base import SAMClient


def get_sam_client(model: Literal["sam2", "sam3", "sam3_streaming"] = "sam3") -> SAMClient:
    """Get a SAM client."""
    if model == "sam2":
        from .sam2_client import SAM2Client as SAM2Client
        return SAM2Client()
    if model == "sam3":
        from .sam3_client import SAM3Client as SAM3Client
        return SAM3Client()
    if model == "sam3_streaming":
        from .sam3_streaming_client import SAM3StreamingClient as SAM3StreamingClient
        return SAM3StreamingClient()
    if model == "sam3_ultralytics":
        # return SAM3Ultralytics()
        raise NotImplementedError("SAM3 Ultralytics was removed")
    raise ValueError(f"Invalid SAM model: {model}")
