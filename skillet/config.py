from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skillet.perception.segmentation.vlm.vlm_base import VLMClient


class VLMClientFactory:
    def __init__(self, client: Literal["gemini", "qwen"] = "gemini"):
        self._client_cls = client

    def get_client(self) -> "VLMClient":
        """Construct a VLMClient"""
        if self._client_cls == "gemini":
            from skillet.perception.segmentation.vlm.gemini_client import GeminiClient

            return GeminiClient()
        if self._client_cls == "qwen":
            from skillet.perception.segmentation.vlm.qwen_client import QwenClient

            return QwenClient()
        raise ValueError(f"Invalid client: {self._client_cls}")
