import base64
import io
import platform
import subprocess
import time

import numpy as np
import requests
import torch
from PIL import Image

from skillet.perception.segmentation.vlm.vlm_base import VLMClient


class QwenClient(VLMClient):
    """Qwen client using Ollama."""

    def __init__(
        self,
        prompt_name: str = "detect_goal_qwen",
        model_id: str | None = "qwen3.5:9b",
        device: str = "cuda",
        host: str = "http://localhost:11434",
    ) -> None:
        """Initialize the Qwen client."""
        super().__init__(prompt_name, model_id, device)

        self.host = host

        if not self._is_server_running():
            print("[INFO][OLLAMA] Ollama server not running, starting it...")
            self._start_server()
            self._wait_for_server()
            print("[INFO][OLLAMA] Ollama server ready.")

    def _is_server_running(self) -> bool:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=3)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def _start_server(self) -> None:
        """Start the Ollama server."""
        system = platform.system()
        assert system == "Linux"
        subprocess.Popen(
            ["gnome-terminal", "--", "ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def _wait_for_server(self, timeout: int = 30, interval: float = 1.0) -> None:
        """Wait for the llama server to start."""
        start = time.time()
        while time.time() - start < timeout:
            if self._is_server_running():
                return
            time.sleep(interval)
        raise TimeoutError(f"Ollama server did not start within {timeout} seconds.")

    def chat(self, message: str) -> str:
        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model_id,
                "messages": [{"role": "user", "content": message}],
                "stream": False,
                "options": {
                    "temperature": 0,  # no randomness
                    "top_p": 1,  # disable nucleus sampling
                    "top_k": 1,  # always pick the single most likely token
                    "seed": 42,  # fixed seed for reproducibility
                },
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def chat_with_image(self, message: str, image: Image.Image | np.ndarray | torch.Tensor) -> str:
        """Query the VLM with a message and image."""
        if isinstance(image, Image.Image):
            image_encode = self._encode_image_pil(image)
        elif isinstance(image, (np.ndarray, torch.Tensor)):
            image_encode = self._encode_image_arr(image)

        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model_id,
                "messages": [{"role": "user", "content": message, "images": [image_encode]}],
                "stream": False,
                "options": {"temperature": 0, "top_p": 1, "top_k": 1, "seed": 42, "num_ctx": 4096},
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _encode_image_arr(self, image: np.ndarray | torch.Tensor, max_size: int = 512) -> str:
        """Encode a np.ndarray image into a .jpeg for ollama."""
        # convert BGR -> RGB if 3 channel (OpenCV default)
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1]  # BGR -> RGB

        pil_image = Image.fromarray(image.astype(np.uint8))
        pil_image.thumbnail((max_size, max_size))

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _encode_image_pil(self, image: Image.Image, max_size: int = 512) -> str:
        """Encode a PIL image into a .jpef for ollama."""
        image.thumbnail((max_size, max_size))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def detect_bboxes(self, image: np.ndarray, task_instruction: str) -> str:
        """Detect the image and translate to bounding boxes."""
        return self.parse_response(self.chat_with_image(task_instruction, image))

    def parse_response(self, response_text: str) -> tuple[list, list]:
        """Parse Qwen response text into bboxes, grounded goal atoms, and grounded scene atoms."""
        try:
            result = self._load_json(response_text)
        except Exception:
            raise ValueError(
                f"Qwen returned a non-JSON response; check for a discrepancy in your image: {response_text}"
            )

        return self._parse_response(result)

    def _parse_response(self, result: list | dict) -> tuple[list, list, list]:
        """Parse the response from the VLM. Assumes a prompt in ./prompts/.

        Args:
            result: JSON formatted result from VLM

        Returns:
            Lists of bounding boxes

        """
        bboxes = []
        labels = []
        for o in result:
            bboxes.append(o["bbox_2d"])
            labels.append(o["label"])

        return np.asarray(bboxes), np.asarray(labels)
