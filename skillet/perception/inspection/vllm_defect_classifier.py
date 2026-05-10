"""VLLMDefectClassifier — binary defect detection via OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import base64
import math

import cv2
import numpy as np

from skillet.perception.inspection.defect_classifier import DefectClassifier, DefectResult

_SYSTEM_PROMPT = (
    "You are a visual defect inspector for a robot manipulation task. "
    "Given an image of a block, determine whether it has a visible defect "
    "(e.g. a crack, dark patch, or high-contrast mark). "
    "Respond with a single word: YES if defective, NO if not defective."
)


class VLLMDefectClassifier(DefectClassifier):
    """Calls a remote vLLM server via the OpenAI-compatible chat-completions API.

    Confidence is derived from the log probability of the first generated token
    (YES or NO) rather than a self-reported score, which gives a more calibrated
    signal from the model's own probability distribution.

    The server can be a locally port-forwarded HPC endpoint:
        ssh -L 8000:localhost:8000 hpc_host
    Then set base_url="http://localhost:8000/v1".
    """

    def __init__(self, base_url: str, model: str, api_key: str = "none") -> None:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "VLLMDefectClassifier requires the 'openai' package. "
                "Install it with: pip install 'skillet[vlm]'"
            ) from exc
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def classify(self, image: np.ndarray, object_id: str) -> DefectResult:
        """Send the wrist-camera image to the vLLM endpoint and return a binary result."""
        b64 = self._encode_image(image)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": f"Does block '{object_id}' have a visible defect?",
                        },
                    ],
                },
            ],
            max_tokens=3,
            logprobs=True,
            top_logprobs=1,
        )
        choice = response.choices[0]
        token_text = (choice.message.content or "").strip()
        logprob = choice.logprobs.content[0].logprob
        return self._parse_result_from(token_text, logprob)

    @staticmethod
    def _parse_result_from(token_text: str, logprob: float) -> DefectResult:
        """Convert the model's first token and its log probability into a DefectResult.

        exp(log p) = p, so the token's log probability gives the model's own
        probability estimate — a more principled confidence than a self-reported score.
        """
        defective = token_text.strip().upper().startswith("Y")
        confidence = math.exp(logprob)
        return DefectResult(defective=defective, confidence=confidence)

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """Encode a numpy BGR image as a base64 PNG string."""
        _, buf = cv2.imencode(".png", image)
        return base64.b64encode(buf.tobytes()).decode()
