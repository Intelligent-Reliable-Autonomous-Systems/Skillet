from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from skillet.perception.segmentation.vlm.vlm_base import VLMClient

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class QwenClient(VLMClient):
    """Local Qwen (VL) client for detection + task grounding."""

    def __init__(
        self,
        prompt_name: str = "detect_and_translate",
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "cuda",
    ) -> None:
        super().__init__(prompt_name, model_id, device)
        print(f"[INFO][QWEN] Loading Qwen Model with checkpoint={model_id}, device={self.device}")
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
        ).eval()

    def detect_and_translate(
        self,
        image: Image.Image,
        task_instruction: str,
    ) -> tuple[list[dict], list[dict]]:
        """Run multimodal inference locally."""
        prompt = self.prompt.format(task_instruction=task_instruction)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template (important for Qwen)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
            )

        generated = self.processor.batch_decode(output, skip_special_tokens=True)[0]

        return self.parse_response(generated)

    def parse_response(self, response_text: str) -> tuple[list, list, list]:
        """Parse model output into structured format."""
        try:
            parsed_response = response_text.split("\nassistant\n", 1)[1]
            result = self._load_json(parsed_response)
        except Exception:
            raise ValueError(f"Qwen returned non-JSON output: {response_text}")

        return self._parse_response(result)
