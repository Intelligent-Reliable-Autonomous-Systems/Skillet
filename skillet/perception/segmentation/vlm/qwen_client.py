import json
from functools import cache
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class QwenClient:
    """Local Qwen (VL) client for detection + task grounding."""

    def __init__(
        self,
        prompt_name: str = "detect_and_translate",
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.prompt = self._load_prompt(prompt_name)

        self.device = device
        print(f"[INFO][QWEN] Loading Qwen Model with checkpoint={model_id}, device={self.device}")
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
        ).eval()

    def detect_and_translate(
        self,
        image: Image.Image,
        task_instruction: str,
        temperature: float | None = None,
        max_new_tokens: int = 512,
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
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature is not None else 0.0,
                do_sample=temperature is not None,
            )

        generated = self.processor.batch_decode(output, skip_special_tokens=True)[0]

        return self._parse_response(generated)

    def _parse_response(self, response_text: str) -> tuple[list, list]:
        """Parse model output into structured format."""
        try:
            parsed_response = response_text.split("\nassistant\n", 1)[1]
            result = self._load_json(parsed_response)
        except Exception:
            raise ValueError(f"Qwen returned non-JSON output: {response_text}")

        bboxes = result.get("bboxes", [])
        grounded_atoms = [
            {"predicate": spec["name"], "args": spec["args"]}
            for spec in result.get("predicates", [])
            if spec.get("name") and spec.get("args")
        ]

        return bboxes, grounded_atoms

    def _load_json(self, response_text: str):
        """Strip code fences and load JSON."""
        cleaned = response_text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "")
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "")

        return json.loads(cleaned)

    @cache
    def _load_prompt(self, prompt_name: str) -> str:
        return (_PROMPTS_DIR / f"{prompt_name}.txt").read_text().strip()
