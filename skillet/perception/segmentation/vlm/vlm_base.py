import json
from abc import ABC
from functools import cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class VLMClient(ABC):
    """Base class for VLM Client Interface."""

    def __init__(
        self,
        prompt_name: str = "detect_and_translate",
        model_id: str | None = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the base VLM."""
        self.prompt = self._load_prompt(prompt_name)
        self.device = device
        self.model_id = model_id

    @cache
    def _load_prompt(self, prompt_name: str) -> str:
        """Load a prompt template from the prompts directory."""
        return (_PROMPTS_DIR / f"{prompt_name}.txt").read_text().strip()

    def _load_json(self, response_text: str) -> list | dict:
        """Extract JSON string from code fencing if present."""
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.replace("```", "")

        try:
            results = json.loads(cleaned_text)
        except json.decoder.JSONDecodeError:
            print(f"[ERROR][VLM]Invalid JSON: {cleaned_text}")
            raise
        return results
