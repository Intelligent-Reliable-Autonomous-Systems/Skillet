import json
from abc import ABC, abstractmethod
from functools import cache
from pathlib import Path

from PIL import Image

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

    @abstractmethod
    def detect_and_translate(
        self,
        image: Image.Image,
        task_instruction: str,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Detect objects and translate task in a single Gemini API call.

        Args:
            image: The image to analyze.
            task_instruction: The natural language task to translate.

        Returns:
            Tuple of (bboxes, grounded_atoms) where:
            - bboxes: List of detected objects with bounding boxes
            - grounded_atoms: List of predicate specifications

        """
        raise NotImplementedError

    @abstractmethod
    async def detect_and_translate_async(
        self,
        image: Image.Image,
        task_instruction: str,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Detect objects and translate task in a single Gemini API call asycronously.

        Args:
            image: The image to analyze.
            task_instruction: The natural language task to translate.

        Returns:
            Tuple of (bboxes, grounded_atoms) where:
            - bboxes: List of detected objects with bounding boxes
            - grounded_atoms: List of predicate specifications

        """
        raise NotImplementedError

    @abstractmethod
    def parse_response(self) -> tuple[list, list, list]:
        """Parse the response on a per VLM client basis."""
        raise NotImplementedError

    def _parse_response(self, result: list | dict) -> tuple[list, list, list]:
        """Parse the response from the VLM. Assumes a prompt in ./prompts/.

        Args:
            result: JSON formatted result from VLM

        Returns:
            Lists of bounding boxes, goal predicates, and scene predicates.

        """
        bboxes = result.get("bboxes", [])
        grounded_goal_atoms = [
            {"goal_predicate": spec["name"], "args": spec["args"]}
            for spec in result.get("goal_predicates", [])
            if spec.get("name") and spec.get("args")
        ]
        grounded_scene_atoms = [
            {"scene_predicate": spec["name"], "args": spec["args"]}
            for spec in result.get("scene_predicates", [])
            if spec.get("name") and spec.get("args")
        ]
        return bboxes, grounded_goal_atoms, grounded_scene_atoms

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
