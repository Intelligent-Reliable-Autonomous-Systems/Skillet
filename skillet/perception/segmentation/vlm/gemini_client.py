import json
from functools import cache
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class GeminiClient:
    """Class for Google Gemini Client."""

    def __init__(self, prompt_name: str = "detect_and_translate") -> None:
        self.prompt = self._load_prompt(prompt_name)

    def detect_and_translate(
        self,
        image: Image.Image,
        task_instruction: str,
        client: genai.Client | None = None,
        model_id: str = "gemini-robotics-er-1.5-preview",
        temperature: float | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Detect objects and translate task in a single Gemini API call.

        Args:
            image: The image to analyze.
            task_instruction: The natural language task to translate.
            client: Gemini API client. If None, a new client will be created.
            model_id: Gemini model ID to use.
            temperature: Temperature for generation.

        Returns:
            Tuple of (bboxes, grounded_atoms) where:
            - bboxes: List of detected objects with bounding boxes
            - grounded_atoms: List of predicate specifications

        """
        client = client or self.gemini_client()
        prompt = self.prompt.format(task_instruction=task_instruction)
        response = client.models.generate_content(
            model=model_id,
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        return self._parse_response(response.text)

    async def detect_and_translate_async(
        self,
        image: Image.Image,
        task_instruction: str,
        client: genai.Client | None = None,
        model_id: str = "gemini-robotics-er-1.5-preview",
        temperature: float | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Asynchronously detect objects and translate task in a single Gemini API call.

        Args:
            image: The image to analyze.
            task_instruction: The natural language task to translate.
            client: Gemini API client. If None, a new client will be created.
            model_id: Gemini model ID to use.
            temperature: Temperature for generation.

        Returns:
            Tuple of (bboxes, grounded_atoms) where:
            - bboxes: List of detected objects with bounding boxes.
            - grounded_atoms: List of predicate specifications.

        """
        client = client or self.gemini_client()
        prompt = self.prompt.format(task_instruction=task_instruction)
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        return self._parse_response(response.text)

    def _parse_response(self, response_text: str) -> tuple[list, list]:
        """Parse Gemini response text into bboxes and grounded atoms."""
        try:
            result = self._load_json(response_text)
        except Exception:
            raise ValueError(
                f"Gemini returned a non-JSON response; check for a discrepancy in your image: {response_text}"
            )
        bboxes = result.get("bboxes", [])
        grounded_atoms = [
            {"predicate": spec["name"], "args": spec["args"]}
            for spec in result.get("predicates", [])
            if spec.get("name") and spec.get("args")
        ]
        return bboxes, grounded_atoms

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
            print(f"[ERROR][GEMINI]Invalid JSON: {cleaned_text}")
            raise
        return results

    @cache
    def _load_prompt(self, prompt_name: str) -> str:
        """Load a prompt template from the prompts directory."""
        return (_PROMPTS_DIR / f"{prompt_name}.txt").read_text().strip()

    @cache
    def gemini_client(self) -> genai.Client:
        """Return the gemini client class."""
        return genai.Client(api_key="AIzaSyB0_pZAS3obaSyjnri2TtRqvsUDzH7pt5g")
