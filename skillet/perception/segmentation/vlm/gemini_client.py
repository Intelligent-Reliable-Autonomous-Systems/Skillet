from functools import cache

from google import genai
from google.genai import types
from PIL import Image

from skillet.perception.segmentation.vlm.vlm_base import VLMClient


class GeminiClient(VLMClient):
    """Class for Google Gemini Client."""

    def __init__(
        self,
        prompt_name: str = "detect_goal",
        model_id: str = "gemini-robotics-er-1.5-preview",  # gemini-2.5-flash",
        device: str = "cuda",
    ) -> None:
        super().__init__(prompt_name, model_id, device)

        self.prompt = self._load_prompt(prompt_name)
        self.client = self.gemini_client()

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
        prompt = self.prompt.format(task_instruction=task_instruction)
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                temperature=None, thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        return self.parse_response(response.text)

    async def detect_and_translate_async(
        self,
        image: Image.Image,
        task_instruction: str,
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
        prompt = self.prompt.format(task_instruction=task_instruction)
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                temperature=None, thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        return self.parse_response(response.text)

    def parse_response(self, response_text: str) -> tuple[list, list]:
        """Parse Gemini response text into bboxes, grounded goal atoms, and grounded scene atoms."""
        try:
            result = self._load_json(response_text)
        except Exception:
            raise ValueError(
                f"Gemini returned a non-JSON response; check for a discrepancy in your image: {response_text}"
            )

        return self._parse_response(result)

    @cache
    def gemini_client(self) -> genai.Client:
        """Return the gemini client class."""
        return genai.Client(api_key="AIzaSyB0_pZAS3obaSyjnri2TtRqvsUDzH7pt5g")
