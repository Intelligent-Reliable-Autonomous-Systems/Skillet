from functools import cache

import numpy as np
import torch
from google import genai
from google.genai import types
from PIL import Image

from skillet.perception.segmentation.vlm.vlm_base import VLMClient


class GeminiClient(VLMClient):
    """Class for Google Gemini Client."""

    def __init__(
        self,
        prompt_name: str = "detect_goal",
        model_id: str = "gemini-robotics-er-1.6-preview",  # gemini-2.5-flash",
        device: str = "cuda",
    ) -> None:
        super().__init__(prompt_name, model_id, device)

        self.prompt = self._load_prompt(prompt_name)
        self.client = self.gemini_client()

    def detect_bboxes_and_goal(
        self,
        image: np.ndarray | torch.Tensor,
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
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        rgb_pil = Image.fromarray(image.transpose(1, 2, 0))
        rgb_pil_resized = rgb_pil.resize((800, int(800 * rgb_pil.size[1] / rgb_pil.size[0])), Image.Resampling.LANCZOS)
        prompt = self.prompt.format(task_instruction=task_instruction)
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[rgb_pil_resized, prompt],
            config=types.GenerateContentConfig(
                temperature=None, thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        return self.parse_response(response.text)

    def detect_goal(self, task_instruction: str | None = None, scene: str | None = None) -> str:
        """Parse a task goal to PDDL based on the task instruction."""
        if scene is not None:
            message = self.prompt.format(task_instruction=task_instruction, scene=scene)
        else:
            message = self.prompt.format(task_instruction=task_instruction)

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[message],
            config=types.GenerateContentConfig(
                temperature=None, thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        result = self.parse_response(response.text)
        return np.asarray([o["goal"] for o in result if "goal" in o])

    def parse_response(self, response_text: str) -> tuple[list, list, list]:
        """Parse Gemini response text into bboxes, grounded goal atoms, and grounded scene atoms."""
        try:
            result = self._load_json(response_text)
        except Exception:
            raise ValueError(
                f"Gemini returned a non-JSON response; check for a discrepancy in your image: {response_text}"
            )

        return result

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
    def gemini_client(self) -> genai.Client:
        """Return the gemini client class."""
        return genai.Client(api_key="AIzaSyB0_pZAS3obaSyjnri2TtRqvsUDzH7pt5g")
