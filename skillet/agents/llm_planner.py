"""An LLM-based task planner"""

import base64
import io
from abc import ABC, abstractmethod

from PIL import Image

from skillet.scene.abstract.spatial_grounding import ground_cube_on_relations, ground_gripper_relations
from skillet.scene.abstract.up_utils import AbstractAction, AbstractPlan
from skillet.scene.base import Scene
from skillet.scene.cube import Cube


_PREDICATE_PROMPT = """You are analyzing a robot tabletop workspace from the attached image.

Available blocks: {block_names}
Surface: {table_name}

Output the current spatial state as a list of predicates, one per line, using EXACTLY these forms (no bullets, no extra punctuation):
  X is on Y
  X is clear (nothing on top)
  Robot hand is empty
  Robot is holding X

Use the block names verbatim. Emit a predicate for every block on a surface (block or table), mark each block that has nothing on top of it as clear, and state exactly one of "Robot hand is empty" or "Robot is holding X". Output only the predicate lines, nothing else."""


class LLMClient(ABC):
    """Abstract interface for an LLM text completion client."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send a text prompt and return the raw text response."""
        raise NotImplementedError

    @abstractmethod
    def complete_with_image(self, prompt: str, image: Image.Image) -> str:
        """Send a text prompt with an image and return the raw text response."""
        raise NotImplementedError

    def predicates_from_image(
        self, image: Image.Image, block_names: list[str], table_name: str
    ) -> list[str]:
        """Ask the VLM for scene predicates matching the geometric grounding format."""
        prompt = _PREDICATE_PROMPT.format(
            block_names=", ".join(block_names), table_name=table_name
        )
        response = self.complete_with_image(prompt, image)
        return [line.strip() for line in response.strip().splitlines() if line.strip()]


class OpenAIClient(LLMClient):
    """OpenAI client"""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, base_url: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    def complete_with_image(self, prompt: str, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0.0,
        )
        return response.choices[0].message.content


class GeminiLLMClient(LLMClient):
    """Google Gemini client"""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def complete(self, prompt: str) -> str:
        response = self._client.models.generate_content(model=self._model, contents=[prompt])
        return response.text

    def complete_with_image(self, prompt: str, image: Image.Image) -> str:
        response = self._client.models.generate_content(
            model=self._model, contents=[image, prompt]
        )
        return response.text


class LLMPlanner:
    """Uses an LLM to generate a plan from the current scene state.

    Args:
        client: An LLMClient instance. Defaults to OpenAIClient("gpt-4o")

    """

    def __init__(self, client: LLMClient | None = None, use_vlm_predicates: bool = False) -> None:
        """Initialize the planner.

        Args:
            client: An LLMClient instance. Defaults to OpenAIClient("gpt-4o").
            use_vlm_predicates: If True, derive scene predicates from a workspace
                image via the VLM; otherwise use geometric grounding on the
                scene's 3D poses.

        """
        self._client = client or OpenAIClient()
        self._use_vlm_predicates = use_vlm_predicates

    def _geometric_state_lines(self, scene: Scene) -> list[str]:
        """Build predicate lines from the scene's symbolic grounding."""
        on_preds, clear_preds = ground_cube_on_relations(scene)
        empty_pred, holding_pred = ground_gripper_relations(scene)

        lines = []
        for pred in on_preds:
            lines.append(f"  {pred[1].name} is on {pred[2].name}")
        for pred in clear_preds:
            lines.append(f"  {pred[1].name} is clear (nothing on top)")
        if empty_pred:
            lines.append("  Robot hand is empty")
        for pred in holding_pred:
            lines.append(f"  Robot is holding {pred[1].name}")
        return lines

    def scene_to_prompt(self, scene: Scene, goal: str, image: Image.Image | None = None) -> str:
        """Convert scene state into a natural language prompt for the LLM."""
        block_names = scene.get_object_names(Cube)
        table_name = scene.table.name

        if self._use_vlm_predicates:
            if image is None:
                raise ValueError(
                    "LLMPlanner configured with use_vlm_predicates=True but no image was provided."
                )
            raw_lines = self._client.predicates_from_image(image, block_names, table_name)
            state_lines = [f"  {line}" for line in raw_lines]
            print("[LLMPlanner] Predicates source: VLM (image-based)")
        else:
            state_lines = self._geometric_state_lines(scene)
            print("[LLMPlanner] Predicates source: geometric grounding")

        print("[LLMPlanner] Predicates:")
        for line in state_lines:
            print(line.rstrip())

        state_str = "\n".join(state_lines)

        return f"""You are a robot task planner. Plan a sequence of pick and place actions.

Available objects: {', '.join(block_names)} (blocks), {table_name} (table surface)

Current state:
{state_str}

Goal: {goal}

Available actions:
- pick_block(block, surface): Pick up a block from a surface. Requires: block is clear, block is on surface, hand is empty.
- place_block(block, surface): Place a block onto a surface. Requires: holding the block, surface is clear.

IMPORTANT:
- First check whether the goal is ALREADY satisfied by the current state. If every goal condition already holds, output `[]` (an empty list) — do NOT add pointless actions.
- Skip any step whose effect already holds in the current state.
- Only output actions that change the state toward the goal.

Output ONLY a JSON list of actions. Example (when the goal is not yet met):
[
  {{"action": "pick_block", "params": ["red_block", "table0"]}},
  {{"action": "place_block", "params": ["red_block", "blue_block"]}}
]"""

    def plan(
        self, scene: Scene, goal: str, image: Image.Image | None = None
    ) -> tuple[bool, AbstractPlan | None]:
        """Query the LLM to generate a plan.

        Args:
            scene: Current scene with object poses.
            goal: Natural language goal description.
            image: Workspace image. Required when ``use_vlm_predicates=True``,
                ignored otherwise.

        Returns:
            (success, plan) tuple matching the AbstractModel.plan() interface.

        """
        import json

        prompt = self.scene_to_prompt(scene, goal, image=image)
        content = self._client.complete(prompt)

        # Parse JSON from response (strip markdown code fences if present)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            actions_json = json.loads(content)
        except json.JSONDecodeError:
            print(f"[LLMPlanner] Failed to parse LLM response:\n{content}")
            return (False, None)

        actions = [
            AbstractAction(action=a["action"], parameters=a["params"])
            for a in actions_json
        ]
        return (True, AbstractPlan(actions=actions))