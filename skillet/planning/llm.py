"""LLM clients for translating natural language goals into PDDL."""

import os
import re
from abc import ABC, abstractmethod
from functools import cache
from pathlib import Path
from typing import Literal

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

LLMProvider = Literal["gemini", "claude", "anthropic"]


class GoalTranslator(ABC):
    """Translate a natural language task into a PDDL goal expression."""

    def __init__(self, prompt_name: str = "goal_translation", model_id: str | None = None) -> None:
        self.prompt_name = prompt_name
        self.model_id = model_id

    def translate(self, task_instruction: str, domain_pddl: str, problem_pddl: str, domain: str | None = None) -> str:
        """Translate a natural language task into a PDDL goal expression.

        Args:
            task_instruction: The natural language task.
            domain_pddl: The PDDL domain string.
            problem_pddl: The PDDL problem string containing the objects and initial state.
            domain: The domain name used to select domain specific prompt notes (e.g. "blocks", "sponge").

        Returns:
            The goal expression to go inside (:goal ...), e.g. ``(and (on red_block blue_block))``.

        """
        prompt = self.build_prompt(task_instruction, domain_pddl, problem_pddl, domain)
        return self.parse_goal(self.query(prompt))

    def build_prompt(
        self, task_instruction: str, domain_pddl: str, problem_pddl: str, domain: str | None = None
    ) -> str:
        """Fill the prompt template with the domain, problem, task, and domain specific notes."""
        return _load_prompt(self.prompt_name).format(
            domain=domain_pddl,
            problem=problem_pddl,
            task_instruction=task_instruction,
            domain_notes=self._load_domain_notes(domain),
        )

    def parse_goal(self, response_text: str) -> str:
        """Extract the goal expression from the response, dropping code fences and a (:goal ...) wrapper."""
        goal = re.sub(r"^```\w*|```$", "", response_text.strip()).strip()
        wrapped = re.fullmatch(r"\(\s*:goal\s*(.*)\)", goal, flags=re.DOTALL)
        if wrapped is not None:
            goal = wrapped.group(1).strip()
        if not goal.startswith("("):
            raise ValueError(f"LLM returned an invalid PDDL goal: {response_text}")
        return goal

    def _load_domain_notes(self, domain: str | None) -> str:
        if domain is None:
            return ""
        notes_path = _PROMPTS_DIR / f"{self.prompt_name}_{domain}.txt"
        return notes_path.read_text().strip() if notes_path.exists() else ""

    @abstractmethod
    def query(self, message: str) -> str:
        """Query the LLM with a text message and return the response text."""


class GeminiGoalTranslator(GoalTranslator):
    """Gemini client for natural language to PDDL goal translation."""

    def __init__(
        self,
        prompt_name: str = "goal_translation",
        model_id: str = "gemini-3.8-flash",
        api_key: str | None = None,
    ) -> None:
        """Initialize the Gemini translator.

        Args:
            prompt_name: Name of the prompt template in ./prompts/. Domain specific notes are loaded from
                ``{prompt_name}_{domain}.txt`` if present.
            model_id: The Gemini model to query.
            api_key: The Gemini API key. If None, falls back to GEMINI_API_KEY / GOOGLE_API_KEY.

        """
        from google import genai

        _load_project_env()
        super().__init__(prompt_name, model_id)
        self.client = genai.Client(api_key=api_key)

    def query(self, message: str) -> str:
        """Query Gemini with a text message."""
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[message],
            config=types.GenerateContentConfig(
                temperature=0.0, thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        return response.text


class ClaudeGoalTranslator(GoalTranslator):
    """Anthropic Claude client for natural language to PDDL goal translation."""

    def __init__(
        self,
        prompt_name: str = "goal_translation",
        model_id: str = "claude-opus-5-5",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        """Initialize the Claude translator.

        Args:
            prompt_name: Name of the prompt template in ./prompts/. Domain specific notes are loaded from
                ``{prompt_name}_{domain}.txt`` if present.
            model_id: The Claude model to query.
            api_key: The Anthropic API key. If None, falls back to ANTHROPIC_API_KEY.
            max_tokens: Maximum tokens to generate.

        """
        import anthropic  # type: ignore[import-untyped,import-not-found]

        _load_project_env()
        super().__init__(prompt_name, model_id)
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key)

    def query(self, message: str) -> str:
        """Query Claude with a text message."""
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": message}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


def create_goal_translator(
    provider: LLMProvider = "gemini",
    prompt_name: str = "goal_translation",
    model_id: str | None = None,
    api_key: str | None = None,
) -> GoalTranslator:
    """Construct a goal translator for the given LLM provider."""
    kwargs = {"prompt_name": prompt_name, "api_key": api_key}
    if model_id is not None:
        kwargs["model_id"] = model_id
    if provider == "gemini":
        return GeminiGoalTranslator(**kwargs)
    if provider in ("claude", "anthropic"):
        return ClaudeGoalTranslator(**kwargs)
    raise ValueError(f"Unknown LLM provider `{provider}`. Expected 'gemini', 'claude', or 'anthropic'.")


def _load_project_env() -> None:
    """Load KEY=VALUE pairs from the repo .env without overwriting existing environment variables."""
    if not _ENV_PATH.is_file():
        return
    for line in _ENV_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@cache
def _load_prompt(prompt_name: str) -> str:
    return (_PROMPTS_DIR / f"{prompt_name}.txt").read_text().strip()
