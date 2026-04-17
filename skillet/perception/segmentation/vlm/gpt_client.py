import base64
import io
from functools import cache

from openai import AsyncOpenAI, OpenAI
from PIL import Image

from skillet.perception.segmentation.vlm.vlm_base import VLMClient


class GPTClient(VLMClient):
    """Class for OpenAI GPT-4o Vision Client."""

    def __init__(
        self,
        prompt_name: str = "detect_goal",
        model_id: str = "gpt-4o",
        device: str = "cuda",
        api_key: str | None = None,
    ) -> None:
        super().__init__(prompt_name, model_id, device)

        self.prompt = self._load_prompt(prompt_name)
        self.client = self._openai_client(api_key)

    def detect_and_translate(
        self,
        image: Image.Image,
        task_instruction: str,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Detect objects and translate task in a single GPT-4o API call.

        Args:
            image: The image to analyze.
            task_instruction: The natural language task to translate.

        Returns:
            Tuple of (bboxes, grounded_goal_atoms, grounded_scene_atoms).

        """
        prompt = self.prompt.format(task_instruction=task_instruction)
        b64_image = self._encode_image(image)

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0.0,
        )
        print(self.parse_response(response.choices[0].message.content))
        return self.parse_response(response.choices[0].message.content)

    async def detect_and_translate_async(
        self,
        image: Image.Image,
        task_instruction: str,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Asynchronously detect objects and translate task in a single GPT-4o API call.

        Args:
            image: The image to analyze.
            task_instruction: The natural language task to translate.

        Returns:
            Tuple of (bboxes, grounded_goal_atoms, grounded_scene_atoms).

        """
        async_client = AsyncOpenAI(api_key=self.client.api_key)
        prompt = self.prompt.format(task_instruction=task_instruction)
        b64_image = self._encode_image(image)

        response = await async_client.chat.completions.create(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0.0,
        )
        return self.parse_response(response.choices[0].message.content)

    def parse_response(self, response_text: str) -> tuple[list, list, list]:
        """Parse GPT response text into bboxes, grounded goal atoms, and grounded scene atoms."""
        try:
            result = self._load_json(response_text)
        except Exception:
            raise ValueError(
                f"GPT returned a non-JSON response; check for a discrepancy in your image: {response_text}"
            )

        return self._parse_response(result)

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode a PIL image to a base64 JPEG string."""
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _openai_client(api_key: str | None = None) -> OpenAI:
        """Return the OpenAI client. Uses OPENAI_API_KEY env var if api_key is None."""
        return OpenAI(api_key=api_key)
