import io
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Generic

import httpx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from jaxtyping import Float, UInt8
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from skillet.core.policy import BatchedUPolicy, TBAction, TBPolicyObs
from skillet.core.spaces import ActionSpec, ObservationSpec, SkillParamsSpec, TSkillParams
from skillet.envs.specs import RGBD_Obs, TWIST_TCP_Action

VLA_SERVER_URL = "http://localhost:8001"


@dataclass
class ActionResponse:
    """Dataclass for segmenting from concepts."""

    action: list[float]


class OpenVlaPolicy(BatchedUPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):
    def __init__(self, use_server: bool = True, load_server: bool = False, server_url: str = VLA_SERVER_URL):
        self.server_url = server_url
        self.use_server = use_server
        self.model_name = "OpenVLA"

        if use_server and load_server:
            OpenVlaPolicy.ensure_server(self.server_url, self.model_name)

        if not load_server or not use_server:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # "nf4" or "fp4"
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # nested quantization, saves a bit more
            )
            self._image_processer = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
            self._vla = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b",
                quantization_config=bnb_config,
                # torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="cuda:0",
            )

    @property
    def obs_spec(self) -> ObservationSpec[RGBD_Obs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TWIST_TCP_Action]:  # noqa: D102
        return self._action_spec

    @property
    def params_spec(self) -> SkillParamsSpec[TSkillParams]:
        """The parameter specification for vla parameters."""
        return self._params_spec

    def get_action(self, obs: RGBD_Obs, params: Any = None) -> TBAction:
        """Get the action from the policy."""
        if self.use_server:
            return self.remote_action(obs["rgb"])
        return self.action(obs["rgb"])

    def action(self, rgb: torch.Tensor):
        """Get the action from the VLA."""
        instruction = "Place pink block on purple block"
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        image = Image.fromarray(rgb.permute((1, 2, 0)).cpu().numpy())
        inputs = self._image_processer(prompt, image, return_tensors="pt").to(rgb.device, dtype=torch.float16)
        return self._vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    def remote_action(
        self,
        rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image,
    ) -> tuple[Float[torch.Tensor, "n b"],]:
        """Process actions from image on remote server.

        Args:
            rgb: RGB image. Can be an rgb from an RGBD obs or a PIL image.

        Returns:
            actions: actions for robot take in form (b,7) (XYZ,RPY,Gripper)

        """
        image_bytes = self.arr_to_bytes(rgb)
        client = self._get_http_client()

        data = self._decode_response(
            client.post(
                "/vla/actions",
                files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            )
        )

        return torch.tensor(data["actions"], dtype=torch.float32, device=self.device)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Load model into GPU on startup; nothing to clean up on shutdown."""
        yield

    def create_server(self) -> FastAPI:
        """Create and return the FastAPI app bound to this client instance."""
        app = FastAPI(
            title="OpenVLA Server",
            lifespan=lambda a: self.lifespan(a),  # bind self
        )

        def _tensor_to_list(t: torch.Tensor) -> list:
            return t.cpu().tolist()

        def _upload_to_tensor(file_bytes: bytes) -> torch.Tensor:
            """Decode uploaded image bytes → CHW uint8 tensor on CUDA."""
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            arr = np.array(img, dtype=np.uint8)  # HWC
            return torch.as_tensor(arr).permute(2, 0, 1)  # CHW

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.post("/vla/actions", response_model=ActionResponse)
        async def actions(
            file: Annotated[UploadFile, File(description="RGB image (JPEG/PNG)")],
        ):
            """Segment all instances matching any of the provided text concepts.

            Returns one mask per detected instance, with the concept index it matched.
            """
            rgb = _upload_to_tensor(await file.read())

            action = self.action(rgb)

            return ActionResponse(
                actions=_tensor_to_list(action),
            )

        return app

    @staticmethod
    def _is_server_alive(server_url: str = VLA_SERVER_URL) -> bool:
        """Check if the server is already running."""
        try:
            r = httpx.get(f"{server_url}/health", timeout=2.0)
            return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    @staticmethod
    def _wait_for_server(server_url: str, timeout: float = 90.0) -> bool:
        """Poll until server is up or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if OpenVlaPolicy._is_server_alive(server_url):
                return True
            time.sleep(0.5)
        return False

    @staticmethod
    def ensure_server(server_url: str, model_name: str | None = None, timeout: float = 90.0) -> None:
        """Start the server as a detached background process if not already running."""
        if OpenVlaPolicy._is_server_alive(server_url):
            return

        print("[INFO][VLA] No server found, spawning background process…")
        subprocess.Popen(
            [sys.executable, __file__, "--serve", model_name],
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        print("[INFO][VLA] Waiting for server to come up…")
        if not OpenVlaPolicy._wait_for_server(server_url, timeout=timeout):
            raise RuntimeError(f"VLA server did not start within {timeout}s.")
        print("[INFO][VLA] Server is up.")

    def _get_http_client(self) -> httpx.Client:
        """Lazily create a persistent HTTP client (reuses TCP connection)."""
        if not hasattr(self, "_http_client"):
            self._http_client = httpx.Client(base_url=self.server_url, timeout=60.0)
        return self._http_client

    def arr_to_bytes(self, rgb: UInt8[torch.Tensor | np.ndarray, "3 h w"] | Image.Image) -> bytes:
        """Convert any supported image type to JPEG bytes for upload."""
        if isinstance(rgb, Image.Image):
            img = rgb
        else:
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            img = Image.fromarray(rgb.transpose(1, 2, 0).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def _decode_response(self, response: httpx.Response) -> dict:
        """Parse server response to JSON."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"VLA server returned {response.status_code}: {response.text}") from e
        return response.json()


def main() -> None:
    """Run the VLA server."""
    if "--serve" in sys.argv:
        vla_client = OpenVlaPolicy()
        app = vla_client.create_server()
        uvicorn.run(app, host="0.0.0.0", port=8001, reload=False, workers=1, access_log=False)
    else:
        vla_client.ensure_server(VLA_SERVER_URL)
        print(f"[INFO][VLA] Ready. Server at {VLA_SERVER_URL}")
