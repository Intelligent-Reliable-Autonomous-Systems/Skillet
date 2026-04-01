import os
from functools import cache

import aiohttp
import numpy as np
import requests
from jaxtyping import Float
from scipy.spatial.transform import Rotation


class M2T2Client:
    """Client for producing candidate grasp positions with M2T2 foundation model."""

    def __init__(
        self,
        server_url: str = "http://localhost:8123",
        grasp_threshold: float = 0.035,
        num_grasps: int = 200,
        num_points: int = 16384,
        num_runs: int = 5,
        apply_bounds: bool = True,
    ) -> None:
        """Start the M2T2 client.

        Args:
            server_url: URL that the server is on
            grasp_threshold: Threshold for grasping
            num_grasps: number of grasps to generate
            num_runs: number of runs to make
            apply_bounds: if to apply bounds or not.

        """
        self.server_url = server_url
        self.grasp_threshold = grasp_threshold
        self.num_grasps = num_grasps
        self.num_runs = num_runs
        self.num_points = num_points
        self.apply_bounds = apply_bounds

    def generate_grasps(
        self,
        scene_xyz: Float[np.ndarray, "n 3"],
        scene_rgb: Float[np.ndarray, "n 3"],
    ):
        """Generate grasps from point cloud using M2T2 server (synchronous version).

        Note: the coordinate frame of the grasps are in M2T2's convention.

        Args:
            scene_xyz: XYZ data from objects in scene
            scene_rgb: RGB information from objects in scene

        Returns:
            Dict object of poses, confidences and contacts

        """
        payload = self._build_payload(scene_xyz, scene_rgb)
        endpoint = os.path.join(self.server_url.rstrip("/"), "predict")

        print(f"[DEBUG] Sending inference request to M2T2 server at {endpoint}")
        response = requests.post(endpoint, json=payload, timeout=500)
        result = response.json()

        return self._process_m2t2_response(result)

    async def generate_grasps_async(
        self,
        session: aiohttp.ClientSession,
        scene_xyz: Float[np.ndarray, "n 3"],
        scene_rgb: Float[np.ndarray, "n 3"],
    ) -> dict:
        """Generate grasps from point cloud using M2T2 server (async version).

        Note: the coordinate frame of the grasps are in M2T2's convention.

        Args:
            scene_xyz: XYZ data from objects in scene
            scene_rgb: RGB information from objects in scene

        Returns:
            Dict object of poses, confidences and contacts

        """
        payload = self._build_payload(scene_xyz, scene_rgb)
        endpoint = os.path.join(self.server_url.rstrip("/"), "predict")

        print(f"[DEBUG] Sending inference request to M2T2 server at {endpoint}")
        async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=30.0)) as response:
            result = await response.json()

        return self._process_m2t2_response(result)

    def _process_m2t2_response(self, result: dict) -> dict:
        """Process M2T2 response and return structured grasp outputs.

        Returns:
            Dict object of poses, confidences and contacts

        """
        grasps_list = result.get("grasps", [])
        confidences_list = result.get("grasp_confidence", [])
        contacts_list = result.get("grasp_contacts", [])
        outputs = {}

        for i, (grasps, confidences, contacts) in enumerate(
            zip(grasps_list, confidences_list, contacts_list, strict=False)
        ):
            label = f"object_{i}"
            if len(grasps) == 0:
                outputs[label] = {
                    "poses": np.array([]).reshape(0, 4, 4),
                    "confidences": np.array([]),
                    "contacts": np.array([]),
                }
            else:
                poses = np.array(grasps)
                confs = np.array(confidences)
                conts = np.array(contacts)

                if self.num_grasps is not None and len(poses) > self.num_grasps:
                    top_indices = np.argsort(confs)[-self.num_grasps :]
                    poses = poses[top_indices]
                    confs = confs[top_indices]
                    conts = conts[top_indices]

                outputs[label] = {
                    "poses": poses,
                    "confidences": confs,
                    "contacts": conts,
                }

        return outputs

    def _build_payload(
        self,
        scene_xyz: Float[np.ndarray, "n 3"],
        scene_rgb: Float[np.ndarray, "n 3"],
    ) -> dict:
        """Build payload for M2T2 server."""
        return {
            "pointcloud": {
                "points": scene_xyz.tolist(),
                "rgb": scene_rgb.tolist(),
            },
            "num_points": self.num_points,
            "num_runs": self.num_runs,
            "mask_thresh": self.grasp_threshold,
            "apply_bounds": self.apply_bounds,
        }

    @cache
    @staticmethod
    def m2t2_to_tiptop_transform() -> np.ndarray:
        """4x4 transform to take M2T2 grasp poses to the convention expected by tiptop."""
        # Panda offset
        base_to_tcp = np.eye(4)
        base_to_tcp[2, 3] = 0.1034

        # To tiptop frame with z-up
        to_tiptop_frame = np.eye(4)
        to_tiptop_frame[:3, :3] = Rotation.from_euler("xyz", np.array([np.pi, 0, -np.pi / 2])).as_matrix()
        return base_to_tcp @ to_tiptop_frame
