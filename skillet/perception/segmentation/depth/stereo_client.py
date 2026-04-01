import io
import os.path
import time

import aiohttp
import cv2
import numpy as np
import requests
from jaxtyping import Float, UInt8

from skillet.perception.realsense import RealsenseFrame, RealsenseIntrinsics


class StereoClient:
    """Foundation Stereo model client for inferring depth from two stereo images."""

    def __init__(self, server_url: str = "http://localhost:1234") -> None:
        self.server_url = server_url

    def rs_infer_depth(
        self,
        frame: RealsenseFrame,
        intrinsics: RealsenseIntrinsics,
    ) -> Float[np.ndarray, "h w"]:
        """Estimate depth from Realsense frame and intrinsics using FoundationStereo. Synchronous version.

        Args:
            frame: RealsenseFrame object containing relevant camera information
            intrinsics: Camera intrinsics

        Returns:
            Depth map from image.

        """
        ir_left_rgb, ir_right_rgb, rgb_size = self._prepare_ir_stereo(frame)
        k_ir = intrinsics.K_ir
        depth = self._infer_depth(
            ir_left_rgb,
            ir_right_rgb,
            fx=k_ir[0, 0],
            fy=k_ir[1, 1],
            cx=k_ir[0, 2],
            cy=k_ir[1, 2],
            baseline=intrinsics.baseline_ir,
        )
        return self._depth_ir_to_color(depth, k_ir, intrinsics.T_color_from_ir, intrinsics.K_color, color_size=rgb_size)

    async def rs_infer_depth_async(
        self,
        session: aiohttp.ClientSession,
        frame: RealsenseFrame,
        intrinsics: RealsenseIntrinsics,
    ) -> Float[np.ndarray, "h w"]:
        """Estimate depth from Realsense frame and intrinsics using FoundationStereo. Async version.

        Args:
            session: Client session
            frame: RealsenseFrame object containing relevant camera information
            intrinsics: Camera intrinsics

        Returns:
            Depth map from image.

        """
        ir_left_rgb, ir_right_rgb, rgb_size = self._prepare_ir_stereo(frame)
        k_ir = intrinsics.k_ir
        depth = await self._infer_depth_async(
            session,
            self.server_url,
            ir_left_rgb,
            ir_right_rgb,
            fx=k_ir[0, 0],
            fy=k_ir[1, 1],
            cx=k_ir[0, 2],
            cy=k_ir[1, 2],
            baseline=intrinsics.baseline_ir,
        )
        return self._depth_ir_to_color(depth, k_ir, intrinsics.T_color_from_ir, intrinsics.K_color, color_size=rgb_size)

    def _prepare_ir_stereo(
        self,
        frame: RealsenseFrame,
    ) -> tuple[UInt8[np.ndarray, "h w 3"], UInt8[np.ndarray, "h w 3"], tuple[int, int]]:
        """Prepare IR stereo images for FoundationStereo inference.

        Args:
            frame: RealsenseFrame object containing relevant image information.

        Returns:
            Left and Right RGB image and image size.

        """
        rgb_size = frame.rgb.shape[:2]
        ir_size = frame.ir_left.shape[:2]
        if rgb_size != ir_size:
            raise NotImplementedError("We don't currently support different color and IR resolutions")

        # Convert IR to RGB (FoundationStereo expects 3-channel input)
        ir_left, ir_right = frame.ir_left, frame.ir_right
        ir_left_rgb = np.stack([ir_left, ir_left, ir_left], axis=-1)
        ir_right_rgb = np.stack([ir_right, ir_right, ir_right], axis=-1)

        return ir_left_rgb, ir_right_rgb, rgb_size

    def _infer_depth(
        self,
        left_rgb: UInt8[np.ndarray, "h w 3"],
        right_rgb: UInt8[np.ndarray, "h w 3"],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        baseline: float,
    ) -> Float[np.ndarray, "h w"]:
        """Predict depth given a stereo pair using FoundationStereo (synchronous version).

        Args:
            left_rgb: RGB image from left camera
            right_rgb: RGB image from right camera
            fx: fx intrinsics
            fy: fy intrinsics
            cx: cx intrinsics
            cy: cy intrinsics
            baseline: Stereo baseline

        Returns:
            Depthmap of image computed from FoundationStereo model

        """
        start_time = time.perf_counter()
        left_bytes, right_bytes = self._encode_images_to_png(left_rgb, right_rgb)
        files = {
            "left_image": ("left.png", left_bytes, "image/png"),
            "right_image": ("right.png", right_bytes, "image/png"),
        }
        data = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "baseline": baseline,
            "scale": 1.0,
            "hiera": 0,
            "valid_iters": 32,
        }

        infer_endpoint = os.path.join(self.server_url.rstrip("/"), "infer")
        print(f"[DEBUG] Sending inference request to FoundationStereo server at {infer_endpoint}")
        response = requests.post(infer_endpoint, files=files, data=data)
        if response.status_code != 200:
            raise RuntimeError(
                f"FoundationStereo request failed with status code {response.status_code}. Response: {response.text}"
            )

        depth = self._decode_depth_response(response.content)
        duration = time.perf_counter() - start_time
        print(f"[INFO] FoundationStereo depth map={depth.shape}, inference time={duration:.2f}s")
        return depth

    async def _infer_depth_async(
        self,
        session: aiohttp.ClientSession,
        left_rgb: UInt8[np.ndarray, "h w 3"],
        right_rgb: UInt8[np.ndarray, "h w 3"],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        baseline: float,
    ) -> Float[np.ndarray, "h w"]:
        """Predict depth given a stereo pair using FoundationStereo (synchronous version).

        Args:
            session: Client session
            left_rgb: RGB image from left camera
            right_rgb: RGB image from right camera
            fx: fx intrinsics
            fy: fy intrinsics
            cx: cx intrinsics
            cy: cy intrinsics
            baseline: Stereo baseline

        Returns:
            Depthmap of image computed from FoundationStereo model

        """
        start_time = time.perf_counter()
        left_bytes, right_bytes = self._encode_images_to_png(left_rgb, right_rgb)

        # Create FormData for multipart upload
        data = aiohttp.FormData()
        data.add_field("left_image", left_bytes, filename="left.png", content_type="image/png")
        data.add_field("right_image", right_bytes, filename="right.png", content_type="image/png")
        data.add_field("fx", str(fx))
        data.add_field("fy", str(fy))
        data.add_field("cx", str(cx))
        data.add_field("cy", str(cy))
        data.add_field("baseline", str(baseline))
        data.add_field("scale", "1.0")
        data.add_field("hiera", "0")
        data.add_field("valid_iters", "32")

        # Call the server
        infer_endpoint = os.path.join(self.server_url.rstrip("/"), "infer")
        print(f"[DEBUG] Sending inference request to FoundationStereo server at {infer_endpoint}")

        async with session.post(infer_endpoint, data=data, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(
                    f"FoundationStereo request failed with status code {response.status}. Response: {text}"
                )
            content = await response.read()

        # Decode response
        depth = self._decode_depth_response(content)

        duration = time.perf_counter() - start_time
        print(f"[INFO] FoundationStereo depth map={depth.shape}, inference time={duration:.2f}s")
        return depth

    def _depth_ir_to_color(
        self,
        depth_ir: Float[np.ndarray, "h w"],
        k_ir: Float[np.ndarray, "3 3"],
        t_color_from_ir: Float[np.ndarray, "4 4"],
        k_color: Float[np.ndarray, "3 3"],
        color_size: tuple[int, int],
    ) -> np.ndarray:
        """Warp IR depth (meters) onto color pixel grid using forward projection.

        Uses 4-neighbor splatting with z-buffer min, then fills small holes via min-filter.

        Args:
            depth_ir: Depth IR image
            k_ir: camera IR intrinsics
            t_color_from_ir: T_color intrinsics
            k_color: camera RGB intrinsics
            color_size: size of RGB image

        Returns:
            A depth colormap

        """
        Hc, Wc = color_size
        Hi, Wi = depth_ir.shape
        assert Hc > 0 and Wc > 0 and Hi > 0 and Wi > 0, "invalid image sizes for depth warp"

        fx_i, fy_i = float(k_ir[0, 0]), float(k_ir[1, 1])
        cx_i, cy_i = float(k_ir[0, 2]), float(k_ir[1, 2])
        fx_c, fy_c = float(k_color[0, 0]), float(k_color[1, 1])
        cx_c, cy_c = float(k_color[0, 2]), float(k_color[1, 2])

        u, v = np.meshgrid(np.arange(Wi, dtype=np.float32), np.arange(Hi, dtype=np.float32))
        z = depth_ir.astype(np.float32)
        valid = (z > 0.0) & np.isfinite(z)
        if not np.any(valid):
            return np.zeros((Hc, Wc), dtype=np.float32)

        # Unproject IR pixels to 3D
        x_i = (u[valid] - cx_i) / max(fx_i, 1e-6) * z[valid]
        y_i = (v[valid] - cy_i) / max(fy_i, 1e-6) * z[valid]
        pts_ir = np.stack([x_i, y_i, z[valid]], axis=0)

        # Transform to color frame
        R = t_color_from_ir[:3, :3].astype(np.float32)
        t = t_color_from_ir[:3, 3].astype(np.float32).reshape(3, 1)
        pts_c = R @ pts_ir + t
        Xc, Yc, Zc = pts_c[0], pts_c[1], pts_c[2]
        valid_c = Zc > 1e-6
        if not np.any(valid_c):
            return np.zeros((Hc, Wc), dtype=np.float32)
        Xc, Yc, Zc = Xc[valid_c], Yc[valid_c], Zc[valid_c]

        # Project to color image
        uc_f = fx_c * (Xc / Zc) + cx_c
        vc_f = fy_c * (Yc / Zc) + cy_c
        x0 = np.floor(uc_f).astype(np.int32)
        y0 = np.floor(vc_f).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        depth_color = np.full((Hc, Wc), np.inf, dtype=np.float32)

        def splat(ix: np.ndarray, iy: np.ndarray, zvals: np.ndarray) -> None:
            inb = (ix >= 0) & (ix < Wc) & (iy >= 0) & (iy < Hc)
            if not np.any(inb):
                return
            np.minimum.at(depth_color, (iy[inb], ix[inb]), zvals[inb])

        # Splat to 4 neighbors to reduce gaps
        splat(x0, y0, Zc)
        splat(x1, y0, Zc)
        splat(x0, y1, Zc)
        splat(x1, y1, Zc)

        # Fill holes with iterative erosion (min-filter)
        # This handles larger holes from FoundationStereo by propagating valid depth values
        holes = np.isinf(depth_color)
        if np.any(holes):
            depth_color[holes] = 0.0
            kernel = np.ones((3, 3), np.uint8)
            max_iterations = 5  # Fill holes up to ~5 pixels wide
            for _ in range(max_iterations):
                holes_mask = depth_color <= 0.0
                if not np.any(holes_mask):
                    break
                # Use large sentinel value for unfilled regions, erode to get min of neighbors
                sentinel = np.where(depth_color > 0.0, depth_color, 65535.0).astype(np.float32)
                min_neigh = cv2.erode(sentinel, kernel)

                # Only fill pixels that have at least one valid neighbor (not all sentinels)
                newly_filled = holes_mask & (min_neigh < 65000.0)
                depth_color[newly_filled] = min_neigh[newly_filled]

            # Clean up any remaining unfilled holes
            depth_color[depth_color > 65000.0] = 0.0

        return depth_color

    def _decode_depth_response(self, content: bytes) -> Float[np.ndarray, "h w"]:
        """Decode depth map from NPZ response content.

        Args:
            content: Depth map as NPZ in byes

        Returns:
            np.ndarray of depth map

        """
        buffer = io.BytesIO(content)
        return np.load(buffer)["depth"]

    def _encode_images_to_png(
        self, left_rgb: UInt8[np.ndarray, "h w 3"], right_rgb: UInt8[np.ndarray, "h w 3"]
    ) -> tuple[bytes, bytes]:
        """Encode left and right rgb image into .png format.

        Args:
            left_rgb: Left RGB image from camera
            right_rgb: Right RGB image from camera

        Returns:
            left and right image encodings as bytes

        """
        if left_rgb.shape != right_rgb.shape:
            raise ValueError(f"Expected shape of left_rgb {left_rgb.shape} to match right_rgb {right_rgb.shape}")
        if not (left_rgb.dtype == right_rgb.dtype == np.uint8):
            raise ValueError(f"Expected uint8 dtype for left_rgb ({left_rgb.dtype}) and right_rgb ({right_rgb.dtype})")

        # Since we're encoding with cv2 need to convert to BGR first
        _, left_bytes = cv2.imencode(".png", cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
        _, right_bytes = cv2.imencode(".png", cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))

        return left_bytes.tobytes(), right_bytes.tobytes()
