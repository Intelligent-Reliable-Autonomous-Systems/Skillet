"""
Camera configuration for Isaac Sim RGBD sensors.
"""

from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

from skillet.envs.util import configclass


@configclass
class RGBDCameraCfg:
    """
    Configuration for an RGBD camera sensor in Isaac Sim.

    Specifies spawn position/orientation relative to a parent prim, as well as
    resolution and intrinsic parameters.

    Usage example::

        cam_cfg = RGBDCameraCfg(
            pos=(0.5, 0.0, 0.8),
            rot=(0.0, 0.0, 0.0, 0.0),
            width=640,
            height=480,
        )
        isaac_cam_cfg = cam_cfg.to_isaac_cfg("/World/envs/env_.*/Camera")
    """

    # --- USD prim path ---

    prim_path: str = "/World/envs/env_.*/Camera"
    """USD prim path where the camera will be created.

    Supports environment wildcards (``env_.*``).  To attach to a robot link use
    the full path to that link, e.g.::

        "/World/envs/env_.*/Robot/Arm/bracelet_link/Camera"
    """

    # --- Spawn pose ---

    pos: tuple = (0.0, 0.0, 0.0)
    """Position (x, y, z) offset relative to the parent prim (meters)."""

    rot: tuple = (0.0, 1.0, 0.0, 0.0)
    """Rotation as (w, x, y, z) quaternion relative to the parent prim."""

    # --- Image resolution ---

    width: int = 640
    """Image width in pixels."""

    height: int = 480
    """Image height in pixels."""

    # --- Camera intrinsics ---

    focal_length: float = 24.0
    """Focal length of the pinhole camera (mm)."""

    horizontal_aperture: float = 20.0
    """Horizontal aperture of the pinhole camera (mm)."""

    clipping_near: float = 0.1
    """Near clipping plane distance (meters)."""

    clipping_far: float = 1000.0
    """Far clipping plane distance (meters)."""

    # --- Sensor timing ---

    update_period: float = 0.0
    """Sensor update period in seconds. 0.0 means every simulation step."""

    # --- Frame convention ---

    convention: str = "world"
    """
    Convention that defines the camera's default viewing axis.

    Isaac Lab applies an implicit rotation to convert from the chosen convention
    to its internal USD frame. The composed rotation is: ``rot × convention_fix``.

    - ``"world"``  : forward axis: +X - up axis +Z - Offset is applied in the World Frame convention
    - ``"ros"``    : forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
    - ``"opengl"`` : forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention

    For link-mounted cameras where you want ``rot`` applied literally, use ``"world"``.
    """

    def to_isaac_cfg(self) -> CameraCfg:
        """
        Convert to an IsaacLab :class:`CameraCfg`.

        Returns:
            A fully configured :class:`CameraCfg` ready to pass to
            :class:`isaaclab.sensors.Camera`.
        """
        return CameraCfg(
            prim_path=self.prim_path,
            update_period=self.update_period,
            height=self.height,
            width=self.width,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=self.focal_length,
                horizontal_aperture=self.horizontal_aperture,
                clipping_range=(self.clipping_near, self.clipping_far),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=self.pos,
                rot=self.rot,
                convention=self.convention,
            ),
        )
