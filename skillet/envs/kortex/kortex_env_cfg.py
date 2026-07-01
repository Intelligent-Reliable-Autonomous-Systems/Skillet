"""kortex_env_cfg.py.

Main configuration for Kortex Env CFG file

Written by Will Solow, 2026

"""

from dataclasses import MISSING

from skillet.envs.util import configclass


@configclass
class KortexEnvCfg:
    """The configuration class for Kortex Envs."""

    """Robot configuration"""
    urdf_path: str = MISSING
    """Path to the model of the robot"""
    srdf_path: str = MISSING
    """Path to the collision model of the robot"""
    assets_dir: list[str] = MISSING
    """Path to parent dir of robot meshes"""

    robot_ip: str = "www.xxx.yyy.zzz"
    """IP of the robot"""

    vision: str = "false"
    """If vision is enabled"""

    default_joint_positions: list[float] = MISSING
    """Default joint position of robot"""

    """RL environment configuration"""
    num_envs: int = 1
    """Number of parallel environments, will always be 1 for ROS"""

    device: str = "cuda"
    """GPU device"""

    dt: float = MISSING
    """Delta time per step"""

    decimation: float = MISSING
    """Decimation (steps through physics)"""

    episode_length_s: float = 1e9
    """Episode length in seconds"""

    skills: list[str] | None = None
    """List of behavior primitives available"""

    seed: int = MISSING
    """Seed for the environment"""

    is_finite_horizon: bool = False
    """Whether learning is treated as a finite or infinite horizon problem"""

    skills: list[str] | None = None
    """List of behavior primitives available"""

    use_sc: bool = False
    """If to use the skill controller through SkillEnvWrapper"""

    joint_ids: list[int] = MISSING
    """Joint Ids in the observation space"""

    tcp_offset: list[float] = MISSING
    """TCP offset from the end effector"""

    ee_link_name: str = MISSING
    """Name of the end effector link for Diff IK"""

    base_link_name: str = MISSING
    """Name of the base link for Diff IK"""

    gripper_joint_names: list[str] = MISSING
    """Name of the gripper joint for gripper pose"""

    arm_joint_names: list[str] = MISSING
    """Name of the arm joints for Diff IK"""

    base_apriltag_id: int = MISSING
    """April tag id used to localize base"""
    base_apriltag_size: int = MISSING
    """April tag id used to localize base"""
    base_apriltag_pose: list = MISSING
    """April tag id used to localize base"""
    base_apriltag_fam: str = MISSING
    """April tag id used to localize base"""

    use_tabletop_camera: bool = True
    """If to use the realsense tabletop camera"""
    gripper_close_time: float = 1.0
    """Time taken to close gripper, 0 = non blocking"""
