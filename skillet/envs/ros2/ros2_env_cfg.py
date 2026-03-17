"""ros2_rl_env_cfg.py.

Main configuration for ROS2 RL Env CFG file

Written by Will Solow, 2026

"""

from dataclasses import MISSING

from skillet.envs.util import configclass


@configclass
class ROS2EnvCfg:
    """The configuration class for ROS2 RL Envs."""

    """Robot configuration"""

    ros2_workspace: str = MISSING
    """Workspace which bringup configuration is defined"""

    launch_ros: bool = True
    """If to launch robot software in gen3_py ROS2 package"""

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

    gripper_cmd_topic: str = MISSING
    """Name of gripper command topic"""

    move_group_name: str = MISSING
    """Name of move group for arm"""

    tool_frame_name: str = MISSING
    """Name of tool frame of robot"""

    base_apriltag_id: int = MISSING
    """April tag id used to localize base"""
