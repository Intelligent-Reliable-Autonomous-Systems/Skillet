"""Skills configuration file.

Written by Will Solow, 2026
"""

from dataclasses import MISSING

from skillet.envs.mujoco import DirectRLEnvCfg


class SkillsDirectRLEnvCfg(DirectRLEnvCfg):
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
