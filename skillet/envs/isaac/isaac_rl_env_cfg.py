"""Skills configuration file.

Written by Will Solow, 2026
"""

from dataclasses import MISSING

from isaaclab.envs import DirectRLEnvCfg


class SkillsDirectRLEnvCfg(DirectRLEnvCfg):
    skills: list[str] | None = None
    """List of behavior primitives available"""

    joint_ids: list[int] = MISSING
    """Joint Ids in the observation space"""

    tcp_offset: list[float] = MISSING
    """TCP offset from the end effector"""

    ee_link_name: str = MISSING
    """Name of the end effector link for Diff IK"""

    base_link_name: str = MISSING
    """Name of the base link for Diff IK"""

    gripper_joint_names: str = MISSING
    """Name of the gripper joint for gripper pose"""
