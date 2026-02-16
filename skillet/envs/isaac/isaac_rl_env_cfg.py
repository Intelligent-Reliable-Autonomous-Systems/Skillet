"""Skills configuration file.

Written by Will Solow, 2026
"""

from isaaclab.envs import DirectRLEnvCfg


class SkillsDirectRLEnvCfg(DirectRLEnvCfg):
    skills: list[str] | None = None
    """List of behavior primitives available"""
