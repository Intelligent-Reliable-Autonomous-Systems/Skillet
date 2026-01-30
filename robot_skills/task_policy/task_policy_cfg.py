"""task_policy_cfg.py.

The task policy (high level) controller class configuration for skills

Written by Will Solow & Jeff Jewett, 2026
"""

from dataclasses import MISSING

from cfg import configclass
from robot_skills.low_level_policy import LowLevelPolicyCfg


@configclass
class TaskPolicyCfg:
    """Configuration of Task Policy."""

    task_policy_name: str = MISSING

    # List of the available skills in the environment
    skills: list[str] = MISSING

    # List of low level policy controller configurations for each skill
    skills_cfgs: list[LowLevelPolicyCfg] = MISSING

    # Number of environments
    num_envs: int = MISSING

    # Device
    device: str = "cuda"

    # Max action dim
    action_dim: int = MISSING
