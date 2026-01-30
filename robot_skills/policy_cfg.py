"""policy_cfg.py.

The high level configuration for policies

Written by Will Solow & Jeff Jewett, 2026
"""

from dataclasses import MISSING

from cfg.utils import configclass
from robot_skills.task_policy import TaskPolicyCfg


@configclass
class PolicyCfg:
    """Configuration of Task Policy."""

    # List of the available skills in the environment
    task_policy: TaskPolicyCfg = MISSING
