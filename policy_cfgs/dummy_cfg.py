"""dummy_cfg.py.

A dummy cfg configuration

Written by Will Solow & Jeff Jewett, 2026
"""

from dataclasses import MISSING

from cfg import configclass
from robot_skills import PolicyCfg
from robot_skills.low_level_policy import LowLevelPolicyCfg
from robot_skills.task_policy import TaskPolicyCfg


@configclass
class DummyCfg(PolicyCfg):
    """Instantiable dummy policy."""

    task_policy = TaskPolicyCfg(
        task_policy_name="RandomTaskPolicy",
        skills=["Random", "Zeroes"],
        skills_cfgs=[LowLevelPolicyCfg(output_dim=8), LowLevelPolicyCfg(output_dim=8)],
        num_envs=MISSING,
    )
