"""dummy_cfg.py.

A dummy cfg configuration

Written by Will Solow & Jeff Jewett, 2026
"""

from dataclasses import MISSING

from robot_skills import PolicyCfg
from robot_skills.task_policy import TaskPolicyCfg
from robot_skills.low_level_policy import LowLevelPolicyCfg
from cfg import configclass


@configclass
class DummyCfg(PolicyCfg):
    """Instantiable dummy policy."""

    task_policy = TaskPolicyCfg(
        task_policy_name="DummyTaskPolicy",
        skills=["Dummy"],
        skills_cfgs=[LowLevelPolicyCfg(output_dim=7)],
        num_envs=MISSING,
    )
