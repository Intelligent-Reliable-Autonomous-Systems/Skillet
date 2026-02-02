"""The core module contains the basic components of environment, policy, skills, and spaces.

This module provides the basic components of the Robot Skills framework.
It includes:
- The environment interface
- The policy interface
- The skill interface
- The space specification interface
"""

from robot_skills.core.env import (
    BasicBatchedEnvironment,
    BasicEnvironment,
    BatchedEnvironment,
    Environment,
)
from robot_skills.core.policy import BatchedPolicy, BatchedUPolicy, Policy, UPolicy
from robot_skills.core.skill import BatchedSkill, CompositeSkill, SingleSkill, Skill
from robot_skills.core.spaces import (
    ActionSpec,
    CommonSpecs,
    ObservationSpec,
    SkillParamsSpec,
    SpaceSpecification,
)

__all__ = [
    "ActionSpec",
    "BasicBatchedEnvironment",
    "BasicEnvironment",
    "BatchedEnvironment",
    "BatchedPolicy",
    "BatchedSkill",
    "BatchedUPolicy",
    "CommonSpecs",
    "CompositeSkill",
    "Environment",
    "ObservationSpec",
    "Policy",
    "SingleSkill",
    "Skill",
    "SkillParamsSpec",
    "SpaceSpecification",
    "UPolicy",
]
