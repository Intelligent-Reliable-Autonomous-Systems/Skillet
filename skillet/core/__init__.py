"""The core module contains the basic components of environment, policy, skills, and spaces.

This module provides the basic components of the Robot Skills framework.
It includes:
- The environment interface
- The policy interface
- The skill interface
- The space specification interface
"""

from skillet.core.env import (
    BasicBatchedEnvironment,
    BasicEnvironment,
    BatchedEnvironment,
    Environment,
)
from skillet.core.policy import BatchedPolicy, BatchedPPolicy, BatchedUPolicy, Policy, PPolicy, UPolicy
from skillet.core.skill import BatchedSkill, CompositeSkill, SingleSkill, Skill
from skillet.core.spaces import (
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
    "BatchedPPolicy",
    "BatchedPolicy",
    "BatchedSkill",
    "BatchedUPolicy",
    "CommonSpecs",
    "CompositeSkill",
    "Environment",
    "ObservationSpec",
    "PPolicy",
    "Policy",
    "SingleSkill",
    "Skill",
    "SkillParamsSpec",
    "SpaceSpecification",
    "UPolicy",
]
