"""Base Skillet Agent class."""

from abc import ABC
from typing import TypeAlias, TypeVar

import torch
from jaxtyping import Int

from skillet.core.skill import Skill
from skillet.core.spaces import Action, BatchedAction, BatchedObservation, BatchedSkillParams, SkillParams
from skillet.scene.abstract.abstract_model import AbstractPlan

THighLevelObs = TypeVar("THighLevelObs", bound=BatchedObservation)
"""The type of the high level observation, batched."""
TLowLevelObs = TypeVar("TLowLevelObs", bound=BatchedObservation)
"""The type of the low level observation, batched."""
TSkillParams = TypeVar("TSkillParams", bound=SkillParams)
"""The type of the skill parameters, unbatched."""
TAction = TypeVar("TAction", bound=Action)
"""The type of the action, unbatched."""

TBHighLevelObs = TypeVar("TBHighLevelObs", bound=BatchedObservation)
"""The type of the high level observation, batched."""
TBLowLevelObs = TypeVar("TBLowLevelObs", bound=BatchedObservation)
"""The type of the low level observation, batched."""
TBAction = TypeVar("TBAction", bound=BatchedAction)
"""The type of the batched action, batched."""
TBSkillParams = TypeVar("TBSkillParams", bound=BatchedSkillParams)
"""The type of the skill parameters, batched."""

SelectedSkill: TypeAlias = int
"""The type of a selected skill. Alias of int."""
SelectedSkills = Int[torch.Tensor, "b"]
"""The indices of the selected skills for each environment according to the order of the skills."""


class Agent(ABC):
    def __init__(self):

        self._selected_skill = None
        self._plan = None

    @property
    def selected_skill(self) -> Skill:
        return self._selected_skill

    @property
    def plan(self) -> AbstractPlan:
        return self._plan
