from abc import ABC
from skillet.core.skill import Skill
from skillet.scene.abstract.abstract_model import AbstractPlan


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
