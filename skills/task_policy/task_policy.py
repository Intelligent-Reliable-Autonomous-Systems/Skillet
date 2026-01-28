"""
task_policy.py

The high level controller class for skills 

Written by Will Solow & Jeff Jewett, 2026
"""

from abc import abstractmethod

from ..skill_controller.skill_controller import SkillController

class IKEEPolicy:

    skill_controller: SkillController

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def reset(self) -> None:
        """
        Reset the low level policy
        """
        pass

    @abstractmethod
    def get_skill(self, obs) -> object:
        """
        Get the next skill for the robot to execute
        """
        pass

    @abstractmethod
    def execute_skill(self) -> None:
        """
        Need something for executing the skill and also querying the environment
        """
        while not self.skill_controller.is_done:
            self.skill_controller.step(obs)

