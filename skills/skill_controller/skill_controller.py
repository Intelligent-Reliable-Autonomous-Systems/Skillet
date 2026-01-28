"""
skill_controller.py

The class for controlling low level skills 
"""

from .skills.base_skill import BaseSkill

class SkillController:

    current_skill: BaseSkill

    def __init__(self, cfg) -> None:
        """
        Initialize the skill controller
        """
        self.cfg = cfg

    def reset(self) -> None:
        """
        Reset the skill controller
        """ 
        pass

    def step(self, obs) -> object:
        """
        Take a step in the current skill
        """
        self.current_skill.step(obs)

    @property
    def is_done(self) -> bool:
        return self.current_skill.is_valid_terminal_state()