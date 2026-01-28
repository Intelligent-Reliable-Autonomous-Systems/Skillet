"""
base_skill.py

The abstract class for skills (options) which define their starting/ending conditions,
low level controller, and success condition

Written by Will Solow & Jeff Jewett, 2026
"""

from abc import abstractmethod
from typing import Tuple

import numpy as np

from ...low_level_policy.low_level_policy import LowLevelPolicy

class BaseSkill:

    low_level_policy: LowLevelPolicy

    def __init__(self, cfg) -> None:
        """
        Initialization, should define starting/ending conditions, low level controller
        and success condition
        """
        self.cfg = cfg

    def reset(self) -> None:
        """
        Reset the skill
        """

    def step(self, obs) -> Tuple[np.ndarray, bool]:
        """
        Take a step in the low level policy controller. Also update internals
        for what the skill should be doing next and if it is done
        """
        action = self.low_level_policy.get_action(obs)

        return action

    @abstractmethod
    def is_valid_initial_state(self) -> bool:
        """
        Test if the current state is a valid initial state for the skill
        """
        pass

    @abstractmethod
    def is_valid_terminal_state(self) -> bool:
        """
        Test if the current state is a valid terminal state
        """
        pass
    
    @abstractmethod
    def is_success(self) -> bool:
        """
        Test if the skill successfully completed
        """

    