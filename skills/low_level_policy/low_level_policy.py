"""
low_level_policy.py

The abstract class for low level control policies.
Subclasses are End Effector Policies and RL Policies

Written by Will Solow & Jeff Jewett, 2026
"""

from abc import abstractmethod

class LowLevelPolicy:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the low level policy
        """
        pass

    @abstractmethod
    def get_action(self, obs) -> object:
        """
        Get the next low level action for the robot based on
        the observation passed to the low level controller
        """
        pass