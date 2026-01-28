"""
rl_policy.py

The low level controller class for a RL controller

Written by Will Solow & Jeff Jewett, 2026
"""

from .low_level_policy import LowLevelPolicy

class RLPolicy(LowLevelPolicy):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def reset(self) -> None:
        """
        Reset the rl policy
        """
        pass

    def get_action(self, obs) -> object:
        """
        Get the next low level action for the robot based on 
        the RL policy
        """
        pass