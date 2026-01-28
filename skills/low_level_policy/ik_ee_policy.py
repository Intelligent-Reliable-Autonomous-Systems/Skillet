"""
ik_ee_policy.py

The low level controller class for a Differential IK Controller 

Written by Will Solow & Jeff Jewett, 2026
"""

from .low_level_policy import LowLevelPolicy

class IKEEPolicy(LowLevelPolicy):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def reset(self) -> None:
        """
        Reset the low level policy
        """
        pass

    def get_action(self, obs) -> object:
        """
        Get the next low level action for the robot based on 
        the inverse kinematics of the robot 
        """
        pass