"""
skill_env_wrapper.py

A wrapper around IsaacSim/ROS2/etc environments to handle the passing
of actions to the underlying simulation/hardware

Written by Will Solow and Jeff Jewett, 2026
"""

from abc import abstractmethod

import gymnasium as gym
import torch

class SkillEnvWrapper:

    def __init__(self, env:gym.Env) -> None:
        
        self.env = env

    @abstractmethod
    def step(self, action: torch.Tensor):
        pass
    
