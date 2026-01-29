"""
isaac_env_wrapper.py

A wrapper around IsaacLab Gym environments

Written by Will Solow and Jeff Jewett, 2026
"""

import gymansium as gym
from .skill_env_wrapper import SkillEnvWrapper

class ROS2EnvWrapper(SkillEnvWrapper):

    def __init__(self, env:gym.Env) -> None:
        super().__init__(env)

    def step():
        pass