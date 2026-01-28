"""
reach_xyz.py

The reach XYZ skill, moving the end effector to a location in XYZ space relative
to the robot base

Written by Will Solow & Jeff Jewett, 2026
"""

from .base_skill import BaseSkill

class Reach_XYZ(BaseSkill):

    def __init__(self, cfg) -> None:
        super().__init__(self, cfg)

    