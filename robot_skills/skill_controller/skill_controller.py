"""skill_controller.py.

The class for controlling low level skills

Written by Will Solow and Jeff Jewett, 2026
"""

import numpy as np
import torch

from robot_skills.skill_controller.skills import BaseSkill
from robot_skills.task_policy import TaskPolicyCfg
from robot_skills.utils import get_subclasses


class SkillController:
    """Base class for controlling executing of skills.

    This class executes each available skill in paralell with the number of environments executing that skill
    """

    current_skills_list: np.ndarray
    skills_params: torch.Tensor

    AVAIL_SKILLS = get_subclasses("robot_skills.skill_controller.skills", "BaseSkill")

    NUM_ROBOT_JOINTS = 7

    def __init__(self, cfg: TaskPolicyCfg) -> None:
        """Initialize the skill controller.

        Args:
            cfg: The configuration

        """
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = cfg.device
        self.env_ids_list = torch.arange(self.num_envs, device=self.device)

        for skill_name in cfg.skills:
            assert skill_name in self.AVAIL_SKILLS

        self.skills_names = [skill_name for skill_name in cfg.skills if skill_name in self.AVAIL_SKILLS]
        self.num_skills = len(self.skills_names)

        self.skills = [
            self.AVAIL_SKILLS[sk](cfg.skills_cfgs[i], device=self.device) for i, sk in enumerate(self.skills_names)
        ]

        self._max_skills_steps = torch.zeros(self.num_envs, device=self.device)

        print(f"[INFO][SkillController] Created Skill Controller with Skills: {self.skills_names}")

    def reset(self, skills: np.ndarray, params: torch.Tensor) -> None:
        """Reset the skill controller.

        Args:
            skills: A list of skills
            params: A parameterization for each of the skills

        """
        self.current_skills_list = skills
        self.skills_params = params
        self._actions = torch.zeros((self.num_envs, self.NUM_ROBOT_JOINTS), device=self.device)
        self._dones = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        for i, sk in enumerate(self.skills_names):
            env_ids = self.env_ids_list[np.where(sk == self.current_skills_list)[0]]
            if env_ids.shape[0] == 0:
                continue
            self._max_skills_steps[env_ids] = self.skills[i]._max_steps
            self.skills[i].reset(env_ids=env_ids, skill_params=self.skills_params[env_ids])

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the next low level action for the robot.

        Step through each of the skills for the skill controller

        Args:
            obs: Observation tensor of shape (N, obs_dim)

        Returns:
            A tensor of shape (N, num_joints)

        """
        for i, sk in enumerate(self.skills_names):
            env_ids = self.env_ids_list[np.where(sk == self.current_skills_list)[0]]
            if env_ids.shape[0] == 0:
                continue
            self._actions[env_ids], self._dones[env_ids] = self.skills[i].step(obs[env_ids])

        return self._actions

    @property
    def is_done(self) -> torch.Tensor:
        """Check if the skill controller is done.

        Returns:
            A tensor of dones for each skill that is running.

        """
        return self._dones
