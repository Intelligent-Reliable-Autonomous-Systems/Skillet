"""A skill controller to handle the execution of multiple skills.

Written by Will Solow, 2026
"""

import numpy as np
import torch

from skillet.core.env import AsGymVectorEnv
from skillet.core.policy import TBAction, TBPolicyObs
from skillet.skill.skill_lib import SKILL_LIB


class SkillController:
    """Class for contrilling skills in an RL environment."""

    def __init__(self, skills: list[str], num_envs: int, env: AsGymVectorEnv, device: str = "cuda") -> None:
        """Initialize the skill controller based on the list of skills."""
        self.skill_names = skills

        for sk_name in self.skill_names:
            assert sk_name in SKILL_LIB, f"{sk_name} not in SKILL LIB: {SKILL_LIB.keys()}"
        self.skills = [SKILL_LIB[sk](env) for sk in self.skill_names]

        self.device = device
        self.num_envs = num_envs
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.param_dim = int(np.max([skill.param_dim for skill in self.skills]))
        self.num_skills = len(self.skills)
        self.action_dim = self.num_skills + self.param_dim
        self._env_action_dim = int(np.prod(env.single_action_space.shape))
        self._obs_func = env.get_observation
        self.num_calls = 0

        self._dones = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._statuses = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._rewards = torch.zeros((self.num_envs,), device=self.device)

    @property
    def dones(self) -> torch.Tensor:
        """Check if all skills are done."""
        for i, sk in enumerate(self.skills):
            sk_env_ids = self.env_ids[self._skills_idx == i]
            if sk_env_ids.shape[0] == 0:
                continue
            term_idx = sk.is_terminated(self._obs_func(sk.obs_spec)[sk_env_ids])
            self._dones[sk_env_ids] = term_idx
        return self._dones

    @property
    def statuses(self) -> torch.Tensor:
        """Check the status of each skill."""
        for i, sk in enumerate(self.skills):
            sk_env_ids = self.env_ids[self._skills_idx == i]
            if sk_env_ids.shape[0] == 0:
                continue
            stat_idx = sk.status(self._obs_func(sk.obs_spec)[sk_env_ids])
            self._statuses[sk_env_ids] = stat_idx
        return self._statuses

    @property
    def successes(self) -> torch.Tensor:
        """Check the success of each skill."""
        for i, sk in enumerate(self.skills):
            sk_env_ids = self.env_ids[self._skills_idx == i]
            if sk_env_ids.shape[0] == 0:
                continue
            success_idx = sk.is_success(self._obs_func(sk.obs_spec)[sk_env_ids])
            self._successes[sk_env_ids] = success_idx
        return self._successes

    def reset(self, action: TBAction) -> None:
        """Reset the skill controller based on the action parameters.

        Args:
            action: A torch Tensor of shape (num_envs, num_skills+max_param_dim).

        """
        assert (
            action.shape[-1] == self.action_dim
        ), f"Action dimension {action.shape[-1]} does not match expected skill dimension {self.action_dim}"
        self._skills_idx = self.get_skill_from_action(action)
        self._skills_params = self.get_params_from_action(action)

        for i, sk in enumerate(self.skills):
            sk_env_ids = self.env_ids[self._skills_idx == i]
            if sk_env_ids.shape[0] == 0:
                continue
            sk.initiate(self._obs_func(sk.obs_spec)[sk_env_ids], self._skills_params[sk_env_ids])
        self._action = torch.zeros((self.num_envs, self._env_action_dim), device=self.device)
        self._dones = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._statuses = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._rewards = torch.zeros((self.num_envs,), device=self.device)
        self.num_calls = 0

    def get_action(self, obs: TBPolicyObs) -> TBAction:
        """Step through the skills, getting the next joint position for each.

        Args:
            obs: The current observation from the environment

        Returns:
            TBAction in shape (num_envs, num_joints)

        """
        action = torch.zeros((self.num_envs, self._env_action_dim), device=self.device)
        for i, sk in enumerate(self.skills):
            sk_env_ids = self.env_ids[self._skills_idx == i]

            if sk_env_ids.shape[0] == 0:
                continue
            action[sk_env_ids] = sk.get_action(self._obs_func(sk.obs_spec)[sk_env_ids])

        self._action = action
        self.num_calls += 1

        return self._action

    def get_skill_from_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Return a tensor of shape (num_envs,) denoting each skill to use."""
        return torch.argmax(actions[:, : self.num_skills], dim=1)

    def get_params_from_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Return a tensor of shape (num_envs, param_dim) denoting each skill parameter."""
        return actions[:, -self.param_dim :]

    def post_process_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Post process the reward according to skill success.

        Args:
            reward: Tensor of shape (num_envs,) for the reward accumulated during the skill.

        """
        success_pen_fac = 1.0  # TODO scale as needed
        sk_penalty_fac = 0.5
        sk_success = self.successes
        reward = torch.where(sk_success, reward, reward / success_pen_fac)

        sk_penalty = 1.0 - torch.where(sk_success, 1.0, self._compute_sk_reward())

        return reward - sk_penalty_fac * sk_penalty

    def _compute_sk_reward(self) -> torch.Tensor:
        for i, sk in enumerate(self.skills):
            sk_env_ids = self.env_ids[self._skills_idx == i]
            if sk_env_ids.shape[0] == 0:
                continue
            reward = sk.reward(self._obs_func(sk.obs_spec)[sk_env_ids])
            self._rewards[sk_env_ids] = reward
        return self._rewards
