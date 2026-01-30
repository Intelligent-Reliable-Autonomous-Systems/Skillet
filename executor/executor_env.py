"""executor.py.

Main executor class to be run in parallel to the task/policy chain

Written by Will Solow & Jeff Jewett, 2026
"""

import torch

from env.skill_env_wrapper import SkillEnvWrapper
from robot_skills import PolicyCfg, get_subclasses
from robot_skills.high_level_obs import HighLevelObs
from robot_skills.task_policy import TaskPolicy


class SkillExecutor:
    """The main class for executing skills.

    This assumes access to a gymansium environment to be executed in paralell
    """

    env: SkillEnvWrapper
    task_policy: TaskPolicy
    obs_converter: HighLevelObs

    def __init__(self, cfg: PolicyCfg, env: SkillEnvWrapper) -> None:
        """Initialize the environment.

        Args:
            cfg: Config dictionary
            env: A SkillEnvWrapper environment of a wrapped IsaacLab/ROS2 environment

        Returns:
            None

        """
        cfg, env = self._fill_cfg_from_env(cfg, env)
        self.cfg = cfg
        self.env = env

        self.num_envs = self.env.num_envs
        self.device = self.env.device
        self.task_policy = get_subclasses("robot_skills.task_policy", "TaskPolicy")[cfg.task_policy.task_policy_name](
            cfg.task_policy
        )
        self.obs_converter = HighLevelObs()

    def execute(self) -> None:
        """Execute a run of the environment.

        Args:
            None

        Returns:
            None

        """
        self.task_policy.reset()
        obs, _ = self.env.reset()
        dones = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.task_policy.reset()
        while not dones.all():
            high_level_obs = self.obs_converter.get_high_level_obs(obs)
            self.task_policy.set_skills_and_params(high_level_obs)

            while not self.task_policy.is_skills_done:
                action = self.task_policy.low_level_step(obs)
                obs, _, term, trunc, _ = self.env.step(action)
            dones = torch.logical_or(term, trunc)

        return

    def _fill_cfg_from_env(self, cfg: PolicyCfg, env: SkillEnvWrapper) -> tuple[PolicyCfg, SkillEnvWrapper]:
        """Fill the configclass with data from the environment.

        Args:
            cfg: Config dictionary
            env: A SkillEnvWrapper environment of a wrapped IsaacLab/ROS2 environment

        Returns:
            None

        """
        print("[INFO][SkillExecutor] Filling cfg from environment (action space)")
        cfg.task_policy.num_envs = env.num_envs
        cfg.task_policy.device = env.device
        for low_level_cfg in cfg.task_policy.skills_cfgs:
            low_level_cfg.output_dim = env.num_actions
        cfg.task_policy.action_dim = env.num_actions
        return cfg, env
