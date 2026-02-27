"""Kinova Gen3 manipulation tasks for mjlab."""

import gymnasium as gym

from . import agents
from .lift_cube_ik_env_cfg import kinova_lift_ik_env_cfg as kinova_lift_ik_env_cfg
from .lift_cube_joint_env_cfg import kinova_lift_cube_joint_env_cfg as kinova_lift_cube_joint_env_cfg

gym.register(
    id="MJ-Lift-Cube-Kinova-v0",
    entry_point="skillet.envs.mujoco:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": kinova_lift_cube_joint_env_cfg(),  # f"{__name__}.lift_cube_base_env_cfg:MJKinovaLiftCubeCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
