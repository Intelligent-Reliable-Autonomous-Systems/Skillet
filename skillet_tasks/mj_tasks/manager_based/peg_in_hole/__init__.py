"""Kinova Gen3 manipulation tasks for mjlab."""

import gymnasium as gym

from . import agents

gym.register(
    id="MJ-Gen3-Peg-In_Hole-v0",
    entry_point="skillet.envs.mujoco:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_peg_in_hole_env_cfg:KinovaPegInHoleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaPegInHolePPORunnerCfg",
    },
    disable_env_checker=True,
)
