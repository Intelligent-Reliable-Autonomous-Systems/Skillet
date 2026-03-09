"""Kinova Gen3 manipulation tasks for mjlab."""

import gymnasium as gym

from . import agents

gym.register(
    id="MJ-Gen3-Peg-In-Hole-v0",
    entry_point="skillet.envs.mujoco:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gen3_peg_in_hole_env_cfg:Gen3PegInHoleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3PegInHolePPORunnerCfg",
    },
    disable_env_checker=True,
)
