"""Kinova Gen3 manipulation tasks for mjlab."""

import gymnasium as gym

from . import agents

gym.register(
    id="MJ-Kinova-Lift-Cube-v0",
    entry_point="skillet.envs.mujoco:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_lift_cube_env_cfg:KinovaLiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="MJ-Kinova-Lift-Cube-IK-v0",
    entry_point="skillet.envs.mujoco:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_lift_cube_env_cfg:KinovaLiftCubeIKEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KinovaLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
