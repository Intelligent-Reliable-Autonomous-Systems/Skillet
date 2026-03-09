"""Franka-Cabinet environment."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="MJ-Gen3-Lift-Cube-Direct-v0",
    entry_point=f"{__name__}.gen3_lift_cube_env:Gen3LiftCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gen3_lift_cube_env:Gen3LiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3LiftCubePPORunnerCfg",
    },
)
