"""Franka-Cabinet environment."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Gen3-Reach-Direct-v0",
    entry_point=f"{__name__}.gen3_reach_env:Gen3ReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gen3_reach_env:Gen3ReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Gen3-Reach-IK-v0",
    entry_point=f"{__name__}.gen3_reach_env:Gen3ReachIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gen3_reach_env:Gen3ReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Gen3-Reach-OSC-v0",
    entry_point=f"{__name__}.gen3_reach_env:Gen3ReachOSCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gen3_reach_env:Gen3ReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3ReachPPORunnerCfg",
    },
)

gym.register(
    id="Franka-Reach-Direct-v0",
    entry_point=f"{__name__}.franka_reach_env:FrankaReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_reach_env:FrankaReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
    },
)

gym.register(
    id="Franka-Reach-IK-v0",
    entry_point=f"{__name__}.franka_reach_env:FrankaReachIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_reach_env:FrankaReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
    },
)

gym.register(
    id="Franka-Reach-OSC-v0",
    entry_point=f"{__name__}.franka_reach_env:FrankaReachOSCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_reach_env:FrankaReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
    },
)
