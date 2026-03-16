import gymnasium as gym

from . import agents

gym.register(
    id="MJ-Gen3-Reach-Direct-v0",
    entry_point=f"{__name__}.gen3_reach_env:Gen3ReachEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gen3_reach_env:Gen3ReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3ReachPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="MJ-Gen3Lite-Reach-Direct-v0",
    entry_point=f"{__name__}.gen3lite_reach_env:Gen3LiteReachEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gen3lite_reach_env:Gen3LiteReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3ReachPPORunnerCfg",
    },
    disable_env_checker=True,
)
