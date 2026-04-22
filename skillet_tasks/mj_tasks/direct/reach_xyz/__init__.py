import gymnasium as gym

from . import agents

gym.register(
    id="Mj-Gen3-ReachXyz-Direct-v0",
    entry_point=f"{__name__}.gen3_reachxyz_env:Gen3ReachXyzEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gen3_reachxyz_env:Gen3ReachXyzEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3ReachXyzPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Mj-Gen3Lite-ReachXYZ-Direct-v0",
    entry_point=f"{__name__}.gen3lite_reachxyz_env:Gen3LiteReachXyzEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gen3lite_reachxyz_env:Gen3LiteReachXyzEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3ReachXyzPPORunnerCfg",
    },
    disable_env_checker=True,
)
