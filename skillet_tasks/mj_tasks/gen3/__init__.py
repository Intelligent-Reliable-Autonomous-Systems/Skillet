import gymnasium as gym

from . import agents

gym.register(
    id="Mj-Gen3-v0",
    entry_point=f"{__name__}.mj_gen3:MjGen3Env",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mj_gen3:MjGen3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3ReachPPORunnerCfg",
    },
    disable_env_checker=True,
)
