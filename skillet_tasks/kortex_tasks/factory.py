from typing import Any

from skillet.envs.kortex import KortexEnv
from skillet.envs.kortex.kortex_bridge import setup_kortex
from skillet_tasks.kortex_tasks.gen3.gen3_kortex import Gen3KortexEnv, Gen3KortexEnvCfg
from skillet_tasks.kortex_tasks.gen3_lite.gen3lite_kortex import Gen3LiteKortexEnv, Gen3LiteKortexEnvCfg


def create_kortex_env(task_name: str, cfg: dict[str, Any]) -> KortexEnv:
    """Create a Kortex environment for the given task name and configuration."""
    kortex, kortex_cyclic = setup_kortex(ip=cfg.robot_ip)
    if task_name == "Kortex-Gen3-v0":
        env_cfg = Gen3KortexEnvCfg(**cfg)
        return Gen3KortexEnv(cfg=env_cfg, kortex=kortex, kortex_c=kortex_cyclic)
    if task_name == "Kortex-Gen3Lite-v0":
        env_cfg = Gen3LiteKortexEnvCfg(**cfg)
        return Gen3LiteKortexEnv(cfg=env_cfg, kortex=kortex, kortex_c=kortex_cyclic)

    raise ValueError(f"Invalid task name: {task_name}")
