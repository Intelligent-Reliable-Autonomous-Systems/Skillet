from typing import Any

from skillet.envs.mujoco import MjDirectRlEnv
from skillet_tasks.mj_tasks.gen3.mj_gen3 import MjGen3Env, MjGen3EnvCfg


def create_mj_env(task_name: str, cfg: dict[str, Any]) -> MjDirectRlEnv:
    """Create a mujoco environment for the given task name and configuration."""
    if task_name == "Mj-Gen3-v0":
        env_cfg = MjGen3EnvCfg(**cfg)
        return MjGen3Env(
            cfg=env_cfg,
        )

    raise ValueError(f"Invalid task name: {task_name}")
