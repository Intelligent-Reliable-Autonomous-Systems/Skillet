"""Factory for creating Isaac Lab environments."""

import warnings
from typing import Any, cast

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg

from skillet.envs.compatibility import DirectRlInterface, ManagerBasedRlInterface
from skillet.envs.isaac import IsaacDirectRlEnv, IsaacManagerBasedRlEnv


def create_isaac_env(task_name: str, cfg: dict[str, Any]) -> DirectRlInterface | ManagerBasedRlInterface:
    """Create an environment for the given task name.

    Cast as skillet-compatible environment interface.
    """
    return cast("DirectRlInterface | ManagerBasedRlInterface", _create_isaac_env(task_name, cfg))


def _create_isaac_env(task_name: str, cfg: dict[str, Any]) -> IsaacDirectRlEnv | IsaacManagerBasedRlEnv:
    # Cabinet
    env_cfg = parse_env_cfg(task_name, **cfg)
    if task_name == "Gen3-Cabinet-Direct-v0":
        from skillet_tasks.isaac_tasks.direct.gen3_cabinet.gen3_cabinet_env import (
            Gen3CabinetEnv,
        )

        return Gen3CabinetEnv(env_cfg)
    # Lift Cube
    if task_name == "Gen3-Lift-Cube-Direct-v0":
        from skillet_tasks.isaac_tasks.direct.lift_cube.gen3_lift_cube_env import (
            Gen3LiftCubeEnv,
        )

        return Gen3LiftCubeEnv(env_cfg)
    if task_name == "Gen3-Lift-Cube-IK-v0":
        from skillet_tasks.isaac_tasks.direct.lift_cube.gen3_lift_cube_env import (
            Gen3LiftCubeIKEnv,
        )

        return Gen3LiftCubeIKEnv(env_cfg)
    if task_name == "Gen3-Lift-Cube-OSC-v0":
        from skillet_tasks.isaac_tasks.direct.lift_cube.gen3_lift_cube_env import (
            Gen3LiftCubeOSCEnv,
        )

        return Gen3LiftCubeOSCEnv(env_cfg)
    if task_name == "Franka-Lift-Cube-Direct-v0":
        from skillet_tasks.isaac_tasks.direct.lift_cube.franka_lift_cube_env import (
            FrankaLiftCubeEnv,
        )

        return FrankaLiftCubeEnv(env_cfg)
    if task_name == "Franka-Lift-Cube-IK-v0":
        from skillet_tasks.isaac_tasks.direct.lift_cube.franka_lift_cube_env import (
            FrankaLiftCubeIKEnv,
        )

        return FrankaLiftCubeIKEnv(env_cfg)
    if task_name == "Franka-Lift-Cube-OSC-v0":
        from skillet_tasks.isaac_tasks.direct.lift_cube.franka_lift_cube_env import (
            FrankaLiftCubeOSCEnv,
        )

        return FrankaLiftCubeOSCEnv(env_cfg)
    # Reach
    if task_name == "Gen3-Reach-Direct-v0":
        from skillet_tasks.isaac_tasks.direct.reach.gen3_reach_env import (
            Gen3ReachEnv,
        )

        return Gen3ReachEnv(env_cfg)
    if task_name == "Gen3-Reach-IK-v0":
        from skillet_tasks.isaac_tasks.direct.reach.gen3_reach_env import (
            Gen3ReachIKEnv,
        )

        return Gen3ReachIKEnv(env_cfg)
    if task_name == "Gen3-Reach-OSC-v0":
        from skillet_tasks.isaac_tasks.direct.reach.gen3_reach_env import (
            Gen3ReachOSCEnv,
        )

        return Gen3ReachOSCEnv(env_cfg)
    if task_name == "Gen3-Reach-No-Table-Direct-v0":
        from skillet_tasks.isaac_tasks.direct.reach.gen3_reach_no_table_env import (
            Gen3ReachNoTableEnv,
        )

        return Gen3ReachNoTableEnv(env_cfg)
    if task_name == "Franka-Reach-Direct-v0":
        from skillet_tasks.isaac_tasks.direct.reach.franka_reach_env import (
            FrankaReachEnv,
        )

        return FrankaReachEnv(env_cfg)
    if task_name == "Franka-Reach-IK-v0":
        from skillet_tasks.isaac_tasks.direct.reach.franka_reach_env import (
            FrankaReachIKEnv,
        )

        return FrankaReachIKEnv(env_cfg)
    if task_name == "Franka-Reach-OSC-v0":
        from skillet_tasks.isaac_tasks.direct.reach.franka_reach_env import (
            FrankaReachOSCEnv,
        )

        return FrankaReachOSCEnv(env_cfg)
    # Manager-based
    if task_name == "Franka-Lift-Cube-v0":
        # env_cfg = FrankaCubeLiftEnvCfg(**cfg)
        return IsaacManagerBasedRlEnv(env_cfg)
    if task_name == "Gen3-Lift-Cube-IK-Rel-v0":
        return IsaacManagerBasedRlEnv(env_cfg)
    if task_name == "Gen3-Lift-Cube-v0":
        return IsaacManagerBasedRlEnv(env_cfg)
    # Manager-based Reach
    if task_name == "Gen3-Reach-v0":
        # env_cfg = Gen3ReachEnvCfg(**cfg)
        return IsaacManagerBasedRlEnv(env_cfg)
    if task_name == "Gen3-Reach-IK-Rel-v0":
        return IsaacManagerBasedRlEnv(env_cfg)

    warnings.warn(f"{task_name} cannot be explicitly constructed. Falling back to gym.make.", stacklevel=1)
    return cast("IsaacDirectRlEnv | IsaacManagerBasedRlEnv", gym.make(task_name, cfg=cfg))
