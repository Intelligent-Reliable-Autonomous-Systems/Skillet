"""Factory for creating Isaac Lab environments."""

import warnings
from typing import Any, cast

import gymnasium as gym
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from skillet.envs.isaac_lab import DirectRlInterface, ManagerBasedRlInterface


def create_isaac_env(task_name: str, cfg: dict[str, Any]) -> DirectRlInterface | ManagerBasedRlInterface:
    """Create an environment for the given task name.

    Cast as skillet-compatible environment interface.
    """
    return cast("DirectRlInterface | ManagerBasedRlInterface", _create_isaac_env(task_name, cfg))


def _create_isaac_env(task_name: str, cfg: dict[str, Any]) -> DirectRLEnv | ManagerBasedRLEnv:
    # Cabinet
    # TODO: construct configs explicitly for each task
    env_cfg = parse_env_cfg(task_name, **cfg)
    if task_name == "Kinova-Cabinet-Direct-v0":
        from kinova_tasks.isaac_tasks.direct.kinova_cabinet.kinova_cabinet_env import (
            KinovaCabinetEnv,
            KinovaCabinetEnvCfg,
        )

        # env_cfg = KinovaCabinetEnvCfg(**cfg)
        return KinovaCabinetEnv(env_cfg)
    # Lift Cube
    if task_name == "Kinova-Lift-Cube-Direct-v0":
        from kinova_tasks.isaac_tasks.direct.lift_cube.kinova_lift_cube_env import (
            KinovaLiftCubeEnv,
            KinovaLiftCubeEnvCfg,
        )

        # env_cfg = KinovaLiftCubeEnvCfg(**cfg)
        return KinovaLiftCubeEnv(env_cfg)
    if task_name == "Kinova-Lift-Cube-IK-v0":
        from kinova_tasks.isaac_tasks.direct.lift_cube.kinova_lift_cube_env import (
            KinovaLiftCubeEnvCfg,
            KinovaLiftCubeIKEnv,
        )

        # env_cfg = KinovaLiftCubeEnvCfg(**cfg)
        return KinovaLiftCubeIKEnv(env_cfg)
    if task_name == "Kinova-Lift-Cube-OSC-v0":
        from kinova_tasks.isaac_tasks.direct.lift_cube.kinova_lift_cube_env import (
            KinovaLiftCubeEnvCfg,
            KinovaLiftCubeOSCEnv,
        )

        # env_cfg = KinovaLiftCubeEnvCfg(**cfg)
        return KinovaLiftCubeOSCEnv(env_cfg)
    if task_name == "Franka-Lift-Cube-Direct-v0":
        from kinova_tasks.isaac_tasks.direct.lift_cube.franka_lift_cube_env import (
            FrankaLiftCubeEnv,
            FrankaLiftCubeEnvCfg,
        )

        # env_cfg = FrankaLiftCubeEnvCfg(**cfg)
        return FrankaLiftCubeEnv(env_cfg)
    if task_name == "Franka-Lift-Cube-IK-v0":
        from kinova_tasks.isaac_tasks.direct.lift_cube.franka_lift_cube_env import (
            FrankaLiftCubeEnvCfg,
            FrankaLiftCubeIKEnv,
        )

        # env_cfg = FrankaLiftCubeEnvCfg(**cfg)
        return FrankaLiftCubeIKEnv(env_cfg)
    if task_name == "Franka-Lift-Cube-OSC-v0":
        from kinova_tasks.isaac_tasks.direct.lift_cube.franka_lift_cube_env import (
            FrankaLiftCubeEnvCfg,
            FrankaLiftCubeOSCEnv,
        )

        # env_cfg = FrankaLiftCubeEnvCfg(**cfg)
        return FrankaLiftCubeOSCEnv(env_cfg)
    # Reach
    if task_name == "Kinova-Reach-Direct-v0":
        from kinova_tasks.isaac_tasks.direct.reach.kinova_reach_env import (
            KinovaReachEnv,
            KinovaReachEnvCfg,
        )

        # env_cfg = KinovaReachEnvCfg(**cfg)
        return KinovaReachEnv(env_cfg)
    if task_name == "Kinova-Reach-IK-v0":
        from kinova_tasks.isaac_tasks.direct.reach.kinova_reach_env import (
            KinovaReachEnvCfg,
            KinovaReachIKEnv,
        )

        # env_cfg = KinovaReachEnvCfg(**cfg)
        return KinovaReachIKEnv(env_cfg)
    if task_name == "Kinova-Reach-OSC-v0":
        from kinova_tasks.isaac_tasks.direct.reach.kinova_reach_env import (
            KinovaReachEnvCfg,
            KinovaReachOSCEnv,
        )

        # env_cfg = KinovaReachEnvCfg(**cfg)
        return KinovaReachOSCEnv(env_cfg)
    if task_name == "Kinova-Reach-No-Table-Direct-v0":
        from kinova_tasks.isaac_tasks.direct.reach.kinova_reach_env import (
            KinovaReachEnvCfg,
        )
        from kinova_tasks.isaac_tasks.direct.reach.kinova_reach_no_table_env import (
            KinovaReachNoTableEnv,
        )

        # env_cfg = KinovaReachEnvCfg(**cfg)
        return KinovaReachNoTableEnv(env_cfg)
    if task_name == "Franka-Reach-Direct-v0":
        from kinova_tasks.isaac_tasks.direct.reach.franka_reach_env import (
            FrankaReachEnv,
            FrankaReachEnvCfg,
        )

        # env_cfg = FrankaReachEnvCfg(**cfg)
        return FrankaReachEnv(env_cfg)
    if task_name == "Franka-Reach-IK-v0":
        from kinova_tasks.isaac_tasks.direct.reach.franka_reach_env import (
            FrankaReachEnvCfg,
            FrankaReachIKEnv,
        )

        # env_cfg = FrankaReachEnvCfg(**cfg)
        return FrankaReachIKEnv(env_cfg)
    if task_name == "Franka-Reach-OSC-v0":
        from kinova_tasks.isaac_tasks.direct.reach.franka_reach_env import (
            FrankaReachEnvCfg,
            FrankaReachOSCEnv,
        )

        # env_cfg = FrankaReachEnvCfg(**cfg)
        return FrankaReachOSCEnv(env_cfg)
    # Manager-based
    if task_name == "Franka-Lift-Cube-v0":
        from kinova_tasks.isaac_tasks.manager_based.franka_lift_cube.franka_lift_env_cfg import (
            FrankaCubeLiftEnvCfg,
        )

        # env_cfg = FrankaCubeLiftEnvCfg(**cfg)
        return ManagerBasedRLEnv(env_cfg)
    if task_name == "Kinova-Lift-Cube-IK-Rel-v0":
        from kinova_tasks.isaac_tasks.manager_based.kinova_lift_cube.kinova_lift_env_cfg import (
            TeleOpKinovaCubeLiftEnvCfg,
        )

        # env_cfg = TeleOpKinovaCubeLiftEnvCfg(**cfg)
        return ManagerBasedRLEnv(env_cfg)
    if task_name == "Kinova-Lift-Cube-v0":
        from kinova_tasks.isaac_tasks.manager_based.kinova_lift_cube.kinova_lift_env_cfg import (
            KinovaLiftCubeEnvCfg,
        )

        # env_cfg = KinovaLiftCubeEnvCfg(**cfg)
        return ManagerBasedRLEnv(env_cfg)
    # Manager-based Reach
    if task_name == "Kinova-Reach-v0":
        from kinova_tasks.isaac_tasks.manager_based.kinova_reach.kinova_reach_env_cfg import (
            KinovaReachEnvCfg,
        )

        # env_cfg = KinovaReachEnvCfg(**cfg)
        return ManagerBasedRLEnv(env_cfg)
    if task_name == "Kinova-Reach-IK-Rel-v0":
        from kinova_tasks.isaac_tasks.manager_based.kinova_reach.kinova_reach_env_cfg import (
            TeleOpKinovaReachEnvCfg,
        )

        # env_cfg = TeleOpKinovaReachEnvCfg(**cfg)
        return ManagerBasedRLEnv(env_cfg)

    warnings.warn(f"{task_name} cannot be explicitly constructed. Falling back to gym.make.", stacklevel=1)
    return cast("DirectRLEnv | ManagerBasedRLEnv", gym.make(task_name, cfg=cfg))
