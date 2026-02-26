"""Kinova Gen3 lift task with joint-space actions."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from kinova_tasks.assets.mujoco.kinova import KINOVA_ACTION_SCALE

from .lift_cube_base_env_cfg import kinova_base_env_cfg


def kinova_lift_cube_joint_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Kinova lift cube with joint-space position control."""
    cfg = kinova_base_env_cfg(play=play)

    # Joint position actions for arm + gripper
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KINOVA_ACTION_SCALE

    return cfg
