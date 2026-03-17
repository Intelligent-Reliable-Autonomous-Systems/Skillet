"""A library for skills ready to be initialized.

Written by Will Solow, 2026
"""

import gymnasium as gym
import torch
from jaxtyping import Float

from skillet.core.skill import Skill
from skillet.core.spaces import ObservationSpec
from skillet.envs.compatibility import GymVectorInterface
from skillet.envs.specs import BxM_Action, BxN_Obs
from skillet.policy.ik_ee import IKEEPolicy, PosAbsIKEEPolicy, PoseAbsIKEEPolicy, PoseRelIKEEPolicy, XYZRPYAbsIKEEPolicy
from skillet.policy.joint_pos import GripperPolicy, JointPosPolicy
from skillet.policy.osc_ee import PoseAbsOSCEEPolicy
from skillet.skill.high_level import GraspXYZSkill, PickSkill, PlaceSkill, PushSkill
from skillet.skill.low_level import (
    GripperGraspSkill,
    GripperOCSkill,
    GripperOpenSkill,
    JointPosSkill,
    OrientRPYSkill,
    OrientYSkill,
    ReachXYZRPYSkill,
    ReachXYZSkill,
)


def make_joint_obs_spec(device: str = "cuda") -> ObservationSpec:
    """Make an observation spec for IK controllers."""
    return ObservationSpec[Float[torch.Tensor, "b ..."]](
        space=gym.spaces.Dict(),
        name="ik_ee",
        is_torch=True,
        is_batched=True,
        n_envs=-1,
        device=device,
    )


def make_ik_obs_spec(device: str = "cuda") -> ObservationSpec:
    """Make an observation spec for IK controllers."""
    return ObservationSpec[Float[torch.Tensor, "b ..."]](
        space=gym.spaces.Dict(),
        name="ik_ee",
        is_torch=True,
        is_batched=True,
        n_envs=-1,
        device=device,
    )


def make_osc_obs_spec(device: str = "cuda") -> ObservationSpec:
    """Make an observation spec for OSC controllers."""
    return ObservationSpec[Float[torch.Tensor, "b ..."]](
        space=gym.spaces.Dict(),
        name="osc",
        is_torch=True,
        is_batched=True,
        n_envs=-1,
        device=device,
    )


def make_osc_ee_pose_policy(env: GymVectorInterface) -> IKEEPolicy:
    return PoseAbsOSCEEPolicy[BxN_Obs, BxM_Action](make_osc_obs_spec(env.device), env.action_spec)


def make_ik_ee_xyzrpy_policy(env: GymVectorInterface) -> IKEEPolicy:
    return XYZRPYAbsIKEEPolicy[BxM_Action](make_ik_obs_spec(env.device), env.action_spec)


def make_ik_ee_pose_policy(env: GymVectorInterface) -> IKEEPolicy:
    return PoseAbsIKEEPolicy[BxM_Action](make_ik_obs_spec(env.device), env.action_spec)


def make_ik_ee_pos_policy(env: GymVectorInterface) -> IKEEPolicy:
    return PosAbsIKEEPolicy[BxM_Action](make_ik_obs_spec(env.device), env.action_spec)


def make_rel_ik_ee_pose_policy(env: GymVectorInterface) -> IKEEPolicy:
    return PoseRelIKEEPolicy[BxM_Action](make_ik_obs_spec(env.device), env.action_spec)


def make_rel_ik_ee_pose_policy(env: GymVectorInterface) -> IKEEPolicy:
    return PoseRelIKEEPolicy[BxM_Action](make_ik_obs_spec(env.device), env.action_spec)


def make_gripper_policy(env: GymVectorInterface) -> GripperPolicy:
    return GripperPolicy[BxN_Obs, BxM_Action](make_joint_obs_spec(env.device), env.action_spec)


def make_joint_pos_policy(env: GymVectorInterface) -> JointPosPolicy:
    return JointPosPolicy[BxN_Obs, BxM_Action](make_joint_obs_spec(env.device), env.action_spec)


def make_reach_xyzrpy_skill(env: GymVectorInterface, skill_length: int = 15) -> Skill:
    return ReachXYZRPYSkill[BxM_Action, None](
        name="reach_xyzrpy_skill", policy=make_ik_ee_pose_policy(env), length=skill_length
    )


def make_rel_reach_xyzrpy_skill(env: GymVectorInterface, skill_length: int = 5) -> Skill:
    return ReachXYZRPYSkill[BxM_Action, None](
        name="rel_reach_xyzrpy_skill", policy=make_rel_ik_ee_pose_policy(env), length=skill_length
    )


def make_rel_reach_xyzrpy_skill(env: GymVectorInterface, skill_length: int = 5) -> Skill:
    return ReachXYZRPYSkill[BxM_Action, None](
        name="rel_reach_xyzrpy_skill", policy=make_rel_ik_ee_pose_policy(env), length=skill_length
    )


def make_reach_xyz_skill(env: GymVectorInterface, skill_length: int = 15) -> Skill:
    return ReachXYZSkill[BxN_Obs, BxM_Action, None](
        name="reach_xyz_skill", policy=make_ik_ee_pos_policy(env), length=skill_length, clip=True
    )


def make_orient_rpy_skill(env: GymVectorInterface, skill_length: int = 15) -> Skill:
    return OrientRPYSkill[BxN_Obs, BxM_Action, None](
        name="orient_rpy_skill", policy=make_ik_ee_pose_policy(env), length=skill_length
    )


def make_orient_y_skill(env: GymVectorInterface, skill_length: int = 15) -> Skill:
    return OrientYSkill[BxN_Obs, BxM_Action, None](
        name="orient_y_skill", policy=make_ik_ee_pose_policy(env), length=skill_length
    )


def make_gripper_oc_skill(env: GymVectorInterface, skill_length: int = 4) -> Skill:
    return GripperOCSkill[BxN_Obs, BxM_Action, None](
        name="gripper_oc_skill", policy=make_gripper_policy(env), length=skill_length
    )


def make_gripper_o_skill(env: GymVectorInterface, skill_length: int = 4) -> Skill:
    return GripperOpenSkill[BxN_Obs, BxM_Action, None](
        name="gripper_o_skill", policy=make_gripper_policy(env), length=skill_length
    )


def make_gripper_c_skill(env: GymVectorInterface, skill_length: int = 4) -> Skill:
    return GripperGraspSkill[BxN_Obs, BxM_Action, None](
        name="gripper_o_skill", policy=make_gripper_policy(env), length=skill_length
    )


def make_joint_pos_skill(env: GymVectorInterface, skill_length: int = 15) -> Skill:
    return JointPosSkill[BxN_Obs, BxM_Action, None](
        name="joint_pos_skill", policy=make_joint_pos_policy(env), length=skill_length
    )


def make_pick_skill(env: GymVectorInterface, lift_height: float = 0.23, skill_length: int = 15) -> Skill:
    return PickSkill[BxN_Obs, BxM_Action, None](
        reach_policy=make_ik_ee_pose_policy(env), gripper_policy=None, lift_height=lift_height, length=skill_length
    )


def make_place_skill(env: GymVectorInterface, lift_height: float = 0.23, skill_length: int = 15) -> Skill:
    return PlaceSkill[BxN_Obs, BxM_Action, None](
        reach_policy=make_ik_ee_pose_policy(env), gripper_policy=None, lift_height=lift_height, length=skill_length
    )


def make_push_skill(env: GymVectorInterface, skill_length: int = 15) -> Skill:
    return PushSkill[BxN_Obs, BxM_Action, None](
        reach_policy=make_ik_ee_pose_policy(env), gripper_policy=None, length=skill_length
    )


def make_grasp_xyz_skill(env: GymVectorInterface, skill_length: int = 15) -> Skill:
    return GraspXYZSkill[BxN_Obs, BxM_Action, None](
        reach_policy=make_ik_ee_pose_policy(env), gripper_policy=None, length=skill_length
    )


def make_osc_reach_xyz_skill(env: GymVectorInterface, skill_length: int = 15) -> Skill:
    return ReachXYZSkill[BxN_Obs, BxM_Action, None](
        name="reach_xyz_skill_osc", policy=make_osc_ee_pose_policy(env), length=skill_length, clip=True
    )


def make_osc_orient_rpy_skill(env: GymVectorInterface, skill_length: int = 15) -> Skill:  # TODO Change
    return OrientRPYSkill[BxN_Obs, BxM_Action, None](
        name="orient_rpy_skill_osc", policy=make_osc_ee_pose_policy(env), length=skill_length
    )


SKILL_LIB = {
    "push": make_push_skill,
    "place": make_place_skill,
    "pick": make_pick_skill,
    "grasp_xyz": make_grasp_xyz_skill,
    "orient_y": make_orient_y_skill,
    "orient_rpy": make_orient_rpy_skill,
    "orient_rpy_osc": make_osc_orient_rpy_skill,
    "reach_xyz": make_reach_xyz_skill,
    "reach_xyz_osc": make_osc_reach_xyz_skill,
    "reach_xyzrpy": make_reach_xyzrpy_skill,
    "rel_reach_xyzrpy": make_rel_reach_xyzrpy_skill,
    "gripper_oc": make_gripper_oc_skill,
    "gripper_c": make_gripper_c_skill,
    "gripper_o": make_gripper_o_skill,
    "joint_pos": make_joint_pos_skill,
}
