"""skillet_skill_wrapper.py.

A wrapper around IsaacLab Gym compatible with skills

Written by Will Solow and Jeff Jewett, 2026
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeVar

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Bool, Float

from skillet.core.policy import TBAction, TBPolicyObs
from skillet.core.skill import BatchedSkill, Skill
from skillet.core.spaces import SpaceSpecification
from skillet.envs.compatibility import GymVectorInterface
from skillet.envs.skillet_env import SkilletEnv
from skillet.envs.specs import BxM_Action, BxN_Obs
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
from skillet.skill.policy.ik_ee import (
    IkEePolicy,
    PosAbsIkEePolicy,
    PoseAbsIkEePolicy,
    PoseRelIkEePolicy,
    XYZRPYAbsIkEePolicy,
)
from skillet.skill.policy.joint_pos import GripperPolicy, JointPosPolicy
from skillet.skill.policy.osc_ee import PoseAbsOscEePolicy

if TYPE_CHECKING:
    from skillet.envs.compatibility import SkilletGymEnv

TBatchedObsTorch = TypeVar(
    "TBatchedObsTorch", bound=Float[torch.Tensor, "b ..."] | Mapping[str, Float[torch.Tensor, "b ..."]]
)
"""A generic type of the batched observation tensor returned by the environment.

Can be a batched observation tensor or a dictionary of batched observation tensors.

torch.Tensor[(b, ...), float] | Mapping[str, torch.Tensor[(b, ...), float]]"""
TBatchedActionTorch = TypeVar("TBatchedActionTorch", bound=Float[torch.Tensor, "b n"])
"""A generic type of the batched action tensor expected by the environment.

torch.Tensor[(b, n), float]
"""


class SkillEnvWrapper(SkilletEnv):
    """Wrapper for ROS2/Mujoco/IsaacLab Environments for Skill Control.

    This assumes that the environment is a SkilletGymEnv.
    """

    def __init__(self, env: "SkilletGymEnv") -> None:
        """Initialize the environment.

        Args:
            env: SkilletGymEnv Gymnasium environment

        Returns:
            None

        """
        super().__init__(env)
        if hasattr(env.unwrapped.cfg, "skills"):
            assert (
                env.unwrapped.cfg.skills is not None
            ), "`env.cfg.skills` must not be None. Configure to list of skills."
        else:
            raise ValueError(
                f"Cannot use `SkillEnvWrapper` when `{type(env.unwrapped.cfg)}` does not contain the `skills` attribute."
            )

        self.sc = SkillController(
            env.unwrapped.cfg.skills,
            num_envs=env.unwrapped.num_envs,
            env=self,
            device=env.unwrapped.device,
        )
        # Update action space based on skill controller
        self.unwrapped.single_action_space = gym.spaces.Box(float("-inf"), float("inf"), shape=(self.sc.action_dim,))
        self.unwrapped.action_space = gym.spaces.Box(
            float("-inf"),
            float("inf"),
            shape=(
                env.unwrapped.num_envs,
                self.sc.action_dim,
            ),
        )

    def step(self, action: TBatchedActionTorch) -> tuple[
        TBatchedObsTorch,
        Float[torch.Tensor, "b"],  # noqa: F821
        Bool[torch.Tensor, "b"],  # noqa: F821
        Bool[torch.Tensor, "b"],  # noqa: F821
        Mapping[str, torch.Tensor],
    ]:
        """Step through the environment.

        Args:
            action: The action tensor of shape (N, num_actions)

        Returns:
            A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

        """
        action = action.to(self.device)
        self.sc.reset(action)  # Reset skills and parse
        _rewards = torch.zeros((self.num_envs,), device=self.device)
        _skill_length = torch.zeros((self.num_envs,), device=self.device)
        _dones = self.sc.dones  # Will always start as false
        while not _dones.all():
            ll_action = self.sc.get_action(self.get_observation())

            obs_dict, reward, term, trunc, info = super().step(ll_action)

            _rewards[~_dones] += reward[~_dones]
            _skill_length += ~_dones
            _dones = self.sc.dones
        _rewards = _rewards / _skill_length  # Normalize rewards based on skills length
        rewards = self.sc.post_process_reward(reward)  # If we don't sum, should we still normalize by skill length?
        self.last_obs = obs_dict

        return obs_dict, rewards, term, trunc, info


class SkillController(BatchedSkill):
    """Class for contrilling skills in an RL environment."""

    def __init__(self, skills: list[str], num_envs: int, env: GymVectorInterface, device: str = "cuda") -> None:
        """Initialize the skill controller based on the list of skills."""
        self.skill_names = skills

        for sk_name in self.skill_names:
            assert sk_name in SKILL_LIB, f"{sk_name} not in SKILL LIB: {SKILL_LIB.keys()}"
        self.skills = [SKILL_LIB[sk](env) for sk in self.skill_names]

        self.device = device
        self.num_envs = num_envs
        self.env_ids = torch.arange(self.num_envs, device=self.device)

        def get_param_dim(param_spec: SpaceSpecification) -> int:
            if isinstance(param_spec.space, gym.spaces.Dict):
                raise TypeError("Cannot get param dimension for a dictionary space.")
            if param_spec.is_batched and param_spec.n_envs >= 0:
                return np.prod(param_spec.space.shape[1:])
            return np.prod(param_spec.space.shape)  # space is single-env

        self.sk_param_dim = int(np.max([get_param_dim(skill.params_spec) for skill in self.skills]))
        self.num_skills = len(self.skills)
        self.action_dim = self.num_skills + self.sk_param_dim
        self._env_action_dim = int(np.prod(env.single_action_space.shape))
        self._obs_func = env.get_observation
        self.num_calls = 0

        self._dones = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._statuses = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._rewards = torch.zeros((self.num_envs,), device=self.device)

    @property
    def dones(self) -> torch.Tensor:
        """Check if all skills are done."""
        for i, sk in enumerate(self.skills):
            sk_env_ids = self.env_ids[self._skills_idx == i]
            if sk_env_ids.shape[0] == 0:
                continue
            term_idx = sk.is_terminated(self._obs_func(sk.obs_spec)[sk_env_ids])
            self._dones[sk_env_ids] = term_idx
        return self._dones

    @property
    def statuses(self) -> torch.Tensor:
        """Check the status of each skill."""
        for i, sk in enumerate(self.skills):
            sk_env_ids = self.env_ids[self._skills_idx == i]
            if sk_env_ids.shape[0] == 0:
                continue
            stat_idx = sk.status(self._obs_func(sk.obs_spec)[sk_env_ids])
            self._statuses[sk_env_ids] = stat_idx
        return self._statuses

    @property
    def successes(self) -> torch.Tensor:
        """Check the success of each skill."""
        for i, sk in enumerate(self.skills):
            sk_env_ids = self.env_ids[self._skills_idx == i]
            if sk_env_ids.shape[0] == 0:
                continue
            success_idx = sk.is_success(self._obs_func(sk.obs_spec)[sk_env_ids])
            self._successes[sk_env_ids] = success_idx
        return self._successes

    @property
    def param_dim(self): ...
    @property
    def policy(self): ...
    @property
    def status(self): ...
    def reward(self): ...

    def reset(self, action: TBAction) -> None:
        """Reset the skill controller based on the action parameters.

        Args:
            action: A torch Tensor of shape (num_envs, num_skills+max_param_dim).

        """
        assert (
            action.shape[-1] == self.action_dim
        ), f"Action dimension {action.shape[-1]} does not match expected skill dimension {self.action_dim}"
        self._skills_idx = self.get_skill_from_action(action)
        self._skills_params = self.get_params_from_action(action)

        for i, sk in enumerate(self.skills):
            sk_env_ids = self.env_ids[self._skills_idx == i]
            if sk_env_ids.shape[0] == 0:
                continue
            sk.initiate(self._obs_func(sk.obs_spec)[sk_env_ids], self._skills_params[sk_env_ids])
        self._action = torch.zeros((self.num_envs, self._env_action_dim), device=self.device)
        self._dones = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._statuses = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._rewards = torch.zeros((self.num_envs,), device=self.device)
        self.num_calls = 0

    def get_action(self, obs: TBPolicyObs) -> TBAction:
        """Step through the skills, getting the next joint position for each.

        Args:
            obs: The current observation from the environment

        Returns:
            TBAction in shape (num_envs, num_joints)

        """
        action = torch.zeros((self.num_envs, self._env_action_dim), device=self.device)
        for i, sk in enumerate(self.skills):
            sk_env_ids = self.env_ids[self._skills_idx == i]

            if sk_env_ids.shape[0] == 0:
                continue
            action[sk_env_ids] = sk.get_action(self._obs_func(sk.obs_spec)[sk_env_ids])

        self._action = action
        self.num_calls += 1

        return self._action

    def get_skill_from_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Return a tensor of shape (num_envs,) denoting each skill to use."""
        return torch.argmax(actions[:, : self.num_skills], dim=1)

    def get_params_from_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Return a tensor of shape (num_envs, sk_param_dim) denoting each skill parameter."""
        return actions[:, -self.sk_param_dim :]

    def post_process_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Post process the reward according to skill success.

        Args:
            reward: Tensor of shape (num_envs,) for the reward accumulated during the skill.

        """
        success_pen_fac = 1.0
        sk_penalty_fac = 0.2
        sk_success = self.successes
        reward = torch.where(sk_success, reward, reward / success_pen_fac)

        sk_penalty = 1.0 - torch.where(sk_success, 1.0, self._compute_sk_reward())

        return reward - sk_penalty_fac * sk_penalty

    def _compute_sk_reward(self) -> torch.Tensor:
        for i, sk in enumerate(self.skills):
            sk_env_ids = self.env_ids[self._skills_idx == i]
            if sk_env_ids.shape[0] == 0:
                continue
            reward = sk.reward(self._obs_func(sk.obs_spec)[sk_env_ids])
            self._rewards[sk_env_ids] = reward
        return self._rewards


def make_osc_ee_pose_policy(env: GymVectorInterface) -> IkEePolicy:
    return PoseAbsOscEePolicy[BxN_Obs, BxM_Action](env.coerce_obs_spec("osc_ee"), env.action_spec)


def make_ik_ee_xyzrpy_policy(env: GymVectorInterface) -> IkEePolicy:
    return XYZRPYAbsIkEePolicy[BxM_Action](env.coerce_obs_spec("ik_ee"), env.action_spec)


def make_ik_ee_pose_policy(env: GymVectorInterface) -> IkEePolicy:
    return PoseAbsIkEePolicy[BxM_Action](env.coerce_obs_spec("ik_ee"), env.action_spec)


def make_ik_ee_pos_policy(env: GymVectorInterface) -> IkEePolicy:
    return PosAbsIkEePolicy[BxM_Action](env.coerce_obs_spec("ik_ee"), env.action_spec)


def make_rel_ik_ee_pose_policy(env: GymVectorInterface) -> IkEePolicy:
    return PoseRelIkEePolicy[BxM_Action](env.coerce_obs_spec("ik_ee"), env.action_spec)


def make_rel_ik_ee_pose_policy(env: GymVectorInterface) -> IkEePolicy:
    return PoseRelIkEePolicy[BxM_Action](env.coerce_obs_spec("ik_ee"), env.action_spec)


def make_gripper_policy(env: GymVectorInterface) -> GripperPolicy:
    raise NotImplementedError
    return GripperPolicy[BxN_Obs, BxM_Action](make_joint_obs_spec(env.device), env.action_spec)


def make_joint_pos_policy(env: GymVectorInterface) -> JointPosPolicy:
    raise NotImplementedError
    return JointPosPolicy[BxN_Obs, BxM_Action](make_joint_obs_spec(env.device), env.action_spec)


def make_reach_xyzrpy_skill(env: GymVectorInterface, skill_length: int = 15) -> Skill:
    return ReachXYZRPYSkill[BxN_Obs, BxM_Action, None](
        name="reach_xyzrpy_skill", policy=make_ik_ee_xyzrpy_policy(env), length=skill_length
    )


def make_rel_reach_xyzrpy_skill(env: GymVectorInterface, skill_length: int = 5) -> Skill:
    return ReachXYZRPYSkill[BxN_Obs, BxM_Action, None](
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


def make_pick_skill(env: GymVectorInterface, lift_height: float = 0.3, skill_length: int = 15) -> Skill:
    return PickSkill[BxN_Obs, BxM_Action, None](
        reach_policy=make_ik_ee_pose_policy(env), gripper_policy=None, lift_height=lift_height, length=skill_length
    )


def make_place_skill(env: GymVectorInterface, lift_height: float = 0.3, skill_length: int = 15) -> Skill:
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


def make_osc_orient_rpy_skill(env: GymVectorInterface, skill_length: int = 15) -> Skill:
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
