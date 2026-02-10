"""A pick skill for picking an object up at a location and lifting to a desired height."""

from enum import IntEnum
from typing import Generic

import torch
from jaxtyping import Int

from skillet.core.math import quat_error_magnitude, quat_from_euler_xyz
from skillet.core.policy import BatchedPPolicy
from skillet.core.skill import (
    BatchedSkill,
    SkillStatusCodes,
    TBAction,
    TBSkillObs,
    TBSkillParams,
)
from skillet.core.spaces import ArrayLike


class PickStatusCodes(IntEnum):
    """The codes for the status of a skill."""

    OPEN = 0
    """The skill is opening the gripper"""
    REACH = 1
    """The skill is reaching the hovering position."""
    ORIENT = 2
    """The skill is orienting above the object."""
    LOWER = 3
    """The skill is lowering to the object."""
    GRASP = 4
    """The skill is grasping the object."""
    LIFT = 5
    """The skill lifting the object."""
    DONE = 6
    """The skill has lifted the ojbect."""


class PickSkill(BatchedSkill[TBSkillObs, TBAction, TBSkillParams], Generic[TBSkillObs, TBAction, TBSkillParams]):
    """A pick skill for picking an object up at a location and lifting to a desired height."""

    def __init__(
        self,
        name: str,
        reach_policy: BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams],
        orient_policy: BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams],
        grasp_policy: BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams],
        length: int,
    ) -> None:
        """Initialize the pick skill.

        Args:
            name: The name of the skill.
            reach_policy: The policy for reaching.
            orient_policy: The policy for orienting.
            grasp_policy: The policy for grasping.
            length: The number of steps to execute the skill for.

        """
        self._name = name
        self._reach_policy = reach_policy
        self._orient_policy = orient_policy
        self._grasp_policy = grasp_policy
        self._length = length
        self._status = None
        self._pick_status = None
        self._params = None

    @property
    def name(self) -> str:  # noqa: D102
        return self._name

    @property
    def policy(self) -> BatchedPPolicy[TBSkillObs, TBAction, TBSkillParams]:
        """The policy for the skill."""
        return self._reach_policy

    @property
    def status(self) -> Int[ArrayLike, "b"]:  # noqa: F821
        """The status of the skills."""
        if self._status is None:
            raise ValueError("The status is not initialized. Must call initiate() before using this property.")
        return self._status

    def initiate(self, obs: TBSkillObs, params: TBSkillParams) -> None:  # noqa: D102
        self.n_envs = self.obs_spec.n_envs_from(obs)
        self._status = self.policy.obs_spec.with_n_envs(self.n_envs).zeros(shape=(self.n_envs,), dtype=int)
        self._pick_status = self.policy.obs_spec.with_n_envs(self.n_envs).zeros(shape=(self.n_envs,), dtype=int)
        self._status[:] = SkillStatusCodes.RUNNING
        self._pick_status[:] = PickStatusCodes.OPEN
        self._reach_policy.reset(obs, params[:, 0:3])
        self._orient_policy.reset(obs, params[:, 3:6])
        self._grasp_policy.reset(obs, params)
        self._params = params
        self._n_steps = 0

    def get_action(self, obs: TBSkillObs) -> TBAction:  # noqa: D102
        print(f"[INFO][PICK STATUS]: {self._pick_status}")

        zeros = torch.zeros((self.n_envs, 1), device=self._params.device)
        ones = torch.ones((self.n_envs, 1), device=self._params.device)

        prev_pick_status = self._pick_status.clone()

        gripper_lim = obs["gripper_lim"]
        gripper_pos = obs["tcp_pose_b"][:, -1].unsqueeze(1)
        goal_open_gripper_pos = (zeros - gripper_lim[:, 0]) / (gripper_lim[:, 1] - gripper_lim[:, 0])
        goal_close_gripper_pos = (ones - gripper_lim[:, 0]) / (gripper_lim[:, 1] - gripper_lim[:, 0])

        tcp_rpy = obs["tcp_pose_b"][:, 3:6]
        tcp_quat = quat_from_euler_xyz(tcp_rpy[:, 0], tcp_rpy[:, 1], tcp_rpy[:, 2])
        goal_tcp_quat = quat_from_euler_xyz(self._params[:, 3], self._params[:, 4], self._params[:, 5])

        open_action = self._grasp_policy.get_action(obs, zeros)[self._pick_status == PickStatusCodes.OPEN]

        reach_action = self._reach_policy.get_action(obs, self._params[:, 0:3])[
            self._pick_status == PickStatusCodes.REACH
        ]

        orient_action = self._orient_policy.get_action(obs, self._params[:, 3:6])[
            self._pick_status == PickStatusCodes.ORIENT
        ]

        lower_action = self._reach_policy.get_action(obs, torch.cat((self._params[:, 0:2], zeros + 0.02), dim=1))[
            self._pick_status == PickStatusCodes.LOWER
        ]

        grasp_action = self._grasp_policy.get_action(obs, ones)[self._pick_status == PickStatusCodes.GRASP]

        lift_action = self._reach_policy.get_action(obs, self._params[:, 0:3])[
            self._pick_status == PickStatusCodes.LIFT
        ]

        action = torch.zeros((self.n_envs, open_action.shape[1]), device=self._params.device)
        action[self._pick_status == PickStatusCodes.OPEN] = open_action
        action[self._pick_status == PickStatusCodes.REACH] = reach_action
        action[self._pick_status == PickStatusCodes.ORIENT] = orient_action
        action[self._pick_status == PickStatusCodes.LOWER] = lower_action
        action[self._pick_status == PickStatusCodes.GRASP] = grasp_action
        action[self._pick_status == PickStatusCodes.LIFT] = lift_action

        # Transition from opening to reaching
        self._pick_status = torch.where(
            self._pick_status
            == PickStatusCodes.OPEN & (torch.linalg.vector_norm(gripper_pos - goal_open_gripper_pos, dim=1) < 0.02),
            PickStatusCodes.REACH,
            self._pick_status,
        )

        # Transition from reaching to orienting
        self._pick_status = torch.where(
            self._pick_status
            == PickStatusCodes.REACH
            & (torch.linalg.vector_norm(obs["tcp_pose_b"][:, 0:3] - self._params[:, 0:3], dim=1) < 0.05),
            PickStatusCodes.ORIENT,
            self._pick_status,
        )

        # Transition from orienting to lowering
        self._pick_status = torch.where(
            self._pick_status == PickStatusCodes.ORIENT & (quat_error_magnitude(tcp_quat, goal_tcp_quat) < 0.03),
            PickStatusCodes.LOWER,
            self._pick_status,
        )

        # Transition from lowering to grasping
        self._pick_status = torch.where(
            self._pick_status
            == PickStatusCodes.LOWER
            & (
                torch.linalg.vector_norm(
                    obs["tcp_pose_b"][:, 0:3] - torch.cat((self._params[:, 0:2], zeros + 0.02), dim=1), dim=1
                )
                < 0.05
            ),
            PickStatusCodes.GRASP,
            self._pick_status,
        )

        # Transition from grasping to lifting
        self._pick_status = torch.where(
            self._pick_status
            == PickStatusCodes.GRASP & (torch.linalg.vector_norm(gripper_pos - goal_close_gripper_pos, dim=1) < 0.05),
            PickStatusCodes.LIFT,
            self._pick_status,
        )

        # Transition from lifting to done
        self._pick_status = torch.where(
            self._pick_status
            == PickStatusCodes.LIFT
            & (torch.linalg.vector_norm(obs["tcp_pose_b"][:, 0:3] - self._params[:, 0:3], dim=1) < 0.05),
            PickStatusCodes.DONE,
            self._pick_status,
        )

        # self._grasp_policy.reset(obs, self._params, prev_pick_status != self._pick_status)
        # self._reach_policy.reset(obs, self._params, prev_pick_status != self._pick_status)
        if ((prev_pick_status != self._pick_status) & (self._pick_status == PickStatusCodes.ORIENT)).any():
            self._orient_policy.reset(obs, self._params[:, 3:6], env_ids=prev_pick_status != self._pick_status)
        if ((prev_pick_status != self._pick_status) & (self._pick_status == PickStatusCodes.LOWER)).any():
            self._reach_policy.reset(
                obs,
                torch.cat((self._params[:, 0:2], zeros + 0.02), dim=1),
                env_ids=prev_pick_status != self._pick_status,
            )
        if ((prev_pick_status != self._pick_status) & (self._pick_status == PickStatusCodes.LIFT)).any():
            self._reach_policy.reset(
                obs,
                self._params[:, 0:3],
                env_ids=prev_pick_status != self._pick_status,
            )

        self._n_steps += 1

        self._status = torch.where(
            self._pick_status == PickStatusCodes.DONE,
            SkillStatusCodes.SUCCESS,
            self._status,
        )
        if self._n_steps >= self._length:
            self._status[:] = SkillStatusCodes.FAILED
        return action
