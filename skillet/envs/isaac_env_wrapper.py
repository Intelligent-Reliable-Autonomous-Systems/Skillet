"""isaac_env_wrapper.py.

A wrapper around IsaacLab Gym environments

Written by Will Solow and Jeff Jewett, 2026
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from jaxtyping import Bool, Float

from skillet.core.env import BatchedEnvironment
from skillet.core.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
from skillet.core.spaces import ActionSpec
from skillet.envs.utils import AsGymVectorEnv

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

from skillet.core import ObservationSpec

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


class IsaacEnvWrapper(
    BatchedEnvironment[TBatchedObsTorch, TBatchedActionTorch], Generic[TBatchedObsTorch, TBatchedActionTorch]
):
    """Wrapper for IsaacLab Environments.

    This assumes that the environment is either a DirectRLEnv or ManagerBasedRLEnv.
    """

    def __init__(self, env: "ManagerBasedRLEnv | DirectRLEnv") -> None:
        """Initialize the environment.

        Args:
            env: IsaacLab Gymnasium environment

        Returns:
            None

        """
        self._isaac_env = env
        self._env = env.unwrapped
        self._n_envs = env.unwrapped.cfg.scene.num_envs
        self.device = env.unwrapped.device
        vector_env = AsGymVectorEnv(env, num_envs=self._n_envs)
        super().__init__(vector_env)
        self._obs_spec_policy = ObservationSpec[Float[torch.Tensor, "b ..."]](
            space=vector_env.single_observation_space["policy"],
            name="policy",
            is_torch=True,
            is_batched=True,
            n_envs=-1,
            device=self.device,
        )
        self._obs_spec_state = ObservationSpec[Mapping[str, Float[torch.Tensor, "b ..."]]](
            space=vector_env.single_observation_space,
            name="state",
            is_torch=True,
            is_batched=True,
            n_envs=-1,
            device=self.device,
        )
        self._action_spec = ActionSpec[TBatchedActionTorch](
            space=vector_env.single_action_space,
            name="action",
            is_torch=True,
            is_batched=True,
            n_envs=-1,
            device=self.device,
        )

    @property
    def obs_spec(self):  # noqa: ANN201, D102
        return self._obs_spec_policy

    @property
    def action_spec(self):  # noqa: ANN201, D102
        return self._action_spec

    @property
    def n_envs(self) -> int:  # noqa: D102
        return self._n_envs

    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:  # noqa: D102
        return obs_spec.name in ["policy", "state"]

    def reset(self) -> tuple[TBatchedObsTorch, dict]:
        """Reset the environment.

        Args:
            None

        Returns:
            A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

        """
        obs_dict, info = self.env.reset()
        self.last_obs = obs_dict
        obs = obs_dict["policy"]
        if isinstance(obs, dict):
            obs = torch.cat(list(obs.values()), dim=1)

        return obs, info

    def get_observation(self, obs_spec=None):  # noqa: ANN001, ANN201, D102
        if self.last_obs is None:
            raise ValueError("No observation has been received yet. Call reset() first.")
        if obs_spec is None or obs_spec.name == "policy":
            return self.last_obs["policy"]
        if obs_spec.name == "state":
            return self.last_obs
        if obs_spec.name == "ik_ee":
            return {
                "joint_pos": self._get_joint_positions(),
                "jacobians": self._get_jacobians(),
                "ee_pose_b": self._get_ee_pose_b(),
            }
        if obs_spec.name == "ik_ee_callable":
            return {
                "joint_pos": self._get_joint_positions,
                "tcp_pose_xyz_b": self._get_tcp_pose_xyz_b,
                "jacobians": self._get_jacobians,
                "ee_pose_b": self._get_ee_pose_b,
            }
        raise ValueError(f"Observation spec {obs_spec} not supported by environment.")

    def get_state(self) -> TBatchedObsTorch:  # noqa: D102
        return self.get_observation(self._obs_spec_state)

    def step(
        self, action: TBatchedActionTorch
    ) -> tuple[
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
        obs_dict, reward, term, trunc, info = self.env.step(action)
        self.last_obs = obs_dict
        obs = obs_dict["policy"]
        if isinstance(obs, dict):
            obs = torch.cat(list(obs.values()), dim=1)

        return obs, reward, term, trunc, info

    """
    Helper functions
    """

    def _get_joint_positions(self, env_ids: torch.Tensor | None = None, joint_ids: list | None = None) -> torch.Tensor:
        """Return the joint positions.

        Args:
            env_ids: environment ids from which to get the joint ids
            joint_ids: the list of joint ids to retrieve
        Returns:
            torch tensor of jacobians of shape (n_envs, num_joints, 3)

        """
        if env_ids is None:
            env_ids = self._env._robot._ALL_INDICES
        if joint_ids is None:
            joint_ids = [0, 1, 2, 3, 4, 5, 6]
        return self._env._robot.data.joint_pos[:, joint_ids][env_ids].clone()

    def _get_joint_velocities(self, env_ids: torch.Tensor | None = None, joint_ids: list | None = None) -> torch.Tensor:
        """Return the joint velocities.

        Args:
            env_ids: environment ids from which to get the joint ids
            joint_ids: the list of joint ids to retrieve
        Returns:
            torch tensor of jacobians of shape (n_envs, num_joints, 3)

        """
        if env_ids is None:
            env_ids = self._env._robot._ALL_INDICES
        if joint_ids is None:
            joint_ids = [0, 1, 2, 3, 4, 5, 6]
        return self._env._robot.data.joint_vel[:, joint_ids][env_ids].clone()

    def _get_jacobians(
        self,
        env_ids: torch.Tensor | None = None,
        ee_link: str = "end_effector_link",
        base_link: str = "base_link",
        arm_joint_ids: list | None = None,
    ) -> torch.Tensor:
        """Return the jacobians.

        Args:
            env_ids: environment ids to compute jacobian
            ee_link: string for the name of the end effector link
            base_link: string for the name of the base link of the robot
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of jacobians of shape (n_envs, num_joints, 3)

        """
        if env_ids is None:
            env_ids = self._env._robot._ALL_INDICES
        if arm_joint_ids is None:
            arm_joint_ids = [0, 1, 2, 3, 4, 5, 6]
        ee_link_idx = self._env._robot.find_bodies(ee_link)[0][0]
        base_link_idx = self._env._robot.find_bodies(base_link)[0][0]
        robot_base_pose_w = self._env._robot.data.body_pose_w[env_ids, base_link_idx]
        base_rot_matrix = matrix_from_quat(quat_inv(robot_base_pose_w[:, 3:7]))
        jacobian = self._env._robot.root_physx_view.get_jacobians()[:, ee_link_idx, :, arm_joint_ids][env_ids]
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        return jacobian

    def _get_tcp_pose_xyz_b(
        self,
        env_ids: torch.Tensor | None = None,
        ee_link: str = "gripper_base_link",
        gripper_joint: str = "finger_joint",
        tcp_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get the TCP pose of the robot in the robot base frame.

        Args:
            env_ids: environment ids to tcp pose in XYZ
            ee_link: string for the name of the end effector link
            gripper_joint: string for the name of the gripper joint
            tcp_offset: The offset of the tcp frame from the end effector

        Returns:
            Tensor in shape (N,7) with 7 in (X,Y,Z,R,P,Y,Gripper) with 0 being open, 1 being closed
            for the gripper

        """
        if env_ids is None:
            env_ids = self._env._robot._ALL_INDICES
        if tcp_offset is None:
            tcp_offset = (
                torch.as_tensor([0.120, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=self.device)
                .unsqueeze(0)
                .repeat(env_ids.shape[0], 1)
            )

        gripper_joint_idx = self._env._robot.find_joints(gripper_joint)[0][0]

        ee_pos_w = self._env._robot.data.body_pos_w[env_ids, ee_link]
        ee_quat_w = self._env._robot.data.body_quat_w[env_ids, ee_link]
        root_pos_w = self._env._robot.data.root_pos_w[env_ids]
        root_quat_w = self._env._robot.data.root_quat_w[env_ids]

        ee_pos_b = quat_apply_inverse(root_quat_w, ee_pos_w - root_pos_w)
        ee_quat_b = quat_mul(quat_inv(root_quat_w), ee_quat_w)

        tcp_pos_b = ee_pos_b + quat_apply(ee_quat_b, tcp_offset[env_ids, 0:3])
        tcp_quat_b = quat_mul(ee_quat_b, tcp_offset[env_ids, 3:7])

        r, p, y = euler_xyz_from_quat(tcp_quat_b)

        gripper_low = self.robot_dof_lower_limits[gripper_joint_idx]
        gripper_high = self.robot_dof_upper_limits[gripper_joint_idx]
        gripper_pos = (self._robot.data.joint_pos[env_ids, gripper_joint_idx] - gripper_low) / (
            gripper_high - gripper_low
        )

        return torch.concatenate(
            (
                tcp_pos_b,
                r.unsqueeze(1),
                p.unsqueeze(1),
                y.unsqueeze(1),
                gripper_pos.unsqueeze(1),
            ),
            dim=1,
        )

    def _get_ee_pose_b(
        self,
        env_ids: torch.Tensor | None = None,
        ee_link: str = "gripper_base_link",
        base_link: str = "base_link",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the end effector pose of the robot in the robot's base frame.

        Args:
            env_ids: environment ids to compute jacobian
            ee_link: string for the name of the end effector link
            base_link: string for the name of the base link of the robot
            arm_joint_ids: the list of joint ids that correspond to the arm

        Returns:
            The robot EE position in shape (N, 3) relative to the base of the robot
            The robot EE orientation in shape (N, 4) relative to the base of the robot

        """
        if env_ids is None:
            env_ids = self._env._robot._ALL_INDICES
        ee_link_idx = self._env._robot.find_bodies(ee_link)[0][0]
        base_link_idx = self._env._robot.find_bodies(base_link)[0][0]

        robot_ee_pose_w = self._env._robot.data.body_pose_w[:, ee_link_idx]
        robot_base_pose_w = self._env._robot.data.body_pose_w[:, base_link_idx]

        robot_ee_pos_b, robot_ee_quat_b = subtract_frame_transforms(
            robot_base_pose_w[:, :3],
            robot_base_pose_w[:, 3:7],
            robot_ee_pose_w[:, :3],
            robot_ee_pose_w[:, 3:7],
        )

        return torch.cat((robot_ee_pos_b, robot_ee_quat_b), dim=1)

    def _compute_ee_pose_b_from_xyz_b(
        self, pose_b: torch.Tensor, env_ids: torch.Tensor | None = None, tcp_offset: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the goal end effector pose from the goal TCP pose.

        Args:
            pose_b: The goal TCP pose in the shape (N,6) relative to the robot base frame
            env_ids: environment ids to compute ee pose in base frame
            tcp_offset: The offset of the tcp frame from the end effector


        Returns:
            The goal end effector pose in shape (N,7)

        """
        if env_ids is None:
            env_ids = self._env._robot._ALL_INDICES
        if tcp_offset is None:
            tcp_offset = (
                torch.as_tensor([0.120, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=self.device)
                .unsqueeze(0)
                .repeat(env_ids.shape[0], 1)
            )

        goal_tcp_quat_b = quat_from_euler_xyz(pose_b[:, 3], pose_b[:, 4], pose_b[:, 5])
        goal_tcp_pos_b = pose_b[:, 0:3]

        # invert offset
        q_te = quat_inv(tcp_offset[:, 3:7])
        p_te = -quat_apply(q_te, tcp_offset[env_ids, 0:3])

        # compose
        q_be = quat_mul(goal_tcp_quat_b, q_te)
        p_be = goal_tcp_pos_b + quat_apply(goal_tcp_quat_b, p_te)

        return torch.cat((p_be, q_be), dim=1)
