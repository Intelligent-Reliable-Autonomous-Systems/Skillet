"""isaac_env_wrapper.py.

A wrapper around IsaacLab Gym environments

Written by Will Solow and Jeff Jewett, 2026
"""

from collections.abc import Mapping
from typing import Any, Generic, TypeVar, overload

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Bool, Float
from tensordict import TensorDict

from skillet.core import ObservationSpec
from skillet.core.env import AsGymVectorEnv, BatchedEnvironment, TSpecObs
from skillet.core.math import (
    convert_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
from skillet.core.spaces import ActionSpec

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


class ROS2EnvWrapper(
    BatchedEnvironment[TBatchedObsTorch, TBatchedActionTorch], Generic[TBatchedObsTorch, TBatchedActionTorch]
):
    """Wrapper for ROS2 Environments.

    This assumes that the environment is a gym.Env and interfaces directly with ROS2.
    """

    def __init__(self, env: gym.Env) -> None:
        """Initialize the environment.

        Args:
            env: Gymnasium environment that interfaces with ROS2

        Returns:
            None

        """
        self._ros_env = env
        self._env = env.unwrapped
        self.device = env.unwrapped.device
        self.max_episode_length = env.unwrapped.max_episode_length

        vector_env = AsGymVectorEnv(env, num_envs=self.num_envs)
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
        self._obs_spec_rgbd = ObservationSpec[Mapping[str, Float[torch.Tensor, "b ..."]]](
            space=gym.spaces.Dict({
                "rgb": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "depth": gym.spaces.Box(low=0, high=1000, shape=(480, 640), dtype=np.uint16),
                "intrinsic_k": gym.spaces.Box(low=0.0, high=2000.0, shape=(3, 3), dtype=np.float32),
                "camera_pose": gym.spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32),
                "timestamp": gym.spaces.Box(low=0.0, high=1e10, shape=(), dtype=np.float64),
            }),
            name="rgb-d",
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

        # Robot specific information
        self.joint_ids = env.unwrapped.cfg.joint_ids
        self.tcp_offset = env.unwrapped.cfg.tcp_offset
        self.ee_link_name = env.unwrapped.cfg.ee_link_name
        self.base_link_name = env.unwrapped.cfg.base_link_name
        self.gripper_joint_name = env.unwrapped.cfg.gripper_joint_name

        self.tcp_offset = torch.as_tensor(self.tcp_offset, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self.last_obs = None

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return torch.tensor([self.env.unwrapped.episode_length_buf], device=self.device)

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.unwrapped.episode_length_buf = value.squeeze().item()

    @property
    def obs_spec(self):
        return self._obs_spec_policy

    @property
    def action_spec(self):
        return self._action_spec

    @property
    def n_envs(self) -> int:  # noqa: D102
        return self._env.unwrapped.num_envs

    @property
    def num_envs(self) -> int:  # noqa: D102
        return self._env.unwrapped.num_envs

    @property
    def robot_dof_lower_limits(self) -> torch.Tensor:
        """Process and return lower joint limits."""
        lower_limits = torch.as_tensor(self._env._robot_lower_joint_limits, device=self.device)[self.joint_ids]
        lower_limits[lower_limits == 0] = -2 * torch.pi
        return lower_limits

    @property
    def robot_dof_upper_limits(self) -> torch.Tensor:
        """Process and return upper joint limits"""
        upper_limits = torch.as_tensor(self._env._robot_upper_joint_limits, device=self.device)[self.joint_ids]
        upper_limits[upper_limits == 0] = -2 * torch.pi
        return upper_limits

    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:
        return obs_spec.name in ["policy", "state", "rgb-d"]

    def reset(self) -> tuple[TBatchedObsTorch, dict]:
        """Reset the environment.

        Args:
            None

        Returns:
            A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

        """
        obs_dict, info = self.env.reset()
        self.last_obs = obs_dict
        for k, v in obs_dict.items():
            obs_dict[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)

        return obs_dict, info

    @overload
    def get_observation(self) -> TBatchedObsTorch: ...
    @overload
    def get_observation(self, obs_spec: ObservationSpec[TSpecObs]) -> TSpecObs: ...
    def get_observation(self, obs_spec: ObservationSpec[TSpecObs] | None = None) -> Any:  # noqa: D102
        if self.last_obs is None:
            raise ValueError("No observation has been received yet. Call reset() first.")
        if obs_spec is None:
            return self.last_obs  # TODO convert to TensorDict
        if obs_spec.is_batched:
            obs_spec = obs_spec.with_n_envs(self.n_envs)
        if obs_spec.name == "policy":
            return torch.as_tensor(self.last_obs["policy"], device=self.device).unsqueeze(0)
        if obs_spec.name == "rgb-d":
            latest = self._env._get_latest_rgbd()
            return obs_spec.cast(latest)
        if obs_spec.name == "state":
            return self.last_obs
        if obs_spec.name == "ik_ee":
            return TensorDict(
                {
                    "joint_pos": self._get_joint_positions(joint_ids=self.joint_ids),
                    "joint_vel": self._get_joint_velocities(joint_ids=self.joint_ids),
                    "tcp_offset": self.tcp_offset,
                    "jacobians": self._get_jacobians(ee_link=self.ee_link_name, base_link=self.base_link_name),
                    "ee_pose_b": self._get_ee_pose_b(ee_link=self.ee_link_name, base_link=self.base_link_name),
                    "tcp_pose_b": self._get_tcp_pose_b(ee_link=self.ee_link_name),
                    "gripper_lim": self._get_gripper_lims(gripper_joint=self.gripper_joint_name),
                    "gripper": self._get_gripper_state(gripper_joint=self.gripper_joint_name),
                    "joint_lims": self._get_joint_lims(),
                },
                batch_size=self.num_envs,
            )
        raise ValueError(f"Observation spec {obs_spec} not supported by environment.")

    def get_state(self) -> TBatchedObsTorch:  # noqa: D102
        return self.get_observation(self._obs_spec_state)

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
        obs_dict, reward, term, trunc, info = self.env.step(action)
        self.last_obs = obs_dict
        for k, v in obs_dict.items():
            obs_dict[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)

        reward = torch.as_tensor(reward, device=self.device)
        term = torch.as_tensor([term], device=self.device)
        trunc = torch.as_tensor([trunc], device=self.device)

        return obs_dict, reward, term, trunc, info

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
            env_ids = torch.arange(self.n_envs, device=self.device)
        if joint_ids is None:
            joint_ids = self.joint_ids
        return (
            torch.as_tensor(self._env._current_joint_positions, device=self.device)
            .unsqueeze(0)[:, joint_ids][env_ids]
            .to(torch.float32)
        )

    def _get_joint_velocities(self, env_ids: torch.Tensor | None = None, joint_ids: list | None = None) -> torch.Tensor:
        """Return the joint velocities.

        Args:
            env_ids: environment ids from which to get the joint ids
            joint_ids: the list of joint ids to retrieve
        Returns:
            torch tensor of jacobians of shape (n_envs, num_joints, 3)

        """
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)
        if joint_ids is None:
            joint_ids = self.joint_ids

        return (
            torch.as_tensor(self._env._current_joint_velocities, device=self.device)
            .unsqueeze(0)[:, joint_ids][env_ids]
            .to(torch.float32)
        )

    def _get_jacobians(
        self,
        env_ids: torch.Tensor | None = None,
        ee_link: str = "robotiq_85_base_link",
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
            env_ids = torch.arange(self.n_envs, device=self.device)
        if arm_joint_ids is None:
            arm_joint_ids = self.joint_ids[:-1]

        ee_link_idx = self._env._find_link_idx(ee_link)
        base_link_idx = self._env._find_link_idx(base_link)

        robot_base_pose_w = torch.as_tensor(
            self._env._robot_body_pose_w, device=self.device, dtype=torch.float32
        ).unsqueeze(0)[env_ids, base_link_idx]

        # Have to convert quaternion from ROS format (x,y,z,w) to IsaacLab format (w,x,y,z)
        robot_base_pose_w[:, 3:7] = convert_quat(robot_base_pose_w[:, 3:7], to="wxyz")
        base_rot_matrix = matrix_from_quat(quat_inv(robot_base_pose_w[:, 3:7]))

        jacobian = torch.as_tensor(self._env._jacobians, device=self.device, dtype=torch.float32)  # (N, 6, n_joints)
        # .unsqueeze(0)[
        #     :, ee_link_idx, :, arm_joint_ids
        # ][env_ids]
        jacobian = jacobian.unsqueeze(0)[env_ids, ee_link_idx][:, :, arm_joint_ids]

        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        return jacobian.to(torch.float32)

    def _get_tcp_pose_b(
        self,
        env_ids: torch.Tensor | None = None,
        ee_link: str = "robotiq_85_base_link",
    ) -> torch.Tensor:
        """Get the TCP pose of the robot in the robot base frame.

        Args:
            env_ids: environment ids to tcp pose in XYZ
            ee_link: string for the name of the end effector link

        Returns:
            Tensor in shape (N,7) with 7 in (X,Y,Z,Quat)

        """
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)

        ee_link_idx = self._env._find_link_idx(ee_link)

        ee_pose_w = torch.as_tensor(
            self._env._robot_body_pose_w[ee_link_idx], device=self.device, dtype=torch.float32
        ).unsqueeze(0)[env_ids]
        root_pose_w = torch.as_tensor(self._env._robot_root_pose_w, device=self.device, dtype=torch.float32).unsqueeze(
            0
        )[env_ids]

        # Have to convert quaternion from ROS format (x,y,z,w) to IsaacLab format (w,x,y,z)
        ee_pose_w[:, 3:7] = convert_quat(ee_pose_w[:, 3:7], to="wxyz")
        root_pose_w[:, 3:7] = convert_quat(root_pose_w[:, 3:7], to="wxyz")

        ee_pos_b = quat_apply_inverse(root_pose_w[:, 3:7], ee_pose_w[:, 0:3] - root_pose_w[:, 0:3])
        ee_quat_b = quat_mul(quat_inv(root_pose_w[:, 3:7]), ee_pose_w[:, 3:7])

        tcp_pos_b = ee_pos_b + quat_apply(ee_quat_b, self.tcp_offset[env_ids, 0:3])
        tcp_quat_b = quat_mul(ee_quat_b, self.tcp_offset[env_ids, 3:7])

        return torch.concatenate((tcp_pos_b, tcp_quat_b), dim=1).to(torch.float32)

    def _get_gripper_state(
        self, env_ids: torch.Tensor | None = None, gripper_joint: str = "robotiq_85_left_knuckle_joint"
    ) -> torch.Tensor:
        """Get the gripper state of the robot."""
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)

        gripper_joint_idx = self._env._find_joint_idx(gripper_joint)
        gripper_pos = torch.as_tensor(self._env._joint_positions[gripper_joint_idx], device=self.device).unsqueeze(0)[
            env_ids
        ]
        return gripper_pos.unsqueeze(1).to(torch.float32)

    def _get_ee_pose_b(
        self,
        env_ids: torch.Tensor | None = None,
        ee_link: str = "robotiq_85_base_link",
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
            env_ids = torch.arange(self.n_envs, device=self.device)
        ee_link_idx = self._env._find_link_idx(ee_link)
        base_link_idx = self._env._find_link_idx(base_link)

        # Get the pose of the end effector and base in the world frame
        # (B, 7) with (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w)
        robot_ee_pose_w = torch.as_tensor(
            self._env._robot_body_pose_w[ee_link_idx], device=self.device, dtype=torch.float32
        ).unsqueeze(0)[env_ids]
        robot_base_pose_w = torch.as_tensor(
            self._env._robot_body_pose_w[base_link_idx], device=self.device, dtype=torch.float32
        ).unsqueeze(0)[env_ids]

        # Have to convert quaternion from ROS format (x,y,z,w) to IsaacLab format (w,x,y,z)
        robot_ee_pose_w[:, 3:7] = convert_quat(robot_ee_pose_w[:, 3:7], to="wxyz")
        robot_base_pose_w[:, 3:7] = convert_quat(robot_base_pose_w[:, 3:7], to="wxyz")

        # Compute the end effector pose in the robot base frame
        robot_ee_pos_b, robot_ee_quat_b = subtract_frame_transforms(
            robot_base_pose_w[:, :3],
            robot_base_pose_w[:, 3:7],
            robot_ee_pose_w[:, :3],
            robot_ee_pose_w[:, 3:7],
        )

        return torch.cat((robot_ee_pos_b, robot_ee_quat_b), dim=1).to(torch.float32)

    def _get_gripper_lims(
        self, env_ids: torch.Tensor | None = None, gripper_joint: str = "robotiq_85_left_knuckle_joint"
    ) -> torch.Tensor:
        """Get the gripper limits (low and high).

        Args:
            env_ids: environment ids to tcp pose in XYZ
            gripper_joint: string for the name of the gripper joint

        Returns:
            A tensor of shape (N, 2) for the gripper lower/upper limits.

        """
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)

        gripper_joint_idx = self._env._find_joint_idx(gripper_joint)

        # gripper_low = self.robot_dof_lower_limits[gripper_joint_idx]
        gripper_low = torch.tensor([0], device=self.device)
        gripper_high = self.robot_dof_upper_limits[gripper_joint_idx]
        return torch.cat(
            (
                gripper_low.unsqueeze(0).expand(env_ids.shape[0], 1),
                gripper_high.unsqueeze(0).expand(env_ids.shape[0], 1),
            ),
            dim=1,
        ).to(torch.float32)

    def _get_joint_lims(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get the joint limits (low and high).

        Args:
            env_ids: environment ids to tcp pose in XYZ

        Returns:
            A tensor of shape (N, 2) for the gripper lower/upper limits.

        """
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)

        return (
            torch.cat((self.robot_dof_lower_limits.unsqueeze(0), self.robot_dof_upper_limits.unsqueeze(0)), dim=0)
            .unsqueeze(0)
            .repeat(env_ids.shape[0], 1, 1)
            .to(torch.float32)
        )
