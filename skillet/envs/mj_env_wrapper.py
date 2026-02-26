"""mj_env_wrapper.py.

A wrapper around Mujoco Warp Gym environments

Written by Will Solow and Jeff Jewett, 2026
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import gymnasium as gym
import torch
from jaxtyping import Bool, Float
from mjlab.utils.spaces import Box as MjBox
from mjlab.utils.spaces import Dict as MjDict
from mjlab.utils.spaces import Space as MjSpace

from skillet.core.env import AsGymVectorEnv, BatchedEnvironment
from skillet.core.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
from skillet.core.spaces import ActionSpec

if TYPE_CHECKING:
    from skillet.envs.mujoco import ManagerBasedRLEnv

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


class MJEnvWrapper(
    BatchedEnvironment[TBatchedObsTorch, TBatchedActionTorch], Generic[TBatchedObsTorch, TBatchedActionTorch]
):
    """Wrapper for Mujoco Environments.

    This assumes that the environment is either a DirectRLEnv or ManagerBasedRLEnv.
    """

    def __init__(self, env: "ManagerBasedRLEnv") -> None:
        """Initialize the environment.

        Args:
            env: Mujoco Gymnasium environment

        Returns:
            None

        """
        self._mj_env = env
        self._env = env.unwrapped
        self.device = env.unwrapped.device
        self.max_episode_length = env.unwrapped.max_episode_length

        """if hasattr(self._env, "robot"):
            self.robot = self._env.robot
        elif hasattr(self._env, "_robot"):
            self.robot = self._env._robot
        elif hasattr(self._env, "scene"):
            if hasattr(self._env.scene, "_robot"):
                self.robot = self._env.scene._robot
            elif hasattr(self._env.scene, "robot"):
                self.robot = self._env.scene.robot
            elif hasattr(self._env.scene, "_articulations"):
                self.robot = self._env.scene._articulations["robot"]
            else:
                raise ValueError(
                    f"Environment `{self._env} `scene.robot` or `scene._robot`. Unable to parse robot Articulation."
                )
        else:
            raise ValueError(
                f"Environment `{self._env}` has no attribute `_robot` or `robot` or `scene.robot` or `scene._robot`. Unable to parse robot Articulation."
            )"""
        vector_env = AsGymVectorEnv(env, num_envs=self.num_envs)
        super().__init__(vector_env)
        self._obs_spec_policy = ObservationSpec[Float[torch.Tensor, "b ..."]](
            space=self._handle_mj_space(vector_env.single_observation_space.spaces["actor"]),
            name="policy",
            is_torch=True,
            is_batched=True,
            n_envs=-1,
            device=self.device,
        )
        self._obs_spec_state = ObservationSpec[Mapping[str, Float[torch.Tensor, "b ..."]]](
            space=self._handle_mj_space(vector_env.single_observation_space),
            name="state",
            is_torch=True,
            is_batched=True,
            n_envs=-1,
            device=self.device,
        )
        self._action_spec = ActionSpec[TBatchedActionTorch](
            space=self._handle_mj_space(vector_env.single_action_space),
            name="action",
            is_torch=True,
            is_batched=True,
            n_envs=-1,
            device=self.device,
        )

        """# Robot specific information
        self.joint_ids = env.unwrapped.cfg.joint_ids
        self.tcp_offset = env.unwrapped.cfg.tcp_offset
        self.ee_link_name = env.unwrapped.cfg.ee_link_name
        self.base_link_name = env.unwrapped.cfg.base_link_name
        self.gripper_joint_names = env.unwrapped.cfg.gripper_joint_names

        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)[
            self.joint_ids
        ]
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)[
            self.joint_ids
        ]
        self.robot_dof_lower_limits[self.robot_dof_lower_limits == -float("inf")] = -torch.pi
        self.robot_dof_upper_limits[self.robot_dof_upper_limits == float("inf")] = torch.pi

        self.tcp_offset = torch.as_tensor(self.tcp_offset, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)"""

    def _handle_mj_space(self, mj_space: MjSpace) -> gym.spaces.Space:
        if isinstance(mj_space, MjBox):
            return gym.spaces.Box(low=mj_space.low, high=mj_space.high, shape=mj_space.shape, dtype=mj_space.dtype)
        if isinstance(mj_space, MjDict):
            return self._handle_mj_dict_recursive(mj_space)
        raise ValueError(f"Unable to process space `{type(mj_space)}`")

    def _handle_mj_dict_recursive(self, d: MjDict) -> gym.spaces.Dict:
        """Handle Mujoco Dictionary to Gym Spaec recursively."""
        fields: dict[str, type[Any]] = {}
        for k, subspace in d.spaces.items():
            fields[k] = self._handle_mj_space(subspace)
        return gym.spaces.Dict(fields)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.unwrapped.episode_length_buf

    @property
    def obs_spec(self):  # noqa: ANN201, D102
        return self._obs_spec_policy

    @property
    def action_spec(self):  # noqa: ANN201, D102
        return self._action_spec

    @property
    def n_envs(self) -> int:  # noqa: D102
        return self._env.unwrapped.num_envs

    @property
    def num_envs(self) -> int:  # noqa: D102
        return self._env.unwrapped.num_envs

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

        return obs_dict, info

    def get_observation(self, obs_spec=None):  # noqa: ANN001, ANN201, D102
        if self.last_obs is None:
            raise ValueError("No observation has been received yet. Call reset() first.")
        if obs_spec is None:
            return self.last_obs
        if obs_spec.name == "policy":
            return self.last_obs["actor"]
        if obs_spec.name == "state":
            return self.last_obs
        """if obs_spec.name == "ik_ee":
            return TensorDict(
                {
                    "joint_pos": self._get_joint_positions(joint_ids=self.joint_ids),
                    "joint_vel": self._get_joint_velocities(joint_ids=self.joint_ids),
                    "tcp_offset": self.tcp_offset,
                    "jacobians": self._get_jacobians(
                        ee_link=self.ee_link_name, base_link=self.base_link_name, arm_joint_ids=self.joint_ids[:7]
                    ),
                    "ee_pose_b": self._get_ee_pose_b(ee_link=self.ee_link_name, base_link=self.base_link_name),
                    "tcp_pose_b": self._get_tcp_pose_b(ee_link=self.ee_link_name),
                    "gripper_lim": self._get_gripper_lims(gripper_joints=self.gripper_joint_names),
                    "gripper": self._get_gripper_state(gripper_joints=self.gripper_joint_names),
                    "joint_lims": self._get_joint_lims(),
                },
                batch_size=self.num_envs,
            )
        if obs_spec.name == "osc":
            return TensorDict(
                {
                    "joint_pos": self._get_joint_positions(joint_ids=self.joint_ids),
                    "joint_vel": self._get_joint_velocities(joint_ids=self.joint_ids),
                    "tcp_offset": self.tcp_offset,
                    "jacobians": self._get_jacobians(
                        ee_link=self.ee_link_name, base_link=self.base_link_name, arm_joint_ids=self.joint_ids[:7]
                    ),
                    "ee_pose_b": self._get_ee_pose_b(ee_link=self.ee_link_name, base_link=self.base_link_name),
                    "tcp_pose_b": self._get_tcp_pose_b(ee_link=self.ee_link_name),
                    "gripper_lim": self._get_gripper_lims(gripper_joints=self.gripper_joint_names),
                    "gripper": self._get_gripper_state(gripper_joints=self.gripper_joint_names),
                    "joint_lims": self._get_joint_lims(),
                    "mass_matrix": self._get_mass_matrices(arm_joint_ids=self.joint_ids[:7]),
                    "joint_gravity": self._get_joint_gravity(arm_joint_ids=self.joint_ids[:7]),
                    "ee_vel_b": self._get_ee_vel_b(ee_link=self.ee_link_name, base_link=self.base_link_name),
                    "joint_centers": self._get_joint_centers(arm_joint_ids=self.joint_ids[:7]),
                },
                batch_size=self.num_envs,
            )"""
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
            env_ids = self.robot._ALL_INDICES
        if joint_ids is None:
            joint_ids = self.joint_ids
        return self.robot.data.joint_pos[:, joint_ids][env_ids]

    def _get_joint_velocities(self, env_ids: torch.Tensor | None = None, joint_ids: list | None = None) -> torch.Tensor:
        """Return the joint velocities.

        Args:
            env_ids: environment ids from which to get the joint ids
            joint_ids: the list of joint ids to retrieve
        Returns:
            torch tensor of jacobians of shape (n_envs, num_joints, 3)

        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if joint_ids is None:
            joint_ids = self.joint_ids
        return self.robot.data.joint_vel[:, joint_ids][env_ids]

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
            env_ids = self.robot._ALL_INDICES
        if arm_joint_ids is None:
            arm_joint_ids = self.joint_ids[:-1]
        ee_link_idx = self.robot.find_bodies(ee_link)[0][0]
        ee_jacobi_idx = ee_link_idx - 1
        base_link_idx = self.robot.find_bodies(base_link)[0][0]
        robot_base_pose_w = self.robot.data.body_pose_w[env_ids, base_link_idx]
        base_rot_matrix = matrix_from_quat(quat_inv(robot_base_pose_w[:, 3:7]))
        jacobian = self.robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids][env_ids]
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        return jacobian

    def _get_tcp_pose_b(
        self,
        env_ids: torch.Tensor | None = None,
        ee_link: str = "robotiq_85_base_link",
    ) -> torch.Tensor:
        """Get the TCP pose of the robot in the robot base frame.

        Args:
            env_ids: environment ids to tcp pose in XYZ
            ee_link: string for the name of the end effector link
            gripper_joint: string for the name of the gripper joint

        Returns:
            Tensor in shape (N,7) with 7 in (X,Y,Quat) w

        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        ee_link_idx = self.robot.find_bodies(ee_link)[0][0]

        ee_pos_w = self.robot.data.body_pos_w[env_ids, ee_link_idx]
        ee_quat_w = self.robot.data.body_quat_w[env_ids, ee_link_idx]
        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_quat_w = self.robot.data.root_quat_w[env_ids]

        ee_pos_b = quat_apply_inverse(root_quat_w, ee_pos_w - root_pos_w)
        ee_quat_b = quat_mul(quat_inv(root_quat_w), ee_quat_w)

        tcp_pos_b = ee_pos_b + quat_apply(ee_quat_b, self.tcp_offset[env_ids, 0:3])
        tcp_quat_b = quat_mul(ee_quat_b, self.tcp_offset[env_ids, 3:7])

        return torch.concatenate(
            (tcp_pos_b, tcp_quat_b),
            dim=1,
        )

    def _get_gripper_state(
        self, env_ids: torch.Tensor | None = None, gripper_joints: list[str] = ["robotiq_85_left_knuckle_joint"]
    ) -> torch.Tensor:
        """Get the gripper state of the robot."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        gripper_joint_idxs = [self.robot.find_joints(j)[0][0] for j in gripper_joints]

        return self.robot.data.joint_pos[:, gripper_joint_idxs][env_ids]

    def _get_ee_pose_b(  # Passes
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
            env_ids = self.robot._ALL_INDICES
        ee_link_idx = self.robot.find_bodies(ee_link)[0][0]
        base_link_idx = self.robot.find_bodies(base_link)[0][0]

        robot_ee_pose_w = self.robot.data.body_pose_w[env_ids, ee_link_idx]
        robot_base_pose_w = self.robot.data.body_pose_w[env_ids, base_link_idx]

        robot_ee_pos_b, robot_ee_quat_b = subtract_frame_transforms(
            robot_base_pose_w[:, :3],
            robot_base_pose_w[:, 3:7],
            robot_ee_pose_w[:, :3],
            robot_ee_pose_w[:, 3:7],
        )

        return torch.cat((robot_ee_pos_b, robot_ee_quat_b), dim=1)

    def _get_gripper_lims(
        self, env_ids: torch.Tensor | None = None, gripper_joints: str = ["robotiq_85_left_knuckle_joint"]
    ) -> torch.Tensor:
        """Get the gripper limits (low and high).

        Args:
            env_ids: environment ids to tcp pose in XYZ
            gripper_joints: list of strings for the name of the gripper joint

        Returns:
            A tensor of shape (N, 2) for the gripper lower/upper limits.

        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        gripper_joint_idxs = [self.robot.find_joints(j)[0][0] for j in gripper_joints]
        gripper_low = self.robot_dof_lower_limits[gripper_joint_idxs]
        gripper_high = self.robot_dof_upper_limits[gripper_joint_idxs]

        return torch.cat(
            (gripper_low.unsqueeze(0).repeat(self.num_envs, 1), gripper_high.unsqueeze(0).repeat(self.num_envs, 1)),
            dim=1,
        )

    def _get_joint_lims(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get the joint limits (low and high).

        Args:
            env_ids: environment ids to tcp pose in XYZ

        Returns:
            A tensor of shape (N, 2) for the gripper lower/upper limits.

        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        return (
            torch.cat((self.robot_dof_lower_limits, self.robot_dof_upper_limits), dim=0)
            .unsqueeze(0)
            .repeat(env_ids.shape[0], 1, 1)
        )

    def _get_mass_matrices(
        self,
        env_ids: torch.Tensor | None = None,
        arm_joint_ids: list | None = None,
    ) -> torch.Tensor:
        """Return the mass matrices.

        Args:
            env_ids: environment ids to compute jacobian
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of mass matrices

        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if arm_joint_ids is None:
            arm_joint_ids = self.joint_ids[:7]

        return self.robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids][
            env_ids
        ]

    def _get_joint_gravity(
        self,
        env_ids: torch.Tensor | None = None,
        arm_joint_ids: list | None = None,
    ) -> torch.Tensor:
        """Return the joint gravity of the arm joints.

        Args:
            env_ids: environment ids to compute jacobian
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of mass matrices

        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if arm_joint_ids is None:
            arm_joint_ids = self.joint_ids[:7]

        return self.robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids][env_ids]

    def _get_ee_vel_b(
        self, env_ids: torch.Tensor = None, ee_link: str = "end_effector_link", base_link: str = "base_link"
    ) -> torch.Tensor:
        """Compute the velocity of the end effector relative to the robot base.

        Args:
            env_ids: environment ids to compute jacobian
            ee_link: string for the name of the end effector link
            base_link: string for the name of the base link of the robot

        """
        # Compute the current velocity of the end-effector
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        ee_link_idx = self.robot.find_bodies(ee_link)[0][0]
        base_link_idx = self.robot.find_bodies(base_link)[0][0]

        ee_vel_w = self.robot.data.body_vel_w[
            env_ids, ee_link_idx, :
        ]  # Extract end-effector velocity in the world frame
        root_vel_w = self.robot.data.body_vel_w[env_ids, base_link_idx, :]  # Extract root velocity in the world frame
        relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
        ee_lin_vel_b = quat_apply_inverse(
            self.robot.data.body_quat_w[env_ids, base_link_idx], relative_vel_w[:, 0:3]
        )  # From world to root frame
        ee_ang_vel_b = quat_apply_inverse(self.robot.data.body_quat_w[env_ids, base_link_idx], relative_vel_w[:, 3:6])
        return torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    def _get_joint_centers(self, env_ids: torch.Tensor = None, arm_joint_ids: torch.Tensor = None) -> torch.Tensor:
        """Return the joint centers of the arm.

        Args:
            env_ids: environment ids to compute jacobian
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of joint centers

        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if arm_joint_ids is None:
            arm_joint_ids = self.joint_ids[:7]
        return torch.nan_to_num(
            torch.mean(self.robot.data.soft_joint_pos_limits[:, arm_joint_ids, :][env_ids], dim=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
