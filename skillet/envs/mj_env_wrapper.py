"""mj_env_wrapper.py.

A wrapper around Mujoco Warp Gym environments

Written by Will Solow and Jeff Jewett, 2026
"""

from collections.abc import Mapping
from typing import Any, cast

import gymnasium as gym
import mujoco_warp as mjwarp
import torch
import warp as wp
from jaxtyping import Bool, Float
from mjlab.utils.spaces import Box as MjBox
from mjlab.utils.spaces import Dict as MjDict
from mjlab.utils.spaces import Space as MjSpace
from typing_extensions import override

from skillet.core import ObservationSpec
from skillet.core.env import BatchedEnvironment
from skillet.core.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
from skillet.core.spaces import ActionSpec
from skillet.envs.compatibility import AsGymVectorEnv, DirectRlInterface, ManagerBasedRlInterface
from skillet.envs.specs import IK_EE_SPEC_BATCHED, OSC_SPEC_BATCHED, RGBD_SPEC_BATCHED, BxM_Action, BxN_Obs


class MjEnvWrapper(
    BatchedEnvironment[BxN_Obs, BxM_Action],
    gym.vector.VectorWrapper,
):
    """Wrapper for Mujoco Environments.

    This assumes that the environment is either a DirectRlEnv or ManagerBasedRlEnv.
    """

    def __init__(self, env: DirectRlInterface | ManagerBasedRlInterface) -> None:
        """Initialize the environment.

        Args:
            env: Mujoco Gymnasium environment

        Returns:
            None

        """
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        self._env = cast("DirectRlInterface | ManagerBasedRlInterface", env)
        vector_env = AsGymVectorEnv(env, num_envs=self.num_envs)
        super().__init__(vector_env)
        self._device = self._env.device

        #  Convert Mj observation spaces to Gym observation spaces
        vector_env.single_observation_space = self._convert_mj_space(vector_env.single_observation_space)
        vector_env.observation_space = self._convert_mj_space(vector_env.observation_space)
        vector_env.action_space = self._convert_mj_space(vector_env.action_space)
        vector_env.single_action_space = self._convert_mj_space(vector_env.single_action_space)

        if hasattr(self._env, "scene"):
            if hasattr(self._env.scene, "entities"):
                if "robot" in self._env.scene.entities:
                    self.robot = self._env.scene.entities["robot"]
        else:
            raise ValueError(f"Environment `{self._env}` `scene['robot']`. Unable to parse robot Articulation.")

        # Robot specific information
        self._joint_ids = env.unwrapped.cfg.joint_ids
        self._tcp_offset = env.unwrapped.cfg.tcp_offset
        self._ee_link_name = env.unwrapped.cfg.ee_link_name
        self._base_link_name = env.unwrapped.cfg.base_link_name
        self._gripper_joint_names = env.unwrapped.cfg.gripper_joint_names

        self._robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self._device)[
            self._joint_ids
        ]
        self._robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self._device)[
            self._joint_ids
        ]
        self._robot_dof_lower_limits[self._robot_dof_lower_limits == -float("inf")] = -torch.pi
        self._robot_dof_upper_limits[self._robot_dof_upper_limits == float("inf")] = torch.pi

        self._tcp_offset = torch.as_tensor(self._tcp_offset, device=self._device).unsqueeze(0).repeat(self.num_envs, 1)

        with wp.ScopedDevice(self._env.sim.wp_device):
            self._jacp_wp = wp.zeros((self.num_envs, 3, self._env.sim.mj_model.nv), dtype=float)
            self._jacr_wp = wp.zeros((self.num_envs, 3, self._env.sim.mj_model.nv), dtype=float)
            self._point_wp = wp.zeros(self.num_envs, dtype=wp.vec3)
            self._body_wp = wp.zeros(self.num_envs, dtype=wp.int32)
            self._body_wp.fill_(self.robot.find_bodies(self._base_link_name)[0][0])

        self._jacp_torch = wp.to_torch(self._jacp_wp)
        self._jacr_torch = wp.to_torch(self._jacr_wp)
        self._point_torch = wp.to_torch(self._point_wp).view(self.num_envs, 3)

        # Define the obseravation and action specifications
        spec_args = {
            "is_torch": True,
            "is_batched": True,
            "n_envs": -1,
            "device": self.device,
        }
        if isinstance(env.observation_space, MjDict) and "policy" in env.observation_space.spaces:
            policy_space = env.single_observation_space.spaces["policy"]
        else:
            policy_space = env.single_observation_space
        self.obs_spec_policy = ObservationSpec[BxN_Obs](
            name="policy",
            space=self._convert_mj_space(policy_space),
        ).replace(**spec_args)
        """Specification of the vector observation passed to a low level policy"""
        self.obs_spec_state = ObservationSpec[Mapping[str, Float[torch.Tensor, "b ..."]]](
            name="state",
            space=self._convert_mj_space(env.single_observation_space),
        ).replace(**spec_args)
        """Specification of the raw dictionary environment state"""
        self.obs_spec_rgbd = RGBD_SPEC_BATCHED.bind(height=480, width=640).replace(device=self.device)
        """Specification of RGB-D observations and metadata. Bound to the height and width of the RGB-D camera."""
        self.obs_spec_ikee = IK_EE_SPEC_BATCHED.bind(
            n_joints=len(self._joint_ids),
            n_arm_joints=len(self._joint_ids[: -len(self._gripper_joint_names)]),
            n_gripper_joints=len(self._gripper_joint_names),
        ).replace(device=self.device)
        """Specification of IK-EE observations."""
        self.obs_spec_osc = OSC_SPEC_BATCHED.bind(
            n_joints=len(self._joint_ids),
            n_arm_joints=len(self._joint_ids[: -len(self._gripper_joint_names)]),
            n_gripper_joints=len(self._gripper_joint_names),
        ).replace(device=self.device)
        """Specification of OSC observations."""
        self._action_spec = ActionSpec[BxM_Action](
            name="action",
            space=self._convert_mj_space(env.single_action_space),
        ).replace(**spec_args)

    def _convert_mj_space(self, mj_space: MjSpace) -> gym.spaces.Space:
        if isinstance(mj_space, MjBox):
            return gym.spaces.Box(low=mj_space.low, high=mj_space.high, shape=mj_space.shape, dtype=mj_space.dtype)
        if isinstance(mj_space, MjDict):
            return self._convert_mj_dict_recursive(mj_space)
        raise ValueError(f"Unable to process space `{type(mj_space)}`")

    def _convert_mj_dict_recursive(self, d: MjDict) -> gym.spaces.Dict:
        """Handle Mujoco Dictionary to Gym Spaec recursively."""
        fields: dict[str, type[Any]] = {}
        for k, subspace in d.spaces.items():
            fields[k] = self._convert_mj_space(subspace)
        return gym.spaces.Dict(fields)

    # ==================== IsaacLab Interface ====================
    @property
    @override
    def num_envs(self) -> int:
        return self._env.unwrapped.num_envs

    @property
    @override
    def device(self) -> torch.device | str:
        return self._device

    @property
    @override
    def unwrapped(self) -> DirectRlInterface | ManagerBasedRlInterface:
        return self._env

    # ==================== Skillet Environment ====================

    @property
    @override
    def obs_spec(self):  # noqa: ANN201
        return self.obs_spec_policy

    @property
    @override
    def action_spec(self):  # noqa: ANN201
        return self._action_spec

    @override
    def supports_observation_spec(self, obs_spec: ObservationSpec) -> bool:
        return obs_spec.name in [
            self.obs_spec_policy.name,
            self.obs_spec_state.name,
            # self.obs_spec_rgbd.name,
            self.obs_spec_ikee.name,
            self.obs_spec_osc.name,
        ]

    @override
    def supports_action_spec(self, action_spec: ActionSpec) -> bool:
        return action_spec.name == self.action_spec.name

    @override
    def coerce_obs_spec(self, obs_spec: str | ObservationSpec[Any]) -> ObservationSpec[Any]:
        for spec in [
            self.obs_spec_policy,
            self.obs_spec_state,
            # self.obs_spec_rgbd,
            self.obs_spec_ikee,
            self.obs_spec_osc,
        ]:
            if spec.name == obs_spec:
                return spec
            if isinstance(obs_spec, str) and obs_spec == spec.name:
                return spec
        raise ValueError(f"Observation spec {obs_spec} not supported by environment.")

    @override
    def get_observation(self, obs_spec=None):  # noqa: ANN001, ANN201
        if self._last_obs is None:
            raise ValueError("No observation has been received yet. Call reset() first.")
        if obs_spec is None:
            obs_spec = self.obs_spec
        if obs_spec.is_batched:
            obs_spec = obs_spec.with_n_envs(self.num_envs)

        if obs_spec.name == self.obs_spec_policy.name:
            return self._last_obs["policy"]
        if obs_spec.name == self.obs_spec_state.name:
            return self._last_obs
        if obs_spec.name == self.obs_spec_ikee.name:
            return self.obs_spec_ikee.cast(
                {
                    "joint_pos": self._get_joint_positions(joint_ids=self._joint_ids),
                    "joint_vel": self._get_joint_velocities(joint_ids=self._joint_ids),
                    "tcp_offset": self._tcp_offset,
                    "jacobians": self._get_jacobians(
                        ee_link=self._ee_link_name, base_link=self._base_link_name, arm_joint_ids=self._joint_ids[:7]
                    ),
                    "ee_pose_b": self._get_ee_pose_b(ee_link=self._ee_link_name, base_link=self._base_link_name),
                    "tcp_pose_b": self._get_tcp_pose_b(ee_link=self._ee_link_name),
                    "gripper_lim": self._get_gripper_lims(gripper_joints=self._gripper_joint_names),
                    "gripper": self._get_gripper_state(gripper_joints=self._gripper_joint_names),
                    "joint_lims": self._get_joint_lims(),
                }
            )
        if obs_spec.name == self.obs_spec_osc.name:
            return self.obs_spec_osc.cast(
                {
                    "joint_pos": self._get_joint_positions(joint_ids=self._joint_ids),
                    "joint_vel": self._get_joint_velocities(joint_ids=self._joint_ids),
                    "tcp_offset": self._tcp_offset,
                    "jacobians": self._get_jacobians(
                        ee_link=self._ee_link_name, base_link=self._base_link_name, arm_joint_ids=self._joint_ids[:7]
                    ),
                    "ee_pose_b": self._get_ee_pose_b(ee_link=self._ee_link_name, base_link=self._base_link_name),
                    "tcp_pose_b": self._get_tcp_pose_b(ee_link=self._ee_link_name),
                    "gripper_lim": self._get_gripper_lims(gripper_joints=self._gripper_joint_names),
                    "gripper": self._get_gripper_state(gripper_joints=self._gripper_joint_names),
                    "joint_lims": self._get_joint_lims(),
                    "mass_matrix": self._get_mass_matrices(arm_joint_ids=self._joint_ids[:7]),
                    "joint_gravity": self._get_joint_gravity(arm_joint_ids=self._joint_ids[:7]),
                    "ee_vel_b": self._get_ee_vel_b(ee_link=self._ee_link_name, base_link=self._base_link_name),
                    "joint_centers": self._get_joint_centers(arm_joint_ids=self._joint_ids[:7]),
                }
            )
        raise ValueError(f"Observation spec {obs_spec} not supported by environment.")

    @override
    def get_state(self) -> Mapping[str, torch.Tensor]:
        return self.get_observation(self.obs_spec_state)

    # ==================== Public methods ====================

    @override
    def reset(self) -> tuple[BxN_Obs, dict]:
        """Reset the environment.

        Args:
            None

        Returns:
            A tuple containing the observation of observations tensor (N, obs_dim) and info dictionary

        """
        obs_dict, info = self.env.reset()
        self._last_obs = obs_dict

        return obs_dict, info

    @override
    def step(
        self, action: BxM_Action
    ) -> tuple[
        BxN_Obs,
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
        obs_dict, reward, term, trunc, info = self.env.step(action)
        self._last_obs = obs_dict

        return obs_dict, reward, term, trunc, info

    """
    Helper functions
    """

    def _get_joint_positions(
        self, env_ids: torch.Tensor | None = None, joint_ids: list | None = None
    ) -> Float[torch.Tensor, "b n_joints"]:
        """Return the joint positions (1 value per dof).

        Args:
            env_ids: environment ids from which to get the joint ids
            joint_ids: the list of joint ids to retrieve
        Returns:
            torch tensor of joint positions of shape (n_envs, num_joints)

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        if joint_ids is None:
            joint_ids = self._joint_ids
        return self.robot.data.joint_pos[:, joint_ids][env_ids]

    def _get_joint_velocities(
        self, env_ids: torch.Tensor | None = None, joint_ids: list | None = None
    ) -> Float[torch.Tensor, "b n_joints 3"]:
        """Return the joint velocities.

        Args:
            env_ids: environment ids from which to get the joint ids
            joint_ids: the list of joint ids to retrieve
        Returns:
            torch tensor of jacobians of shape (n_envs, num_joints, 3)

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        if joint_ids is None:
            joint_ids = self._joint_ids
        return self.robot.data.joint_vel[:, joint_ids][env_ids]

    def _get_jacobians(
        self,
        env_ids: torch.Tensor | None = None,
        ee_link: str = "end_effector_link",
        base_link: str = "base_link",
        arm_joint_ids: list | None = None,
    ) -> Float[torch.Tensor, "b 6 n_arm_joints"]:
        """Return the jacobians.

        For each arm joint, return linear and angular velocities in the robot base frame.

        Args:
            env_ids: environment ids to compute jacobian
            ee_link: string for the name of the end effector link
            base_link: string for the name of the base link of the robot
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of jacobians of shape (n_envs, 6, num_arm_joints)

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        if arm_joint_ids is None:
            arm_joint_ids = self._joint_ids[:-1]
        ee_link_idx = self.robot.find_bodies(ee_link)[0][0]
        ee_jacobi_idx = ee_link_idx - 1
        base_link_idx = self.robot.find_bodies(base_link)[0][0]
        robot_base_pose_w = self.robot.data.body_link_pose_w[env_ids, base_link_idx]
        base_rot_matrix = matrix_from_quat(quat_inv(robot_base_pose_w[:, 3:7]))

        jacobian = self._compute_jacobian()[:, ee_jacobi_idx, :, arm_joint_ids][env_ids]  # TODO TEST
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        return jacobian

    def _get_tcp_pose_b(
        self,
        env_ids: torch.Tensor | None = None,
        ee_link: str = "end_effector_link",
    ) -> Float[torch.Tensor, "b 7"]:
        """Get the TCP pose of the robot in the robot base frame.

        Args:
            env_ids: environment ids to tcp pose in XYZ
            ee_link: string for the name of the end effector link
            gripper_joint: string for the name of the gripper joint

        Returns:
            Tensor in shape (N,7) with 7 in (X,Y,Quat) w

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)

        ee_link_idx = self.robot.find_bodies(ee_link)[0][0]

        ee_pos_w = self.robot.data.body_link_pos_w[env_ids, ee_link_idx]
        ee_quat_w = self.robot.data.body_link_quat_w[env_ids, ee_link_idx]
        root_pos_w = self.robot.data.root_link_pos_w[env_ids]
        root_quat_w = self.robot.data.root_link_quat_w[env_ids]

        ee_pos_b = quat_apply_inverse(root_quat_w, ee_pos_w - root_pos_w)
        ee_quat_b = quat_mul(quat_inv(root_quat_w), ee_quat_w)

        tcp_pos_b = ee_pos_b + quat_apply(ee_quat_b, self._tcp_offset[env_ids, 0:3])
        tcp_quat_b = quat_mul(ee_quat_b, self._tcp_offset[env_ids, 3:7])

        return torch.concatenate(
            (tcp_pos_b, tcp_quat_b),
            dim=1,
        )

    def _get_gripper_state(
        self, env_ids: torch.Tensor | None = None, gripper_joints: list[str] = ["right_driver_joint"]
    ) -> Float[torch.Tensor, "b n_gripper_joints"]:
        """Get the gripper state of the robot."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        gripper_joint_idxs = [self.robot.find_joints(j)[0][0] for j in gripper_joints]

        return self.robot.data.joint_pos[:, gripper_joint_idxs][env_ids]

    def _get_ee_pose_b(  # Passes
        self,
        env_ids: torch.Tensor | None = None,
        ee_link: str = "end_effector_link",
        base_link: str = "base_link",
    ) -> Float[torch.Tensor, "b 7"]:
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
            env_ids = torch.arange(self.num_envs, device=self._device)
        ee_link_idx = self.robot.find_bodies(ee_link)[0][0]
        base_link_idx = self.robot.find_bodies(base_link)[0][0]

        robot_ee_pose_w = self.robot.data.body_link_pose_w[env_ids, ee_link_idx]
        robot_base_pose_w = self.robot.data.body_link_pose_w[env_ids, base_link_idx]

        robot_ee_pos_b, robot_ee_quat_b = subtract_frame_transforms(
            robot_base_pose_w[:, :3],
            robot_base_pose_w[:, 3:7],
            robot_ee_pose_w[:, :3],
            robot_ee_pose_w[:, 3:7],
        )

        return torch.cat((robot_ee_pos_b, robot_ee_quat_b), dim=1)

    def _get_gripper_lims(
        self, env_ids: torch.Tensor | None = None, gripper_joints: str = ["right_driver_joint"]
    ) -> Float[torch.Tensor, "b 2 n_gripper_joints"]:
        """Get the gripper limits (low and high).

        Args:
            env_ids: environment ids to tcp pose in XYZ
            gripper_joints: list of strings for the name of the gripper joint

        Returns:
            A tensor of shape (N, 2) for the gripper lower/upper limits.

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)

        gripper_joint_idxs = [self.robot.find_joints(j)[0][0] for j in gripper_joints]
        gripper_low = self._robot_dof_lower_limits[gripper_joint_idxs]
        gripper_high = self._robot_dof_upper_limits[gripper_joint_idxs]  # TODO this joint is out of range

        return torch.cat(
            (gripper_low.unsqueeze(0).repeat(self.num_envs, 1), gripper_high.unsqueeze(0).repeat(self.num_envs, 1)),
            dim=1,
        )

    def _get_joint_lims(self, env_ids: torch.Tensor | None = None) -> Float[torch.Tensor, "b 2 n_joints"]:
        """Get the joint limits (low and high).

        Args:
            env_ids: environment ids to tcp pose in XYZ

        Returns:
            A tensor of shape (num_envs, 2, num_joints) for the joint lower/upper limits.

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)

        return (
            torch.stack((self._robot_dof_lower_limits, self._robot_dof_upper_limits), dim=0)
            .unsqueeze(0)
            .repeat(env_ids.shape[0], 1, 1)
        )

    def _get_mass_matrices(
        self,
        env_ids: torch.Tensor | None = None,
        arm_joint_ids: list | None = None,
    ) -> Float[torch.Tensor, "b n_arm_joints n_arm_joints"]:
        """Return the mass matrices.

        Args:
            env_ids: environment ids to compute jacobian
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of mass matrices of shape (n_envs, num_arm_joints, num_arm_joints)

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        if arm_joint_ids is None:
            arm_joint_ids = self._joint_ids[:7]

        return self.robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids][
            env_ids
        ]

    def _get_joint_gravity(
        self,
        env_ids: torch.Tensor | None = None,
        arm_joint_ids: list | None = None,
    ) -> Float[torch.Tensor, "b n_arm_joints"]:
        """Return the joint gravity of the arm joints.

        Args:
            env_ids: environment ids to compute jacobian
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of joint gravity of shape (n_envs, num_arm_joints)

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        if arm_joint_ids is None:
            arm_joint_ids = self._joint_ids[:7]

        return self.robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids][
            env_ids
        ]  # TODO need alternative to this

    def _get_ee_vel_b(
        self, env_ids: torch.Tensor = None, ee_link: str = "end_effector_link", base_link: str = "base_link"
    ) -> Float[torch.Tensor, "b 6"]:
        """Compute the velocity of the end effector relative to the robot base.

        Args:
            env_ids: environment ids to compute jacobian
            ee_link: string for the name of the end effector link
            base_link: string for the name of the base link of the robot

        """
        # Compute the current velocity of the end-effector
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)

        ee_link_idx = self.robot.find_bodies(ee_link)[0][0]
        base_link_idx = self.robot.find_bodies(base_link)[0][0]

        ee_vel_w = self.robot.data.body_link_vel_w[
            env_ids, ee_link_idx, :
        ]  # Extract end-effector velocity in the world frame
        root_vel_w = self.robot.data.body_link_vel_w[
            env_ids, base_link_idx, :
        ]  # Extract root velocity in the world frame
        relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
        ee_lin_vel_b = quat_apply_inverse(
            self.robot.data.body_link_quat_w[env_ids, base_link_idx], relative_vel_w[:, 0:3]
        )  # From world to root frame
        ee_ang_vel_b = quat_apply_inverse(
            self.robot.data.body_link_quat_w[env_ids, base_link_idx], relative_vel_w[:, 3:6]
        )
        return torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    def _get_joint_centers(
        self, env_ids: torch.Tensor = None, arm_joint_ids: torch.Tensor = None
    ) -> Float[torch.Tensor, "b n_arm_joints"]:
        """Return the joint centers of the arm.

        Args:
            env_ids: environment ids to compute jacobian
            arm_joint_ids: the list of joint ids that correspond to the arm
        Returns:
            torch tensor of joint centers of shape (n_envs, n_arm_joints)

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        if arm_joint_ids is None:
            arm_joint_ids = self._joint_ids[:7]
        return torch.nan_to_num(
            torch.mean(self.robot.data.soft_joint_pos_limits[:, arm_joint_ids, :][env_ids], dim=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _compute_jacobian(self) -> None:
        """Compute the frame Jacobian."""
        _jacobians = torch.zeros(
            size=(self.num_envs, self._env.sim.mj_model.nv, 6, self.robot.num_joints), device=self.device
        )
        for i in range(self.robot.num_joints):
            self._body_wp.fill_(i)
            with wp.ScopedDevice(self._env.sim.wp_device):
                mjwarp.jac(
                    self._env.sim.wp_model,
                    self._env.sim.wp_data,
                    self._jacp_wp,
                    self._jacr_wp,
                    self._point_wp,
                    self._body_wp,
                )
            _jacobians[:, :, :, i] = torch.cat((self._jacp_torch, self._jacr_torch), dim=1).permute(0, 2, 1)
        return _jacobians
