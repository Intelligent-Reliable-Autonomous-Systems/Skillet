from __future__ import annotations

import torch

from skillet.core.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
    tf_combine,
)
from skillet.envs.mujoco import MjDirectRlEnv


class ReachXyzEnv(MjDirectRlEnv):
    """Reach environment assuming action is in end effector space (XYZ RPY + Gripper)."""

    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.mujoco.timestep * self.cfg.decimation

        # Goal poses and end effector positions
        self.ee_ranges = torch.tensor(self.cfg.ee_ranges, device=self.device)
        self.goal_ee_xyz_b = torch.zeros((self.num_envs, 6), device=self.device)
        self.goal_ee_pose_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.goal_ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_ee_quat_w = torch.zeros((self.num_envs, 4), device=self.device)

        self.robot_ee_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        self.robot_ee_pose_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_tcp_pose_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_ee_vel_b = torch.zeros((self.num_envs, 6), device=self.device)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._current_prev_actions = torch.zeros_like(self.actions, device=self.device)

        # Limits and targets
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)[
            self.cfg.joint_ids
        ]
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)[
            self.cfg.joint_ids
        ]
        # self.robot_effort_limits = self.robot.data.joint_effort_limits[0, :].to(device=self.device)[self.cfg.joint_ids]
        self.robot_dof_lower_limits[self.robot_dof_lower_limits == -float("inf")] = -torch.pi
        self.robot_dof_upper_limits[self.robot_dof_upper_limits == float("inf")] = torch.pi

        self.default_joint_pos = self.robot.data.default_joint_pos[:, self.cfg.joint_ids]
        self.robot_dof_targets = torch.zeros((self.num_envs, len(self.cfg.joint_ids)), device=self.device)

        self.ee_link_idx = self.robot.find_bodies(self.cfg.ee_link_name)[0][0]

        self.tcp_offset = torch.as_tensor(self.cfg.tcp_offset, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

    def _setup_scene(self):
        pass

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        xyz_rpy = self.actions[:, :6]
        r, p, y = euler_xyz_from_quat(self.robot_tcp_pose_b[:, 3:7])
        dx = xyz_rpy - torch.cat((self.robot_tcp_pose_b[:, 0:3], torch.stack((r, p, y), dim=-1)), dim=-1)
        J = self._jacobians[:, self.ee_link_idx, :]
        JT = J.permute(0, -1, -2)
        JJT = J @ JT
        dq = (JT @ torch.linalg.solve(JJT, dx.unsqueeze(-1))).squeeze()
        targets = (self._joint_positions + dq)[:, self.cfg.joint_ids]

        self.robot_dof_targets = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.robot_dof_targets[:, :-1], joint_ids=self.cfg.joint_ids[:-1])
        self.robot.set_tendon_len_target(self.robot_dof_targets[:, -1:], tendon_ids=self.cfg.joint_ids[-1:])

    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate if max length is reached."""
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros((self.num_envs), dtype=torch.bool), truncated

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Return the observations as a vector."""
        # obs = torch.cat((self.robot_tcp_pose_b, self.robot_ee_vel_b, self._prev_actions, self.goal_ee_xyz_b), dim=1)
        return {"policy": self.obs_manager.obs_vec().clamp(-5.0, 5.0)}
        # return {"policy": obs.clamp(-5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        return compute_rewards(
            self.actions,
            self._current_prev_actions,
            self.robot_ee_vel_b,
            self.goal_ee_pos_w,
            self.goal_ee_quat_w,
            self.robot_ee_pos_w,
            self.robot_ee_quat_w,
            self.cfg.ee_dist_reward_scale,
            self.cfg.ee_dist_reward_fine_grained_scale,
            self.cfg.ee_dist_reward_fine_grained_std,
            self.cfg.ee_orientation_reward_scale,
            self.cfg.action_rate_reward_scale,
            self.cfg.ee_vel_reward_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = (
            self.robot.data.default_joint_pos[env_ids]
            + sample_uniform(
                -0.125,
                0.125,
                (len(env_ids), self.robot.num_joints),
                self.device,
            )
        )[:, self.cfg.joint_ids]
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids.unsqueeze(-1), joint_ids=self.cfg.joint_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=self.cfg.joint_ids)

        # target state
        base_pos_w = self.robot.data.root_link_pos_w[env_ids]
        base_quat_w = self.robot.data.root_link_quat_w[env_ids]
        self.goal_ee_xyz_b[env_ids] = (
            self.ee_ranges[:, 0]
            + (self.ee_ranges[:, 1] - self.ee_ranges[:, 0])
            * torch.rand((self.num_envs, len(self.ee_ranges)), device=self.device)
        )[env_ids]
        goal_ee_pos_b = self.goal_ee_xyz_b[env_ids, :3]
        goal_ee_quat_b = quat_from_euler_xyz(
            self.goal_ee_xyz_b[env_ids, 3], self.goal_ee_xyz_b[env_ids, 4], self.goal_ee_xyz_b[env_ids, 5]
        )
        self.goal_ee_pose_b[env_ids] = torch.cat((goal_ee_pos_b, goal_ee_quat_b), dim=-1)
        # World frame
        self.goal_ee_quat_w[env_ids], self.goal_ee_pos_w[env_ids] = tf_combine(
            base_quat_w, base_pos_w, goal_ee_quat_b, goal_ee_pos_b
        )

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """Docstring for _compute_intermediate_values.

        :param self: Description
        :param env_ids: Description
        :type env_ids: torch.Tensor | None

        Compute vlaues in the world frame
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.robot_ee_quat_w[env_ids] = self.robot.data.body_link_quat_w[env_ids, self.ee_link_idx]
        self.robot_ee_pos_w[env_ids] = self.robot.data.body_link_pos_w[env_ids, self.ee_link_idx]

        self._current_prev_actions[env_ids] = torch.clone(self.actions[env_ids])

        # Compute end effector pose
        ee_link_idx = self.robot.find_bodies(self.cfg.ee_link_name)[0][0]
        base_link_idx = self.robot.find_bodies(self.cfg.base_link_name)[0][0]

        ee_pose_w = self.robot.data.body_link_pose_w[env_ids, ee_link_idx]
        base_pose_w = self.robot.data.body_link_pose_w[env_ids, base_link_idx]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            base_pose_w[:, :3],
            base_pose_w[:, 3:7],
            ee_pose_w[:, :3],
            ee_pose_w[:, 3:7],
        )
        self.robot_ee_pose_b[env_ids] = torch.cat((ee_pos_b, ee_quat_b), dim=1)

        # Compute TCP pose
        root_pose_w = self.robot.data.root_link_pose_w[env_ids]
        ee_pos_b = quat_apply_inverse(root_pose_w[:, 3:7], ee_pose_w[:, 0:3] - root_pose_w[:, 0:3])
        ee_quat_b = quat_mul(quat_inv(root_pose_w[:, 3:7]), ee_pose_w[:, 3:7])

        tcp_pos_b = ee_pos_b + quat_apply(ee_quat_b, self.tcp_offset[env_ids, 0:3])
        tcp_quat_b = quat_mul(ee_quat_b, self.tcp_offset[env_ids, 3:7])
        self.robot_tcp_pose_b[env_ids] = torch.concatenate(
            (tcp_pos_b, tcp_quat_b),
            dim=1,
        )

        ee_vel_w = self.robot.data.body_link_vel_w[env_ids, ee_link_idx, :]
        root_vel_w = self.robot.data.body_link_vel_w[env_ids, base_link_idx, :]
        relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
        ee_lin_vel_b = quat_apply_inverse(
            self.robot.data.body_link_pose_w[env_ids, base_link_idx][:, 3:7],
            relative_vel_w[:, 0:3],
        )  # From world to root frame
        ee_ang_vel_b = quat_apply_inverse(
            self.robot.data.body_link_pose_w[env_ids, base_link_idx][:, 3:7],
            relative_vel_w[:, 3:6],
        )
        self.robot_ee_vel_b[env_ids] = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)


@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    ee_vel: torch.Tensor,
    goal_ee_pos: torch.Tensor,
    goal_ee_quat: torch.Tensor,
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    ee_dist_reward_scale: float,
    ee_dist_reward_fine_grained_scale: float,
    ee_dist_reward_fine_grained_std: float,
    ee_orientation_reward_scale: float,
    action_rate_reward_scale: float,
    ee_vel_reward_scale: float,
) -> torch.Tensor:
    ee_distance = torch.norm(ee_pos - goal_ee_pos, dim=1)
    dist_reward = ee_dist_reward_scale * ee_distance
    dist_fine_grained_rew = ee_dist_reward_fine_grained_scale * (
        1 - torch.tanh(ee_distance / ee_dist_reward_fine_grained_std)
    )

    orientation_reward = ee_orientation_reward_scale * quat_error_magnitude(ee_quat, goal_ee_quat)

    action_rate_reward = action_rate_reward_scale * torch.sum(torch.square(actions - prev_actions), dim=1)
    joint_vel_reward = ee_vel_reward_scale * torch.sum(torch.square(ee_vel), dim=1)

    return dist_reward + dist_fine_grained_rew + orientation_reward + joint_vel_reward + action_rate_reward
