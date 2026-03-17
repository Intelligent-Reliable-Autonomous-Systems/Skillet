from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform, subtract_frame_transforms
from isaacsim.core.utils.torch.transformations import tf_combine

from skillet.envs.isaac import IsaacDirectRlEnv


class LiftCubeEnv(IsaacDirectRlEnv):
    """Base clase for lift cube environments."""

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

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions, device=self.device)

        self.cube_init_ranges = torch.tensor(self.cfg.cube_init_ranges, device=self.device)
        self.cube_goal_ranges = torch.tensor(self.cfg.cube_goal_ranges, device=self.device)

        # Cube position
        self.cube_xyz_b = torch.zeros((self.num_envs, 6), device=self.device)
        self.cube_pose_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.cube_pose_w = torch.zeros((self.num_envs, 7), device=self.device)

        self.cube_goal_xyz_b = torch.zeros((self.num_envs, 6), device=self.device)
        self.cube_goal_pose_w = torch.zeros((self.num_envs, 7), device=self.device)

        # create auxiliary variables for joint limits
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)[
            self.cfg.joint_ids
        ]
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)[
            self.cfg.joint_ids
        ]
        self.robot_effort_limits = self._robot.data.joint_effort_limits[0, :].to(device=self.device)[self.cfg.joint_ids]
        self.robot_dof_lower_limits[self.robot_dof_lower_limits == -float("inf")] = -torch.pi
        self.robot_dof_upper_limits[self.robot_dof_upper_limits == float("inf")] = torch.pi

        self.default_joint_pos = self._robot.data.default_joint_pos[:, self.cfg.joint_ids]

        self.robot_dof_targets = torch.zeros((self.num_envs, len(self.cfg.joint_ids)), device=self.device)

        self.ee_link_idx = self._robot.find_bodies(self.cfg.ee_link_name)[0][0]

        # Robot position
        self.tcp_offset = torch.as_tensor(self.cfg.tcp_offset, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.robot_tcp_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_ee_pose_w = torch.zeros((self.num_envs, 7), device=self.device)

        self.cube_goal_marker = VisualizationMarkers(cfg=self.cfg.cube_goal_pose_visualizer_cfg)
        self.cube_current_marker = VisualizationMarkers(cfg=self.cfg.cube_current_pose_visualizer_cfg)
        self.current_marker = VisualizationMarkers(cfg=self.cfg.current_pose_visualizer_cfg)

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.object_cfg)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-5.0, 5.0)
        targets = self.default_joint_pos + self.actions * self.cfg.action_scale
        self.robot_dof_targets = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        # Update markers

        self.cube_current_marker.visualize(self.cube_pose_w[:, 0:3], self.cube_pose_w[:, 3:7])
        self.current_marker.visualize(self.robot_tcp_pose_w[:, 0:3], self.robot_tcp_pose_w[:, 3:7])

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets, joint_ids=self.cfg.joint_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate if max length is reached or cube goes below minimum height"""
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated_pos = self.cube_pose_w[:, 2] < -0.05

        return terminated_pos, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        return compute_rewards(
            self.actions,
            self.prev_actions,
            self._robot.data.joint_pos[:, self.cfg.joint_ids],
            self._robot.data.joint_vel[:, self.cfg.joint_ids],
            self.robot_tcp_pose_w[:, 0:3],
            self.cube_pose_w[:, 0:3],
            self.cube_goal_pose_w[:, 0:3],
            self.cfg.min_height,
            self.cfg.cube_reach_reward_scale,
            self.cfg.cube_reach_reward_std,
            self.cfg.cube_lift_reward_scale,
            self.cfg.cube_goal_reward_scale,
            self.cfg.cube_goal_reward_std,
            self.cfg.cube_goal_fine_grained_scale,
            self.cfg.cube_goal_fine_grained_std,
            self.cfg.action_rate_reward_scale,
            self.cfg.joint_vel_reward_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = (
            self._robot.data.default_joint_pos[env_ids]
            + sample_uniform(
                -0.125,
                0.125,
                (len(env_ids), self._robot.num_joints),
                self.device,
            )
        )[:, self.cfg.joint_ids]
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)[env_ids]
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids, joint_ids=self.cfg.joint_ids)
        self._robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
        self._robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids, joint_ids=self.cfg.joint_ids)
        self._robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)

        # Reset the cube position
        self._reset_cube_pose(env_ids=env_ids)

        self._object.write_root_pose_to_sim(self.cube_pose_w[env_ids], env_ids)
        self._object.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self._robot.data.joint_pos[:, self.cfg.joint_ids]
                - self._robot.data.default_joint_pos[:, self.cfg.joint_ids],
                self._robot.data.joint_vel[:, self.cfg.joint_ids],
                self.cube_pose_b[:, 0:3],
                self.cube_goal_xyz_b[:, 0:3],
                self.prev_actions,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """Docstring for _compute_intermediate_values.

        :param self: Description
        :param env_ids: Description
        :type env_ids: torch.Tensor | None

        Compute vlaues in the world frame
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.cube_pose_w[env_ids] = self._object.data.root_pose_w[env_ids]

        self.robot_ee_pose_w[env_ids] = self._robot.data.body_pose_w[env_ids, self.ee_link_idx]

        self.prev_actions[env_ids] = torch.clone(self.actions[env_ids])

        # Object position in robot frame
        cube_pos_b, cube_quat_b = subtract_frame_transforms(
            self._robot.data.root_pos_w[env_ids],
            self._robot.data.root_quat_w[env_ids],
            self.cube_pose_w[:, 0:3][env_ids],
        )
        self.cube_pose_b[env_ids] = torch.cat((cube_pos_b, cube_quat_b), dim=-1)

        robot_tcp_quat_w, robot_tcp_pos_w = tf_combine(
            self.robot_ee_pose_w[env_ids][:, 3:7],
            self.robot_ee_pose_w[env_ids][:, 0:3],
            self.tcp_offset[env_ids][:, 3:7],
            self.tcp_offset[env_ids][:, 0:3],
        )

        self.robot_tcp_pose_w[env_ids] = torch.cat((robot_tcp_pos_w, robot_tcp_quat_w), dim=-1)

    def _reset_cube_pose(self, env_ids: torch.Tensor | None):
        """Reset the position of the cube"""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # Set cube position and rotation
        base_pos_w = self._robot.data.root_pos_w[env_ids]
        base_quat_w = self._robot.data.root_quat_w[env_ids]

        self.cube_xyz_b[env_ids] = self.cube_init_ranges[:, 0] + (
            self.cube_init_ranges[:, 1] - self.cube_init_ranges[:, 0]
        ) * torch.rand((len(env_ids), len(self.cube_init_ranges)), device=self.device)

        # Cube in base frame
        cube_quat_b = quat_from_euler_xyz(
            self.cube_pose_b[env_ids, 3], self.cube_pose_b[env_ids, 4], self.cube_pose_b[env_ids, 5]
        )
        self.cube_pose_b[env_ids] = torch.cat((self.cube_xyz_b[env_ids][:, 0:3], cube_quat_b), dim=-1)

        # Cube in world frame
        cube_quat_w, cube_pos_w = tf_combine(base_quat_w, base_pos_w, cube_quat_b, self.cube_xyz_b[env_ids][:, 0:3])
        self.cube_pose_w[env_ids] = torch.cat((cube_pos_w, cube_quat_w), dim=-1)

        # Set cube goal position and rotation in base frame
        self.cube_goal_xyz_b[env_ids] = self.cube_goal_ranges[:, 0] + (
            self.cube_goal_ranges[:, 1] - self.cube_goal_ranges[:, 0]
        ) * torch.rand((len(env_ids), len(self.cube_goal_ranges)), device=self.device)

        cube_goal_quat_b = quat_from_euler_xyz(
            self.cube_goal_xyz_b[env_ids, 3], self.cube_goal_xyz_b[env_ids, 4], self.cube_goal_xyz_b[env_ids, 5]
        )
        # Cube goal position in world frame
        cube_goal_quat_w, cube_goal_pos_w = tf_combine(
            base_quat_w, base_pos_w, cube_goal_quat_b, self.cube_goal_xyz_b[env_ids][:, 0:3]
        )
        self.cube_goal_pose_w[env_ids] = torch.cat((cube_goal_pos_w, cube_goal_quat_w), dim=-1)


@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    tcp_pos_w: torch.Tensor,
    cube_pos_w: torch.Tensor,
    cube_goal_pos_w: torch.Tensor,
    min_height: float,
    cube_reach_reward_scale: float,
    cube_reach_reward_std: float,
    cube_lift_reward_scale: float,
    cube_goal_reward_scale: float,
    cube_goal_reward_std: float,
    cube_goal_fine_grained_scale: float,
    cube_goal_fine_grained_std: float,
    action_rate_reward_scale: float,
    joint_vel_reward_scale: float,
):
    # Distance of the end-effector to the object
    cube_tcp_distance = torch.norm(cube_pos_w - tcp_pos_w, dim=1)
    cube_tcp_distance_reward = cube_reach_reward_scale * (1 - torch.tanh(cube_tcp_distance / cube_reach_reward_std))

    cube_lifted_reward = cube_lift_reward_scale * torch.where(cube_pos_w[:, 2] > min_height, 1.0, 0.0)

    cube_distance = torch.norm(cube_goal_pos_w - cube_pos_w, dim=1)

    cube_dist_reward = (
        cube_goal_reward_scale
        * (cube_pos_w[:, 2] > min_height)
        * (1 - torch.tanh(cube_distance / cube_goal_reward_std))
    )
    cube_dist_reward_fine_grained = (
        cube_goal_fine_grained_scale
        * (cube_pos_w[:, 2] > min_height)
        * (1 - torch.tanh(cube_distance / cube_goal_fine_grained_std))
    )
    action_rate_reward = (action_rate_reward_scale * torch.sum(torch.square(actions - prev_actions), dim=1)).clamp(
        -1, 1
    )
    joint_vel_reward = (joint_vel_reward_scale * torch.sum(torch.square(joint_vel), dim=1)).clamp(-1, 1)

    return (
        # gripper_reward
        cube_tcp_distance_reward
        + cube_lifted_reward
        + cube_dist_reward
        + cube_dist_reward_fine_grained
        + action_rate_reward
        + joint_vel_reward
    )
