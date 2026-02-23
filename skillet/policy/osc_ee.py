"""Define simple dummy policies for testing."""

from typing import Any, Generic

import torch

from skillet.controllers import OperationalSpaceController
from skillet.core.math import quat_apply, quat_inv, quat_mul, subtract_frame_transforms
from skillet.core.policy import BatchedPPolicy, TBAction, TBPolicyObs
from skillet.core.spaces import ActionSpec, ObservationSpec


class OSCEEPolicy(BatchedPPolicy[TBPolicyObs, torch.Tensor, TBAction], Generic[TBPolicyObs, TBAction]):
    """Base class for Operational Space Control End Effector Policy."""

    osc: OperationalSpaceController
    _params: torch.Tensor

    def __init__(self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction]) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.

        """
        self._obs_spec = obs_spec
        self._action_spec = action_spec

    @property
    def obs_spec(self) -> ObservationSpec[TBPolicyObs]:  # noqa: D102
        return self._obs_spec

    @property
    def action_spec(self) -> ActionSpec[TBAction]:  # noqa: D102
        return self._action_spec

    def get_action(self, obs: TBPolicyObs, params: Any = None) -> TBAction:
        """Get the next joint positions by computing differential inverse kinematics."""
        ee_pose_b = obs["ee_pose_b"]
        jacobians = obs["jacobians"]
        joint_pos = obs["joint_pos"][:, :7]  # Ignore gripper
        joint_vel = obs["joint_vel"][:, :7]  # Ignore gripper
        mass_matrix = obs["mass_matrix"][:, :7]  # Ignore gripper
        ee_vel_b = obs["ee_vel_b"]
        joint_gravity = obs["joint_gravity"][:, :7]  # Ignore gripper
        arm_joint_pos = self.osc.compute(
            jacobian_b=jacobians,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            mass_matrix=mass_matrix,
            gravity=joint_gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            nullspace_joint_pos_target=self.joint_centers,
        )
        return torch.cat(
            (arm_joint_pos, self.start_gripper_pos),
            dim=1,
        )

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        n_envs = self._obs_spec.n_envs_from(obs)
        self._params = params
        self.osc.reset(n_envs, env_ids=env_ids)
        self.tcp_offset = obs["tcp_offset"]
        gripper_lim = obs["gripper_lim"]
        gripper_dim = obs["gripper"].shape[-1]
        self.joint_centers = obs["joint_centers"][:, :7]  # Ignore gripper
        self.start_gripper_pos = (obs["gripper"] - gripper_lim[:, 0:gripper_dim]) / (
            gripper_lim[:, gripper_dim:] - gripper_lim[:, 0:gripper_dim]
        )

    def _compute_goal_ee_pose_b_from_goal_tcp_b(
        self, tcp_pose_b: torch.Tensor, tcp_offset: torch.Tensor
    ) -> torch.Tensor:
        """Compute the goal end effector pose (xyz, quat) from the goal TCP pose in XYZ Quat.

        Args:
            tcp_pose_b: The goal TCP pose in the shape (N,7) relative to the robot base frame
            tcp_offset: The offset of the tcp frame from the end effector


        Returns:
            The goal end effector pose in shape (N,7)

        """
        goal_tcp_pos_b = tcp_pose_b[:, 0:3]
        goal_tcp_quat_b = tcp_pose_b[:, 3:7]

        # invert offset
        q_te = quat_inv(tcp_offset[:, 3:7])
        p_te = -quat_apply(q_te, tcp_offset[:, 0:3])

        # compose
        q_be = quat_mul(goal_tcp_quat_b, q_te)
        p_be = goal_tcp_pos_b + quat_apply(goal_tcp_quat_b, p_te)

        return torch.cat((p_be, q_be), dim=1)

    def convert_to_task_frame(
        self, command: torch.tensor, ee_target_pose_b: torch.tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert the target commands to the task frame.

        Args:
            osc: OperationalSpaceController object.
            command: Command to be converted.
            ee_target_pose_b: Target pose in the body frame.

        Returns:
            command (torch.tensor): Target command in the task frame.
            task_frame_pose_b (torch.tensor): Target pose in the task frame.

        Raises:
            ValueError: Undefined target_type.

        """
        command = command.clone()
        task_frame_pose_b = ee_target_pose_b.clone()

        cmd_idx = 0
        for target_type in self.osc.target_types:
            if target_type == "pose_abs":
                command[:, :3], command[:, 3:7] = subtract_frame_transforms(
                    task_frame_pose_b[:, :3], task_frame_pose_b[:, 3:], command[:, :3], command[:, 3:7]
                )
                cmd_idx += 7
            elif target_type == "wrench_abs":
                # These are already defined in target frame for ee_goal_wrench_set_tilted_task (since it is
                # easier), so not transforming
                cmd_idx += 6
            else:
                raise ValueError("Undefined target_type within _convert_to_task_frame().")

        return command, task_frame_pose_b


class PoseAbsOSCEEPolicy(OSCEEPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):
    """A policy that produces pose ."""

    def __init__(self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction]) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.

        """
        super().__init__(obs_spec, action_spec)
        self.osc = OperationalSpaceController(
            device=self._obs_spec.device, impedance_mode="variable_kp", gravity_compensation=True
        )

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the PoseAbsolute IK EE Policy by setting the command of the DiffIK Controlller.

        Args:
            obs: dict of bound functions
            params: target pose XYZ + Quat (num_envs, 7)
            env_ids: environment ids to reset

        """
        super().reset(obs, params, env_ids=env_ids)
        goal_pose = self._compute_goal_ee_pose_b_from_goal_tcp_b(params, obs["tcp_offset"])
        wrench = torch.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self._obs_spec.device).unsqueeze(0)
        kp = torch.as_tensor([360.0, 360.0, 360.0, 360.0, 360.0, 360.0], device=self._obs_spec.device).unsqueeze(0)
        goal_task_command = torch.cat(
            (goal_pose, wrench.repeat(goal_pose.shape[0], 1), kp.repeat(goal_pose.shape[0], 1)), dim=-1
        )
        ee_pose_b = obs["ee_pose_b"]
        command, task_frame_pose_b = self.convert_to_task_frame(command=goal_task_command, ee_target_pose_b=goal_pose)
        self.osc.set_command(
            command=command, current_ee_pose_b=ee_pose_b, env_ids=env_ids, current_task_frame_pose_b=task_frame_pose_b
        )
