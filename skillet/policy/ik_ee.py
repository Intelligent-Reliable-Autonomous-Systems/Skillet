"""Define simple dummy policies for testing."""

from typing import Any, Generic

import torch

from skillet.controllers import DifferentialIKController
from skillet.core.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul
from skillet.core.policy import BatchedPPolicy, TBAction, TBPolicyObs
from skillet.core.spaces import ActionSpec, ObservationSpec


class IKEEPolicy(BatchedPPolicy[TBPolicyObs, torch.Tensor, TBAction], Generic[TBPolicyObs, TBAction]):
    """Base class for Inverse Kinematics End Effector Policy."""

    diff_ik: DifferentialIKController
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
        joint_pos = obs["joint_pos"][:, :-1]  # Ignore gripper
        arm_joint_pos = self.diff_ik.compute(ee_pose_b[:, 0:3], ee_pose_b[:, 3:7], jacobians, joint_pos)
        return torch.cat(
            (arm_joint_pos, self.start_gripper_pos),
            dim=1,
        )

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        n_envs = self._obs_spec.n_envs_from(obs)
        self._params = params
        self.diff_ik.reset(n_envs, env_ids=env_ids)
        self.tcp_offset = obs["tcp_offset"]
        self.start_gripper_pos = obs["gripper"]

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


class PosAbsIKEEPolicy(IKEEPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):
    """A policy that produces ."""

    def __init__(self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction]) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.

        """
        super().__init__(obs_spec, action_spec)
        self.diff_ik = DifferentialIKController(
            device=self._obs_spec.device, command_type="position", use_relative_mode=False
        )

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the PoseAbsolute IK EE Policy by setting the command of the DiffIK Controlller.

        Args:
            obs: dict of bound functions
            params: target position XYZ of shape (num_envs, 3)
            env_ids: environment ids to reset

        """
        super().reset(obs, params)
        tcp_pose_b = obs["tcp_pose_b"]
        # Use the params for TCP position and keep the current TCP orientation
        goal_pose_b = torch.cat((params[:, 0:3], tcp_pose_b[:, 3:7]), dim=1)
        goal_ee_pose = self._compute_goal_ee_pose_b_from_goal_tcp_b(goal_pose_b, obs["tcp_offset"])

        self.diff_ik.set_command(goal_ee_pose[:, 0:3], ee_quat=goal_ee_pose[:, 3:7], env_ids=env_ids)


class PoseAbsIKEEPolicy(IKEEPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):
    """A policy that produces pose ."""

    def __init__(self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction]) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.

        """
        super().__init__(obs_spec, action_spec)
        self.diff_ik = DifferentialIKController(
            device=self._obs_spec.device, command_type="pose", use_relative_mode=False
        )

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the PoseAbsolute IK EE Policy by setting the command of the DiffIK Controlller.

        Args:
            obs: dict of bound functions
            params: target pose XYZ + Quat (num_envs, 7)
            env_ids: environment ids to reset

        """
        super().reset(obs, params)
        goal_pose = self._compute_goal_ee_pose_b_from_goal_tcp_b(params, obs["tcp_offset"])
        self.diff_ik.set_command(goal_pose, env_ids=env_ids)


class XYZRPYAbsIKEEPolicy(IKEEPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):
    """A policy that produces pose ."""

    def __init__(self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction]) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.

        """
        super().__init__(obs_spec, action_spec)
        self.diff_ik = DifferentialIKController(
            device=self._obs_spec.device, command_type="pose", use_relative_mode=False
        )

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the PoseAbsolute IK EE Policy by setting the command of the DiffIK Controlller.

        Args:
            obs: dict of bound functions
            params: target pose XYZ + RPY (num_envs, 6)
            env_ids: environment ids to reset

        """
        super().reset(obs, params)
        target_quat_b = quat_from_euler_xyz(params[:, 3], params[:, 4], params[:, 5])
        goal_tcp_b = torch.cat((params[:, 0:3], target_quat_b), dim=1)
        goal_pose = self._compute_goal_ee_pose_b_from_goal_tcp_b(goal_tcp_b, obs["tcp_offset"])
        self.diff_ik.set_command(goal_pose, env_ids=env_ids)


class OrientAbsIKEEPolicy(IKEEPolicy[TBPolicyObs, TBAction], Generic[TBPolicyObs, TBAction]):
    """A policy that produces pose ."""

    def __init__(self, obs_spec: ObservationSpec[TBPolicyObs], action_spec: ActionSpec[TBAction]) -> None:
        """Initialize the policy.

        Args:
            obs_spec: The observation specification.
            action_spec: The action specification.

        """
        super().__init__(obs_spec, action_spec)
        self.diff_ik = DifferentialIKController(
            device=self._obs_spec.device, command_type="pose", use_relative_mode=False
        )

    def reset(self, obs: TBPolicyObs, params: Any = None, env_ids: torch.Tensor = None) -> None:
        """Reset the PoseAbsolute IK EE Policy by setting the command of the DiffIK Controlller.

        Args:
            obs: dict of bound functions
            params: target orientation Roll Pitch Yaw of shape (num_envs, 3)
            env_ids: environment ids to reset

        """
        super().reset(obs, params)
        tcp_pose_b = obs["tcp_pose_b"]
        target_quat = quat_from_euler_xyz(params[:, 0], params[:, 1], params[:, 2])
        # Keep the current TCP position and use the target orientation from the params
        goal_pose_b = torch.cat((tcp_pose_b[:, 0:3], target_quat), dim=1)
        goal_ee_pose = self._compute_goal_ee_pose_b_from_goal_tcp_b(goal_pose_b, obs["tcp_offset"])
        self.diff_ik.set_command(goal_ee_pose, env_ids=env_ids)
