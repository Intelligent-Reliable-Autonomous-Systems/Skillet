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
        joint_pos = obs["joint_pos"]
        arm_joint_pos = self.diff_ik.compute(ee_pose_b[:, 0:3], ee_pose_b[:, 3:7], jacobians, joint_pos)
        return torch.cat(
            (arm_joint_pos, torch.zeros(size=(arm_joint_pos.shape[0],), device=self._obs_spec.device).unsqueeze(1)),
            dim=1,
        )

    def reset(self, obs: TBPolicyObs, params: Any = None) -> None:
        """Reset the policy. Useful if policy is stateful."""
        n_envs = self._obs_spec.n_envs_from(obs)
        self._params = params
        self.diff_ik.reset(n_envs)
        self.tcp_offset = obs["tcp_offset"]

    def _compute_goal_ee_pose_b_from_goal_tcp_xyz_b(
        self, pose_b: torch.Tensor, tcp_offset: torch.Tensor
    ) -> torch.Tensor:
        """Compute the goal end effector pose (xyz, quat) from the goal TCP pose in XYZ RPY.

        Args:
            pose_b: The goal TCP pose in the shape (N,6) relative to the robot base frame
            tcp_offset: The offset of the tcp frame from the end effector


        Returns:
            The goal end effector pose in shape (N,7)

        """
        goal_tcp_quat_b = quat_from_euler_xyz(pose_b[:, 3], pose_b[:, 4], pose_b[:, 5])
        goal_tcp_pos_b = pose_b[:, 0:3]

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

    def reset(self, obs: TBPolicyObs, params: Any = None) -> None:
        """Reset the PoseAbsolute IK EE Policy by setting the command of the DiffIK Controlller.

        Args:
            obs: dict of bound functions
            params: parameters of shape (num_envs, 6)

        """
        super().reset(obs, params)
        ee_pose_b = obs["ee_pose_b"]
        tcp_xyz_b = obs["tcp_xyz_b"]
        goal_ee_pose = self._compute_goal_ee_pose_b_from_goal_tcp_xyz_b(
            torch.cat((params[:, 0:3], tcp_xyz_b[:, 3:6]), dim=1), obs["tcp_offset"]
        )

        self.diff_ik.set_command(goal_ee_pose[:, 0:3], ee_quat=ee_pose_b[:, 3:7])


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

    def reset(self, obs: TBPolicyObs, params: Any = None) -> None:
        """Reset the PoseAbsolute IK EE Policy by setting the command of the DiffIK Controlller.

        Args:
            obs: dict of bound functions
            params: parameters of shape (num_envs, 6)

        """
        super().reset(obs, params)
        goal_pose = self._compute_goal_ee_pose_b_from_goal_tcp_xyz_b(params[:, :6], obs["tcp_offset"])
        self.diff_ik.set_command(goal_pose)
