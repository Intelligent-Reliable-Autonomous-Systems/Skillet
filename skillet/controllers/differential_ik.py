# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import ClassVar, Literal

import torch

from skillet.core.math import apply_delta_pose, compute_pose_error


class DifferentialIKController:
    r"""Differential inverse kinematics (IK) controller.

    This controller is based on the concept of differential inverse kinematics [1, 2] which is a method for computing
    the change in joint positions that yields the desired change in pose.

    .. math::

        \Delta \mathbf{q} &= \mathbf{J}^{\dagger} \Delta \mathbf{x} \\
        \mathbf{q}_{\text{desired}} &= \mathbf{q}_{\text{current}} + \Delta \mathbf{q}

    where :math:`\mathbf{J}^{\dagger}` is the pseudo-inverse of the Jacobian matrix :math:`\mathbf{J}`,
    :math:`\Delta \mathbf{x}` is the desired change in pose, and :math:`\mathbf{q}_{\text{current}}`
    is the current joint positions.

    To deal with singularity in Jacobian, the following methods are supported for computing inverse of the Jacobian:

    - "pinv": Moore-Penrose pseudo-inverse
    - "svd": Adaptive singular-value decomposition (SVD)
    - "trans": Transpose of matrix
    - "dls": Damped version of Moore-Penrose pseudo-inverse (also called Levenberg-Marquardt)


    .. caution::
        The controller does not assume anything about the frames of the current and desired end-effector pose,
        or the joint-space velocities. It is up to the user to ensure that these quantities are given
        in the correct format.

    Reference:

    1. `Robot Dynamics Lecture <https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf>`_
       by Marco Hutter (ETH Zurich)
    2. `Introduction to Inverse Kinematics <https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf>`_
       by Samuel R. Buss (University of California, San Diego)

    """

    default_ik_params: ClassVar[dict[str, dict[str, float]]] = {
        "pinv": {"k_val": 0.01},
        "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
        "trans": {"k_val": 1.0},
        "dls": {"lambda_val": 0.01},
    }

    def __init__(
        self,
        device: str | None = "cuda",
        command_type: Literal["pose", "position"] = "pose",
        use_relative_mode: bool = False,
        ik_method: Literal["pinv", "svd", "trans", "dls"] = "dls",  # "pinv",
    ) -> None:
        """Initialize the controller.

        Args:
            num_envs: The number of environments.
            device: The device to use for computations.
            command_type: Pose or position
            use_relative_mode: bool of if to output absolute or relative joint positions
            ik_method: Inverse kinematics method to use

        """
        # store inputs
        self.command_type = command_type
        self.use_relative_mode = use_relative_mode
        self.ik_method = ik_method
        self._device = device
        self.ik_params = self.default_ik_params[self.ik_method]
        # -- input command
        self._command = None
        self.ee_pos_des = None
        self.ee_quat_des = None

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
        if self.command_type == "position":
            return 3  # (x, y, z)
        if self.command_type == "pose" and self.use_relative_mode:
            return 6  # (dx, dy, dz, droll, dpitch, dyaw)
        return 7  # (x, y, z, qw, qx, qy, qz)

    """
    Operations.
    """

    def reset(self, n_envs: int | None = None, env_ids: torch.Tensor = None) -> None:
        """Reset the internals.

        Args:
            n_envs: The number of environment indices
            env_ids: The environment ids to reset

        """
        # create buffers
        if self._command is None or env_ids is None:
            assert n_envs is not None, "n_envs cannot be none when `self._command` is not set."
            self.ee_pos_des = torch.zeros(n_envs, 3, device=self._device)
            self.ee_quat_des = torch.zeros(n_envs, 4, device=self._device)
            self._command = torch.zeros(n_envs, self.action_dim, device=self._device)
        else:
            self.ee_pos_des[env_ids] = torch.zeros(env_ids.shape[0], 3, device=self._device)
            self.ee_quat_des[env_ids] = torch.zeros(env_ids.shape[0], 4, device=self._device)
            self._command[env_ids] = torch.zeros(env_ids.shape[0], self.action_dim, device=self._device)

    def set_command(
        self,
        command: torch.Tensor,
        ee_pos: torch.Tensor | None = None,
        ee_quat: torch.Tensor | None = None,
        env_ids: torch.Tensor = None,
    ) -> None:
        """Set target end-effector pose command.

        Based on the configured command type and relative mode, the method computes the desired end-effector pose.
        It is up to the user to ensure that the command is given in the correct frame. The method only
        applies the relative mode if the command type is ``position_rel`` or ``pose_rel``.

        Args:
            command: The input command in shape (N, 3) or (N, 6) or (N, 7).
            ee_pos: The current end-effector position in shape (N, 3).
                This is only needed if the command type is ``position_rel`` or ``pose_rel``.
            ee_quat: The current end-effector orientation (w, x, y, z) in shape (N, 4).
                This is only needed if the command type is ``position_*`` or ``pose_rel``.
            env_ids: The environment ids to reset

        Raises:
            ValueError: If the command type is ``position_*`` and :attr:`ee_quat` is None.
            ValueError: If the command type is ``position_rel`` and :attr:`ee_pos` is None.
            ValueError: If the command type is ``pose_rel`` and either :attr:`ee_pos` or :attr:`ee_quat` is None.

        """
        if self.ee_pos_des is None or self.ee_quat_des is None:
            raise ValueError(
                "Neither desired end-effector position nor orientation can be None. Call `reset()` to initialize "
                "the correct shape."
            )
        if env_ids is None:
            env_ids = torch.ones(command.shape[0], dtype=torch.bool, device=self._device)
        # store command
        self._command[env_ids] = command[env_ids]
        # compute the desired end-effector pose
        if self.command_type == "position":
            # we need end-effector orientation even though we are in position mode
            # this is only needed for display purposes
            if ee_quat is None:
                raise ValueError("End-effector orientation can not be None for `position_*` command type!")
            # compute targets
            if self.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("End-effector position can not be None for `position_rel` command type!")
                self.ee_pos_des[env_ids] = ee_pos[env_ids] + self._command[env_ids]
                self.ee_quat_des[env_ids] = ee_quat[env_ids]
            else:
                self.ee_pos_des[env_ids] = self._command[env_ids]
                self.ee_quat_des[env_ids] = ee_quat[env_ids]
        else:
            # compute targets
            if self.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    raise ValueError(
                        "Neither end-effector position nor orientation can be None for `pose_rel` command type!"
                    )
                self.ee_pos_des[env_ids], self.ee_quat_des[env_ids] = apply_delta_pose(
                    ee_pos[env_ids], ee_quat[env_ids], self._command[env_ids]
                )
            else:
                self.ee_pos_des[env_ids] = self._command[env_ids][:, 0:3]
                self.ee_quat_des[env_ids] = self._command[env_ids][:, 3:7]

    def compute(
        self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Compute the target joint positions that will yield the desired end effector pose.

        Args:
            ee_pos: The current end-effector position in shape (N, 3).
            ee_quat: The current end-effector orientation in shape (N, 4).
            jacobian: The geometric jacobian matrix in shape (N, 6, num_joints).
            joint_pos: The current joint positions in shape (N, num_joints).

        Returns:
            The target joint positions commands in shape (N, num_joints).

        """
        # compute the delta in joint-space
        if "position" in self.command_type:
            position_error = self.ee_pos_des - ee_pos
            jacobian_pos = jacobian[:, 0:3]
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=position_error, jacobian=jacobian_pos)
        else:
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
            )
            pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian)
        # return the desired joint positions
        return joint_pos + delta_joint_pos

    """
    Helper functions.
    """

    def _compute_delta_joint_pos(self, delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Compute the change in joint position that yields the desired change in pose.

        The method uses the Jacobian mapping from joint-space velocities to end-effector velocities
        to compute the delta-change in the joint-space that moves the robot closer to a desired
        end-effector position.

        Args:
            delta_pose: The desired delta pose in shape (N, 3) or (N, 6).
            jacobian: The geometric jacobian matrix in shape (N, 3, num_joints) or (N, 6, num_joints).

        Returns:
            The desired delta in joint space. Shape is (N, num-jointsß).

        """
        if self.ik_params is None:
            raise RuntimeError(f"Inverse-kinematics parameters for method '{self.ik_method}' is not defined!")
        # compute the delta in joint-space
        if self.ik_method == "pinv":  # Jacobian pseudo-inverse
            # parameters
            k_val = self.ik_params["k_val"]
            # computation
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.ik_method == "svd":  # adaptive SVD
            # parameters
            k_val = self.ik_params["k_val"]
            min_singular_value = self.ik_params["min_singular_value"]
            # computation
            # U: 6xd, S: dxd, V: d x num-joint
            u, s, vh = torch.linalg.svd(jacobian)
            s_inv = 1.0 / s
            s_inv = torch.where(s > min_singular_value, s_inv, torch.zeros_like(s_inv))
            jacobian_pinv = (
                torch.transpose(vh, dim0=1, dim1=2)[:, :, :6]
                @ torch.diag_embed(s_inv)
                @ torch.transpose(u, dim0=1, dim1=2)
            )
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.ik_method == "trans":  # Jacobian transpose
            # parameters
            k_val = self.ik_params["k_val"]
            # computation
            jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_joint_pos = k_val * jacobian_t @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.ik_method == "dls":  # damped least squares
            # parameters
            lambda_val = self.ik_params["lambda_val"]
            # computation
            jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
            delta_joint_pos = (
                jacobian_t @ torch.inverse(jacobian @ jacobian_t + lambda_matrix) @ delta_pose.unsqueeze(-1)
            )
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        else:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.ik_method}")

        return delta_joint_pos
