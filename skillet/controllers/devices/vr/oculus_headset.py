"""VR headset controller for Oculus"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing_extensions import override
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..device_base import DeviceBase, DeviceCfg

from .joystick_listener import VRJoystickListener

from skillet.core.math import (
    apply_delta_pose,
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    euler_xyz_from_quat,
    subtract_frame_transforms,
)


class VRHeadset(DeviceBase):
    """A Headset controller using the relative positions of the controllers

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle gripper (open/close)    RC - A
        Move along x-axis              RC - Joystick R   RC - Joystick L
        Move along y-axis              RC - Joystick Up  RC - Joystick Down
        Move along z-axis              RC - Gripper      RC - Trigger
        Rotate along x-axis            LC - Joystick R   LC - Joytick L
        Rotate along y-axis            LC - Joystick Up  LC - Joystick Down
        Rotate along z-axis            LC - Gripper      LC - Trigger
        ============================== ================= =================
    """

    def __init__(self, cfg: VRHeadsetCfg):
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.gripper_term = cfg.gripper_term
        self._sim_device = cfg.sim_device
        self._headset_offset = cfg.headset_offset

        self._workspace_lim = np.asarray([[0.1, -0.5, 0.1, -1.57, -1.57, -1.57], [0.7, 0.5, 0.7, 1.57, 1.57, 1.57]])
        # Command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)
        # self._tcp_xyz_des_b = None
        self._tcp_xyz_des_b = np.asarray([0.5, 0.0, 0.2, 3.14, 0.26, 3.14])

        # PID gains
        self.Kp_pos = 1.0
        self.Ki_pos = 0.0
        self.Kd_pos = 0.1
        self.Kp_rot = 1.0
        self.Ki_rot = 0.0
        self.Kd_rot = 0.1

        # PID integrals
        self.integral_pos = np.zeros(3)
        self.integral_rot = np.zeros(3)

        # Last errors for derivative
        self.last_error_pos = np.zeros(3)
        self.last_error_rot = np.zeros(3)

        self._additional_callbacks: dict[str, Callable] = {}
        self._listener = VRJoystickListener(host=cfg.host, port=cfg.port)

    def __del__(self):
        """Stop the VR joystick listener."""
        if hasattr(self, "_listener") and self._listener.is_running:
            self._listener.close()

    def __str__(self) -> str:
        msg = f"VR Joystick Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t--------------------------------------------------\n"
        msg += "\tToggle gripper (open/close): RC - A\n"
        msg += "\n"
        msg += "\tTranslation Controls (Right Controller)\n"
        msg += "\tMove arm along x-axis: RC - Joystick Right / Left\n"
        msg += "\tMove arm along y-axis: RC - Joystick Up / Down\n"
        msg += "\tMove arm along z-axis: RC - Gripper / Trigger\n"
        msg += "\n"
        msg += "\tRotation Controls (Left Controller)\n"
        msg += "\tRotate arm along x-axis: LC - Joystick Right / Left\n"
        msg += "\tRotate arm along y-axis: LC - Joystick Up / Down\n"
        msg += "\tRotate arm along z-axis: LC - Gripper / Trigger\n"
        return msg

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def reset(self):
        self._close_gripper = False
        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)

    @override
    def advance(self, curr_tcp_pose: torch.Tensor, dt: float = 1 / 60) -> torch.Tensor:
        """Return the current command as a tensor.

        Returns:
            torch.Tensor: [x, y, z, rx, ry, rz] or [x, y, z, rx, ry, rz, gripper]

        """
        tcp = curr_tcp_pose.squeeze().cpu().numpy()
        r, p, y = euler_xyz_from_quat(curr_tcp_pose[:, 3:7])
        np.set_printoptions(precision=3, suppress=True)
        print(np.asarray([tcp[0], tcp[1], tcp[2], r.item(), p.item(), y.item()]))
        print(self._tcp_xyz_des_b)
        # self._read_latest()
        if self._tcp_xyz_des_b is None:
            command = torch.zeros((6,))
            if self.gripper_term:
                gripper_value = 1.0 if self._close_gripper else -1.0
                command = (
                    torch.concatenate((command, torch.as_tensor([gripper_value])), dim=0)
                    .to(self._sim_device)
                    .to(torch.float32)
                )
            return command
        r, p, y = euler_xyz_from_quat(curr_tcp_pose[:, 3:7])
        robot_xyz_b = torch.cat((curr_tcp_pose[:, 0:3].squeeze(), r, p, y), dim=-1).cpu().numpy().squeeze()

        # Compute errors
        error_pos = robot_xyz_b[:3] - self._tcp_xyz_des_b[:3]
        error_rot = robot_xyz_b[3:6] - self._tcp_xyz_des_b[3:6]

        # Update integral terms
        self.integral_pos += error_pos * dt
        self.integral_rot += error_rot * dt

        # Compute derivative terms
        derivative_pos = (error_pos - self.last_error_pos) / dt
        derivative_rot = (error_rot - self.last_error_rot) / dt

        # PID control for translation
        delta_pos = self.Kp_pos * error_pos + self.Ki_pos * self.integral_pos + self.Kd_pos * derivative_pos
        self._delta_pos = np.clip(delta_pos, -self.pos_sensitivity, self.pos_sensitivity)
        self._delta_pos[1] = -self._delta_pos[1]

        # PID control for rotation (Euler -> rotation vector)
        delta_rot = self.Kp_rot * error_rot + self.Ki_rot * self.integral_rot + self.Kd_rot * derivative_rot
        self._delta_rot = np.clip(delta_rot, -self.rot_sensitivity, self.rot_sensitivity)
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()

        # Combine translation + rotation for twist command
        command = np.concatenate([self._delta_pos, rot_vec])

        # Append gripper command if needed
        if self.gripper_term:
            gripper_value = 1.0 if self._close_gripper else -1.0
            command = np.append(command, gripper_value)

        # Save last errors
        self.last_error_pos = error_pos
        self.last_error_rot = error_rot
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def _read_latest(self) -> None:
        """Read the latest input from the VR Joystick."""

        s = self._listener.read_latest()

        if s is not None:
            rc_pose_w = torch.as_tensor(s["right_controller"]["pose"]).unsqueeze(0)
            h_pose_w = torch.as_tensor(s["headset"]["pose"]).unsqueeze(0)
            h_pose_w[2] = h_pose_w[2] - self._headset_offset  # Offset Z axis by 0.5 meter

            rc_pose_b = subtract_frame_transforms(
                h_pose_w[:, 0:3], h_pose_w[:, 3:7], rc_pose_w[:, 0:3], rc_pose_w[:, 3:7]
            )
            r, p, y = euler_xyz_from_quat(rc_pose_b[:, 3:7])
            rc_xyz_b = torch.cat((rc_pose_b, r, p, y), dim=-1).cpu().numpy().squeeze()

            self._tcp_xyz_des_b = np.clip(rc_xyz_b, self._workspace_lim[0], self._workspace_lim[1])

            if s["right_controller"]["inputs"]["A_button"]:
                self._close_gripper = not self._close_gripper

    def add_callback(self, key: Any, func: Callable):
        """Callback function."""
        pass


@dataclass
class VRHeadsetCfg(DeviceCfg):
    """Configuration for VR Joystick"""

    gripper_term: bool = True
    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8
    headset_offset: float = 0.5
    retargeters: None = None
    class_type: type[DeviceBase] = VRHeadset
    host: str = "192.168.2.33"
    port: int = 5555
