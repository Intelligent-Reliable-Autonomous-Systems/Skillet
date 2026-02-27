"""Keyboard controller for SE(2) control (standalone, no Omni/Carb dependency)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from pynput import keyboard as pynput_keyboard

from ..device_base import DeviceBase, DeviceCfg


class Se2Keyboard(DeviceBase):
    r"""A keyboard controller for sending SE(2) commands as velocity commands.

    Uses pynput instead of Omniverse's carb/omni keyboard interface, so it can run
    outside of an Isaac Sim / Omniverse context.

    The command comprises of the base linear and angular velocity: :math:`(v_x, v_y, \omega_z)`.

    Key bindings:
        ====================== ========================= ========================
        Command                Key (+ve axis)            Key (-ve axis)
        ====================== ========================= ========================
        Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
        Move along y-axis      Numpad 4 / Arrow Right    Numpad 6 / Arrow Left
        Rotate along z-axis    Numpad 7 / Z              Numpad 9 / X
        ====================== ========================= ========================
    """

    def __init__(self, cfg: Se2KeyboardCfg):
        """Initialize the keyboard layer.

        Args:
            cfg: Configuration object for keyboard settings.

        """
        self.v_x_sensitivity = cfg.v_x_sensitivity
        self.v_y_sensitivity = cfg.v_y_sensitivity
        self.omega_z_sensitivity = cfg.omega_z_sensitivity
        self._sim_device = cfg.sim_device

        # Command buffer
        self._base_command = np.zeros(3)

        # Additional user-registered callbacks
        self._additional_callbacks: dict[str, Callable] = {}

        # Track held keys to guard against OS key-repeat
        self._pressed_keys: set[str] = set()

        # Build key->delta mappings
        self._create_key_bindings()

        # Start pynput listener in background thread
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def __del__(self):
        """Stop the keyboard listener."""
        if hasattr(self, "_listener") and self._listener.is_alive():
            self._listener.stop()

    def __str__(self) -> str:
        msg = f"Keyboard Controller for SE(2): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tReset all commands: L\n"
        msg += "\tMove forward   (along x-axis): Numpad 8 / Arrow Up\n"
        msg += "\tMove backward  (along x-axis): Numpad 2 / Arrow Down\n"
        msg += "\tMove right     (along y-axis): Numpad 4 / Arrow Right\n"
        msg += "\tMove left      (along y-axis): Numpad 6 / Arrow Left\n"
        msg += "\tYaw positively (along z-axis): Numpad 7 / Z\n"
        msg += "\tYaw negatively (along z-axis): Numpad 9 / X"
        return msg

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def reset(self):
        self._base_command.fill(0.0)
        self._pressed_keys.clear()

    def add_callback(self, key: str, func: Callable):
        """Register a function to call when a key is pressed.

        Args:
            key: Single character string (e.g. "P") or special key name
                 matching pynput's Key attribute names (e.g. "up", "num8").
            func: Zero-argument callable.

        """
        self._additional_callbacks[key.upper()] = func

    def advance(self) -> torch.Tensor:
        """Provides the result from keyboard event state.

        Returns:
            Tensor containing the linear (x, y) and angular velocity (z).

        """
        return torch.tensor(self._base_command, dtype=torch.float32, device=self._sim_device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key_to_name(self, key) -> str | None:
        """Convert a pynput key object to a normalised name string, or None.

        Regular character keys  → uppercase char, e.g. "Z"
        Special/numpad keys     → uppercase pynput name, e.g. "UP", "NUMPAD_8"
        """
        try:
            # Regular character key
            if key.char is not None:
                return key.char.upper()
        except AttributeError:
            pass

        # Special key — map pynput Key names to the names used in _INPUT_KEY_MAPPING
        _SPECIAL_KEY_MAP = {
            "up": "UP",
            "down": "DOWN",
            "left": "LEFT",
            "right": "RIGHT",
            "num8": "NUMPAD_8",
            "num2": "NUMPAD_2",
            "num4": "NUMPAD_4",
            "num6": "NUMPAD_6",
            "num7": "NUMPAD_7",
            "num9": "NUMPAD_9",
        }
        key_name = key.name if hasattr(key, "name") else None
        if key_name is not None:
            return _SPECIAL_KEY_MAP.get(key_name)

        return None

    def _on_press(self, key):
        name = self._key_to_name(key)
        if name is None:
            return

        if name == "L":
            self.reset()
            return

        # Guard against OS key-repeat firing duplicate additions
        if name in self._pressed_keys:
            return
        self._pressed_keys.add(name)

        if name in self._INPUT_KEY_MAPPING:
            self._base_command += self._INPUT_KEY_MAPPING[name]

        # User-registered callbacks
        if name in self._additional_callbacks:
            self._additional_callbacks[name]()

    def _on_release(self, key):
        name = self._key_to_name(key)
        if name is None:
            return

        self._pressed_keys.discard(name)

        if name in self._INPUT_KEY_MAPPING:
            self._base_command -= self._INPUT_KEY_MAPPING[name]

    def _create_key_bindings(self):
        """Creates default key bindings."""
        self._INPUT_KEY_MAPPING = {
            # forward
            "NUMPAD_8": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "UP": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            # backward
            "NUMPAD_2": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "DOWN": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            # right
            "NUMPAD_4": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            "LEFT": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            # left
            "NUMPAD_6": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            "RIGHT": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            # yaw positive
            "NUMPAD_7": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            "Z": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            # yaw negative
            "NUMPAD_9": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
            "X": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
        }


@dataclass
class Se2KeyboardCfg(DeviceCfg):
    """Configuration for SE2 keyboard devices."""

    v_x_sensitivity: float = 0.8
    v_y_sensitivity: float = 0.4
    omega_z_sensitivity: float = 1.0
    class_type: type[DeviceBase] = Se2Keyboard
