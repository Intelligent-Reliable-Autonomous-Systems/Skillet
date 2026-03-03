# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package providing interfaces to different teleoperation devices.

Currently, the following categories of devices are supported:

* **Keyboard**: Standard keyboard with WASD and arrow keys.
* **Spacemouse**: 3D mouse with 6 degrees of freedom.
* **Gamepad**: Gamepad with 2D two joysticks and buttons. Example: Xbox controller.

All device interfaces inherit from the :class:`DeviceBase` class, which provides a
common interface for all devices. The device interface reads the input data when
the :meth:`DeviceBase.advance` method is called. It also provides the function :meth:`DeviceBase.add_callback`
to add user-defined callback functions to be called when a particular input is pressed from
the peripheral device.
"""

from .device_base import DeviceBase as DeviceBase
from .device_base import DeviceCfg as DeviceCfg
from .device_base import DevicesCfg as DevicesCfg
from .keyboard import Se2Keyboard as Se2Keyboard
from .keyboard import Se2KeyboardCfg as Se2KeyboardCfg
from .keyboard import Se3Keyboard as Se3Keyboard
from .keyboard import Se3KeyboardCfg as Se3KeyboardCfg
from .retargeter_base import RetargeterBase as RetargeterBase
from .retargeter_base import RetargeterCfg as RetargeterCfg
from .spacemouse import Se2SpaceMouse as Se2SpaceMouse
from .spacemouse import Se2SpaceMouseCfg as Se2SpaceMouseCfg
from .spacemouse import Se3SpaceMouse as Se3SpaceMouse
from .spacemouse import Se3SpaceMouseCfg as Se3SpaceMouseCfg
from .vr import VRJoystick as VRJoystick
from .vr import VRJoystickCfg as VRJoystickCfg
from .vr import VRHeadset as VRHeadset
from .vr import VRHeadsetCfg as VRHeadsetCfg
from .teleop_device_factory import create_teleop_device as create_teleop_device
