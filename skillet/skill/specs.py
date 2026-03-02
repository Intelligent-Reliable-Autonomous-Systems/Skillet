"""Define common skill parameter specifications."""

from typing import TypeAlias

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Float, Int

from skillet.core.spaces import ActionSpec, ParameterizedDiscrete

XYZ_Params = Float[torch.Tensor, "b 3"]
"""XYZ parameters: torch.Tensor[(b, 3), float]"""
XYZ_Params_Spec = ActionSpec[XYZ_Params](
    space=gym.spaces.Box(low=-3.0, high=3.0, shape=(3,)),
    name="xyz",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)
"""Action specification for XYZ parameters."""

XYZ_QUAT_Params = Float[torch.Tensor, "b 7"]
"""XYZ + Quat parameters: torch.Tensor[(b, 7), float]"""
XYZ_QUAT_Params_Spec = ActionSpec[XYZ_QUAT_Params](
    space=gym.spaces.Box(
        low=np.array([-3.0, -3.0, -3.0, -1.0, -1.0, -1.0, -1.0]),
        high=np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0]),
        shape=(7,),
    ),
    name="xyz_quat",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)
"""Action specification for XYZ + Quat parameters."""

XYZ_RPY_Params = Float[torch.Tensor, "b 6"]
"""XYZ + RPY parameters: torch.Tensor[(b, 6), float]"""
XYZ_RPY_Params_Spec = ActionSpec[XYZ_RPY_Params](
    space=gym.spaces.Box(
        low=np.array([-3.0, -3.0, -3.0, -np.pi, -np.pi, -np.pi]),
        high=np.array([3.0, 3.0, 3.0, np.pi, np.pi, np.pi]),
        shape=(6,),
    ),
    name="xyz_rpy",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)
"""Action specification for XYZ + RPY parameters."""

ROLL_PITCH_YAW_Params = Float[torch.Tensor, "b 3"]
"""Roll Pitch Yaw parameters: torch.Tensor[(b, 3), float]"""
ROLL_PITCH_YAW_Params_Spec = ActionSpec[ROLL_PITCH_YAW_Params](
    space=gym.spaces.Box(
        low=np.array([-np.pi, -np.pi, -np.pi]),
        high=np.array([np.pi, np.pi, np.pi]),
        shape=(3,),
    ),
    name="roll_pitch_yaw",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)
"""Action specification for Roll Pitch Yaw parameters."""

XYZ_YAW_Params = Float[torch.Tensor, "b 4"]
"""XYZ + Yaw parameters: torch.Tensor[(b, 4), float]"""
XYZ_YAW_Params_Spec = ActionSpec[XYZ_YAW_Params](
    space=gym.spaces.Box(
        low=np.array([-3.0, -3.0, -3.0, -np.pi]),
        high=np.array([3.0, 3.0, 3.0, np.pi]),
        shape=(4,),
    ),
    name="xyz_yaw",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)
"""Action specification for XYZ + Yaw parameters."""

SELECTED_OPTIONS: TypeAlias = Int[torch.Tensor, "b"]
"""The indices of the selected options for each environment."""
SELECT_OPTIONS_SPEC_BATCHED = ActionSpec[SELECTED_OPTIONS](
    space=ParameterizedDiscrete(n="n_options"),
    name="select_options",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)
"""Batched action specification for selecting options.

Bindings:
- n_options: int
    The number of options to select from.

Make sure to bind the n_options parameter before using the action specification.
"""
