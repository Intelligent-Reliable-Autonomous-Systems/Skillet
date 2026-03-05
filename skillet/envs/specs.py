"""Observation specifications for the environment."""

from collections.abc import Mapping

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Float

from skillet.core import ObservationSpec
from skillet.core.spaces import BatchedSpaceItem, ParameterizedBox

BxN_Obs = Float[torch.Tensor, "b n"]
"""A B-batched N-dim vector observation: torch.Tensor[(b, n), float]"""
BxM_Action = Float[torch.Tensor, "b m"]
"""A B-batched M-dim vector action: torch.Tensor[(b, m), float]"""

RGBD_SPEC_BATCHED = ObservationSpec[dict[str, BatchedSpaceItem]](
    space=gym.spaces.Dict(
        {
            "rgb": ParameterizedBox(low=0, high=255, shape=(3, "height", "width"), dtype=np.uint8),
            # Depth is normalized to float32 meters for downstream perception.
            "depth": ParameterizedBox(low=0.0, high=10.0, shape=(1, "height", "width"), dtype=np.float32),
            "intrinsic_k": gym.spaces.Box(low=0.0, high=2000.0, shape=(3, 3), dtype=np.float32),
            "camera_pose": gym.spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32),
            "timestamp": gym.spaces.Box(low=0.0, high=1e10, shape=(), dtype=np.float64),
        }
    ),
    name="rgb-d",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)

IKEE_Obs = Mapping[str, Float[torch.Tensor, "b ..."]]
OSC_Obs = Mapping[str, Float[torch.Tensor, "b ..."]]
TWIST_TCP_Obs = Mapping[str, Float[torch.Tensor, "b ..."]]
IK_EE_SPEC_BATCHED = ObservationSpec[IKEE_Obs](
    space=gym.spaces.Dict(
        {
            "joint_pos": ParameterizedBox(low=-torch.pi, high=torch.pi, shape=("n_joints",)),
            "joint_vel": ParameterizedBox(low=-10.0, high=10.0, shape=("n_joints",)),
            "tcp_offset": gym.spaces.Box(low=-1.0, high=1.0, shape=(7,)),
            "jacobians": ParameterizedBox(low=-1, high=1, shape=(6, "n_arm_joints")),
            "ee_pose_b": gym.spaces.Box(low=-1.0, high=1.0, shape=(7,)),
            "tcp_pose_b": gym.spaces.Box(low=-1.0, high=1.0, shape=(7,)),
            "gripper_lim": gym.spaces.Box(low=0.0, high=1.0, shape=(2,)),
            "gripper": ParameterizedBox(low=0.0, high=1.0, shape=("n_gripper_joints",)),
            "joint_lims": ParameterizedBox(low=-torch.pi, high=torch.pi, shape=(2, "n_joints")),
        }
    ),
    name="ik_ee",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)

OSC_SPEC_BATCHED = ObservationSpec[OSC_Obs](
    space=gym.spaces.Dict(
        {
            "joint_pos": ParameterizedBox(low=-torch.pi, high=torch.pi, shape=("n_joints",)),
            "joint_vel": ParameterizedBox(low=-10.0, high=10.0, shape=("n_joints",)),
            "tcp_offset": gym.spaces.Box(low=-1.0, high=1.0, shape=(7,)),
            "jacobians": ParameterizedBox(low=-1, high=1, shape=(6, "n_arm_joints")),
            "ee_pose_b": gym.spaces.Box(low=-1.0, high=1.0, shape=(7,)),
            "tcp_pose_b": gym.spaces.Box(low=-1.0, high=1.0, shape=(7,)),
            "gripper_lim": gym.spaces.Box(low=0.0, high=1.0, shape=(2,)),
            "gripper": ParameterizedBox(low=0.0, high=1.0, shape=("n_gripper_joints",)),
            "joint_lims": ParameterizedBox(low=-torch.pi, high=torch.pi, shape=(2, "n_joints")),
            "mass_matrix": ParameterizedBox(low=-1, high=1, shape=("n_arm_joints", "n_arm_joints")),
            "joint_gravity": ParameterizedBox(low=-10.0, high=10.0, shape=("n_arm_joints",)),
            "ee_vel_b": gym.spaces.Box(low=-10.0, high=10.0, shape=(7,)),
            "joint_centers": ParameterizedBox(low=-torch.pi, high=torch.pi, shape=("n_arm_joints",)),
        }
    ),
    name="osc_ee",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)

TWIST_SPEC_BATCHED = ObservationSpec[TWIST_TCP_Obs](
    space=gym.spaces.Dict(
        {
            "tcp_pose_b": gym.spaces.Box(low=-1.0, high=1.0, shape=(7,)),
            "gripper_lim": gym.spaces.Box(low=0.0, high=1.0, shape=(2,)),
            "gripper": ParameterizedBox(low=0.0, high=1.0, shape=("n_gripper_joints",)),
            "dt": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
        }
    ),
    name="twist_tcp",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)
