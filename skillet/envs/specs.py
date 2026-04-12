"""Observation specifications for the environment."""

from collections.abc import Mapping
from typing import Generic, Protocol, TypeAlias, TypeVar

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Float, UInt8

from skillet.core import ObservationSpec
from skillet.core.spaces import ActionSpec, ParameterizedBox

N_Obs = Float[torch.Tensor, "n"]
"""Environment observation: torch.Tensor[(n), float]"""
M_Action = Float[torch.Tensor, "m"]
"""Environment action: torch.Tensor[(m), float]"""
SelectedSkill: TypeAlias = int
"""Selected skill: int"""
BxN_Obs = Float[torch.Tensor, "b n"]
"""A B-batched N-dim vector observation: torch.Tensor[(b, n), float]"""
BxM_Action = Float[torch.Tensor, "b m"]
"""A B-batched M-dim vector action: torch.Tensor[(b, m), float]"""

TNPOrTensor = TypeVar("TNPOrTensor", bound=torch.Tensor | np.ndarray)
"""A numpy or torch tensor generic type."""


class RGBD_Obs(Protocol, Generic[TNPOrTensor]):  # noqa: N801
    """An RGB-D observation with intrinsics and camera pose."""

    rgb: UInt8[TNPOrTensor, "... 3 h w"]
    """An RGB CHW image. UInt8[torch.Tensor | np.ndarray, '... 3 h w']"""
    depth: Float[TNPOrTensor, "... 1 h w"]
    """A depth HW image. Float[TNPOrTensor, '... 1 h w']"""
    intrinsic_k: Float[TNPOrTensor, "3 3"]
    """A 3x3 camera intrinsic matrix. Float[TNPOrTensor, '3 3']"""
    camera_pose: Float[TNPOrTensor, "7"]
    """A 7D camera pose. Float[TNPOrTensor, '7']"""
    timestamp: Float[TNPOrTensor, ""]
    """A timestamp. Float[TNPOrTensor, '']"""


class RGBD_Gripper_Obs(Protocol, Generic[TNPOrTensor]):  # noqa: N801
    """An RGB-D + TCP observation with intrinsics, camera pose tcp pose and gripper."""

    rgb: UInt8[TNPOrTensor, "... 3 h w"]
    """An RGB CHW image. UInt8[torch.Tensor | np.ndarray, '... 3 h w']"""
    depth: Float[TNPOrTensor, "... 1 h w"]
    """A depth HW image. Float[TNPOrTensor, '... 1 h w']"""
    intrinsic_k: Float[TNPOrTensor, "3 3"]
    """A 3x3 camera intrinsic matrix. Float[TNPOrTensor, '3 3']"""
    camera_pose: Float[TNPOrTensor, "7"]
    """A 7D camera pose. Float[TNPOrTensor, '7']"""
    timestamp: Float[TNPOrTensor, ""]
    """A timestamp. Float[TNPOrTensor, '']"""
    gripper: Float[TNPOrTensor, " ... n_gripper_joints"]
    """A N-d array of gripper positions"""
    tcp_pose: Float[TNPOrTensor, "... 7"]
    """A 7D tool frame pose. Float[TNPOrTensor, '7']"""


"""Type of RGB-D observation."""
RGBD_SPEC_BATCHED: ObservationSpec[RGBD_Obs] = ObservationSpec[RGBD_Obs[TNPOrTensor]](
    space=gym.spaces.Dict(
        {
            "rgb": ParameterizedBox(low=0, high=255, shape=(3, "height", "width"), dtype=np.uint8),
            # Depth is normalized to float32 meters for downstream perception.
            "depth": ParameterizedBox(low=0.0, high=10.0, shape=(1, "height", "width"), dtype=np.float32),
            "intrinsic_k": gym.spaces.Box(low=0.0, high=2000.0, shape=(3, 3), dtype=np.float32),
            "camera_pose": gym.spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32),
            "timestamp": gym.spaces.Box(low=0.0, high=1e10, shape=(), dtype=np.float32),
        }
    ),
    name="rgb-d",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)

RGBD_GRIPPER_SPEC_BATCHED: ObservationSpec[RGBD_Gripper_Obs] = ObservationSpec[RGBD_Gripper_Obs[TNPOrTensor]](
    space=gym.spaces.Dict(
        {
            "rgb": ParameterizedBox(low=0, high=255, shape=(3, "height", "width"), dtype=np.uint8),
            # Depth is normalized to float32 meters for downstream perception.
            "depth": ParameterizedBox(low=0.0, high=10.0, shape=(1, "height", "width"), dtype=np.float32),
            "intrinsic_k": gym.spaces.Box(low=0.0, high=2000.0, shape=(3, 3), dtype=np.float32),
            "camera_pose": gym.spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32),
            "timestamp": gym.spaces.Box(low=0.0, high=1e10, shape=(), dtype=np.float32),
            "tcp_pose_b": gym.spaces.Box(low=-1.0, high=1.0, shape=(7,)),
            "gripper": ParameterizedBox(low=0.0, high=1.0, shape=("n_gripper_joints",)),
        }
    ),
    name="rgbd-gripper",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)

IKEE_Obs = Mapping[str, Float[torch.Tensor, "b ..."]]
OSC_Obs = Mapping[str, Float[torch.Tensor, "b ..."]]
TWIST_TCP_Obs = Mapping[str, Float[torch.Tensor, "b ..."]]
MOVEIT_TCP_Obs = Mapping[str, Float[torch.Tensor, "b ..."]]
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

GRIPPER_SPEC_BATCHED = ObservationSpec[IKEE_Obs](
    space=gym.spaces.Dict(
        {
            "tcp_pose_b": gym.spaces.Box(low=-1.0, high=1.0, shape=(7,)),
            "gripper": ParameterizedBox(low=0.0, high=1.0, shape=("n_gripper_joints",)),
        }
    ),
    name="gripper",
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
            "ee_vel_b": gym.spaces.Box(low=-10.0, high=10.0, shape=(6,)),
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
            "ee_vel_b": gym.spaces.Box(low=-10, high=10, shape=(6,)),
            "joint_vel": ParameterizedBox(low=-10, high=10, shape=("n_joints",)),
            "joint_pos": ParameterizedBox(low=-np.pi, high=np.pi, shape=("n_joints",)),
            "joint_eff": ParameterizedBox(low=-np.pi, high=np.pi, shape=("n_joints",)),
        }
    ),
    name="twist_tcp",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)

MOVEIT_SPEC_BATCHED = ObservationSpec[MOVEIT_TCP_Obs](
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

# ========= Action specifications =========

TWIST_TCP_Action = Float[torch.Tensor, "b 6+n_gripper_joints"]
"""Action type for Twist TCP commands."""
TWIST_TCP_SPEC = ActionSpec[TWIST_TCP_Action](
    space=ParameterizedBox(low=-float("inf"), high=float("inf"), shape=("6 + n_gripper_joints",)),
    name="twist_tcp",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)

MOVEIT_Joint_Action = Float[torch.Tensor, "b n_joints"]
"""Action type for MoveIt joint commands."""
MOVEIT_Joint_SPEC = ActionSpec[MOVEIT_Joint_Action](
    space=ParameterizedBox(low=-float("inf"), high=float("inf"), shape=("n_joints",)),
    name="moveit_joint",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)

TCP_QUAT_Action = Float[torch.Tensor, "b 7+n_gripper_joints"]
"""Action type for MoveIt TCP + Quat commands."""
MOVEIT_TCP_QUAT_SPEC = ActionSpec[TCP_QUAT_Action](
    space=ParameterizedBox(low=-float("inf"), high=float("inf"), shape=("7 + n_gripper_joints",)),
    name="moveit_tcp_quat",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)
