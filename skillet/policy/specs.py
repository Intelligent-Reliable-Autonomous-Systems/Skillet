import numpy as np
import torch
from jaxtyping import Float

from skillet.core import ActionSpec
from skillet.core.spaces import ParameterizedBox

JOINTS = Float[torch.Tensor, "b n_joints"]


JOINTS_SPEC = ActionSpec[JOINTS](
    space=ParameterizedBox(low=-float("inf"), high=float("inf"), shape=("n_joints",), dtype=np.float32),
    name="joints_vel",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)

"""
TWIST_Params = Float[torch.Tensor, "b 6+n_gripper_joints"]


TWIST_TCP_SPEC = ActionSpec[TWIST_Params](
    space=ParameterizedBox(low=-float("inf"), high=float("inf"), shape=(6 + "n_gripper_joints",), dtype=np.float32),
    name="tcp_twist",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)


MOVEIT_Params = Float[torch.Tensor, "b 6+n_gripper_joints"]


MOVEIT_TCP_SPEC = ActionSpec[MOVEIT_Params](
    space=ParameterizedBox(low=-float("inf"), high=float("inf"), shape=(6 + "n_gripper_joints",), dtype=np.float32),
    name="tcp_moveit",
    is_torch=True,
    is_batched=True,
    n_envs=-1,
)
"""
