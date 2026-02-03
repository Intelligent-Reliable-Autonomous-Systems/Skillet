# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Kinova Robotics arms.

The following configuration parameters are available:

* :obj:`KINOVA_JACO2_N7S300_CFG`: The Kinova JACO2 (7-Dof) arm with a 3-finger gripper.
* :obj:`KINOVA_JACO2_N6S300_CFG`: The Kinova JACO2 (6-Dof) arm with a 3-finger gripper.
* :obj:`KINOVA_GEN3_N7_CFG`: The Kinova Gen3 (7-Dof) arm with no gripper.

Reference: https://github.com/Kinovarobotics/kinova-ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaac_kinova.assets.utils import KINOVA_ASSET_DIR

##
# Configuration
##

KINOVA_GEN3_2F85_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{KINOVA_ASSET_DIR}/robots/kinova/kinova_gen3_robotiq_2f_85_action_graph.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.523599,
            "joint_3": 0.0,
            "joint_4": 1.5708,
            "joint_5": 0.0,
            "joint_6": 0.785398,
            "joint_7": 0.0,
            "finger_joint": 0.0,  # left outer knuckle joint for manipulation
            "right_outer_knuckle_joint": 0.0,
            "left_outer_finger_joint": 0.0,
            "right_outer_finger_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "right_inner_finger_knuckle_joint": 0.0,
            "left_inner_finger_knuckle_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-7]"],
            velocity_limit_sim=100.0,
            effort_limit_sim={
                "joint_[1-2]": 80.0,
                "joint_[3]": 40.0,
                "joint_[4-7]": 20.0,
            },
            stiffness={
                "joint_[1-3]": 4000.0,
                "joint_[5-7]": 1500.0,
            },
            damping={
                "joint_[1-3]": 1000.0,
                "joint_[4-7]": 500.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint",
                "right_outer_knuckle_joint",
                "left_outer_finger_joint",
                "right_outer_finger_joint",
                "left_inner_finger_joint",
                "right_inner_finger_joint",
                "right_inner_finger_knuckle_joint",
                "left_inner_finger_knuckle_joint",
            ],
            effort_limit_sim=10,
            stiffness=2000.0,
            damping=200.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Kinova Gen3 (7-Dof) arm with Robotiq_2f85 gripper."""

KINOVA_GEN3_2F85_HIGH_CFG = KINOVA_GEN3_2F85_CFG.copy()

KINOVA_GEN3_2F85_HIGH_CFG.spawn.rigid_props.disable_gravity = True
