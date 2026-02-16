# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from skillet.envs.util import configclass
from skillet.rl.cfg import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlPpoPolicyCfg


@configclass
class Gen3LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift_cube_gen3"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    actor = RslRlPpoPolicyCfg(
        init_noise_std=1.0,
        hidden_dims=[256, 128, 64],
        activation="elu",
    )
    critic = RslRlPpoPolicyCfg(
        init_noise_std=1.0,
        hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
