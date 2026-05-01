"""Common RL configuration for Kinova Gen3 tasks."""

from skillet.envs.util import configclass
from skillet.rl.cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoPolicyCfg


@configclass
class Gen3ReachXyzPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    actor = RslRlPpoPolicyCfg(
        hidden_dims=(64, 64),
        activation="elu",
        obs_normalization=True,
        init_noise_std=1.0,
    )
    critic = RslRlPpoPolicyCfg(
        hidden_dims=(64, 64),
        activation="elu",
        obs_normalization=True,
        init_noise_std=1.0,
    )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
        hierarchical_policy=False,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    experiment_name = "mj_gen3_reach_xyz_direct"
    save_interval = 50
    num_steps_per_env = 24
    max_iterations = 5000
