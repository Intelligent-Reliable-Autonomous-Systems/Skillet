from skillet.envs.util import configclass
from skillet.rl.cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoPolicyCfg


@configclass
class Gen3LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2500
    save_interval = 50
    experiment_name = "mj_gen3_lift_cube_direct"
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
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        hierarchical_policy=False,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
