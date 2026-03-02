"""Common RL configuration for Kinova Gen3 tasks."""

from skillet.envs.util import configclass
from skillet.rl.cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoPolicyCfg


@configclass
class KinovaLiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "mj_kinova_lift_cube"
    run_name = ""
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    resume = False
    empirical_normalization = False
    actor = RslRlPpoPolicyCfg(init_noise_std=1.0, hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    critic = RslRlPpoPolicyCfg(
        init_noise_std=1.0, hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True
    )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
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


'''

def kinova_ppo_runner_cfg(
    experiment_name: str = "kinova_lift_cube",
) -> RslRlOnPolicyRunnerCfg:
    """Create PPO runner configuration for Kinova Gen3 tasks.

    Args:
        experiment_name: Name for the experiment/logging.

    Returns:
        PPO runner configuration.

    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            stochastic=True,
            init_noise_std=1.0,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            stochastic=False,
            init_noise_std=1.0,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=experiment_name,
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=5_000,
    )
'''
