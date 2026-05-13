"""Algorithm config for T1 velocity task with symmetry loss."""

from colosseum.config.types.algorithm import PpoConfig
from colosseum.config.types.networks import PpoActorConfig, PpoCriticConfig


def booster_t1_symmetric_ppo_cfg() -> PpoConfig:
    return PpoConfig(
        name="PPO",
        target="colosseum.algorithm.ppo:PPO",
        learning_steps=1_000_000_000,
        num_steps_per_env=24,
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        use_clipped_value_loss=True,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        max_grad_norm=1.0,
        actor_learning_rate=1e-5,
        critic_learning_rate=1e-5,
        weight_decay=0.001,
        desired_kl=0.01,
        schedule="adaptive",
        obs_normalization=True,
        symmetry_loss_coef=2.0,
        symmetry_critic_coef=0.0,
        symmetry_data_augmentation=True,
        actor=PpoActorConfig(
            hidden_layers=[512, 256, 128],
            activation="elu",
            init_noise_std=0.8,
            min_noise_std=0.01,
        ),
        critic=PpoCriticConfig(hidden_layers=[512, 256, 128], activation="elu"),
    )


def booster_t1_symmetric_rsl_rl_runner_cfg():
    """RSL-RL runner config (without symmetry, since RSL-RL doesn't support it)."""
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="t1_velocity_symmetric",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
    )
