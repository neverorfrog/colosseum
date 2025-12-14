"""RL configuration for Booster T1 velocity task."""

from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

def booster_t1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Booster T1 velocity task.

    This configuration uses PPO (Proximal Policy Optimization) for training
    the T1 humanoid robot on velocity tracking tasks.

    Network Architecture:
      - Actor (policy): 512 -> 256 -> 128 neurons with ELU activation
      - Critic (value): 512 -> 256 -> 128 neurons with ELU activation
      - Both use observation normalization for stable training

    Training Hyperparameters:
      - Learning rate: 1e-3 with adaptive scheduling
      - Discount factor (gamma): 0.99
      - GAE lambda: 0.95
      - PPO clip parameter: 0.2
      - Mini-batches: 4 per update
      - Learning epochs: 5 per iteration
      - Steps per env: 24 (with 4096 envs = 98,304 steps per iteration)

    Returns:
      RslRlOnPolicyRunnerCfg: Complete RL training configuration
    """
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            noise_std_type="log",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
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
        experiment_name="t1_velocity",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
    )
