"""Algorithm config for t1-velocity-rma (Phase 1: PPO with privileged encoders)."""

from colosseum.config.types.algorithm import PpoConfig
from colosseum.config.types.networks import PpoActorConfig, PpoCriticConfig


def booster_t1_rma_ppo_cfg() -> PpoConfig:
  return PpoConfig(
    name="RmaPPO",
    target="colosseum.algorithm.rma_ppo:RmaPPO",
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
    symmetry_loss_coef=1.0,
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
