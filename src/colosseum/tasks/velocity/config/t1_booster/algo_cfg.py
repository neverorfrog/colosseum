"""Algorithm config: colosseum PPO with booster_gym-matched hyperparameters.

Identical to the 12-DOF task's algo config (actor [256,128,128], critic
[256,256,128], ELU, logstd init -2.0). The action and observation dims grow by
one (Waist), which the networks pick up automatically from the env.
"""

import math

from colosseum.config.types.algorithm import BoosterPpoConfig, PpoConfig
from colosseum.config.types.networks import PpoActorConfig, PpoCriticConfig


def booster_t1_booster_ppo_cfg() -> PpoConfig:
  return PpoConfig(
    name="PPO",
    target="colosseum.algorithm.ppo:PPO",
    learning_steps=750_000_000,
    num_steps_per_env=24,  # booster horizon_length
    gamma=0.995,
    lam=0.95,
    clip_param=0.2,
    use_clipped_value_loss=True,
    value_loss_coef=1.0,
    entropy_coef=0.01,
    num_learning_epochs=20,  # booster mini_epochs
    num_mini_batches=4,
    max_grad_norm=1.0,
    actor_learning_rate=1e-5,
    critic_learning_rate=1e-5,
    weight_decay=0.0,
    desired_kl=0.01,
    schedule="adaptive",
    obs_normalization=False,
    symmetry_loss_coef=1.0,  # booster symmetric_coef=10 (different loss scale)
    symmetry_critic_coef=0.0,
    symmetry_data_augmentation=True,
    actor=PpoActorConfig(
      hidden_layers=[256, 128, 128],
      activation="elu",
      init_noise_std=math.exp(-2.0),  # ~0.135
      min_noise_std=0.01,
    ),
    critic=PpoCriticConfig(hidden_layers=[256, 256, 128], activation="elu"),
  )


def booster_t1_booster_booster_ppo_cfg() -> BoosterPpoConfig:
  """booster_gym PPO (runner.py): only_positive_rewards, bound_loss, full-batch
  GAE recomputed each epoch, single shared LR. Adds left-right symmetry via
  rollout data augmentation (a deliberate divergence from booster's runner)."""
  return BoosterPpoConfig(
    learning_steps=750_000_000,
    num_steps_per_env=24,  # booster horizon_length
    gamma=0.995,
    lam=0.95,
    clip_param=0.2,
    use_clipped_value_loss=False,  # booster uses plain MSE
    value_loss_coef=1.0,
    entropy_coef=0.01,  # loss -= 0.01 * entropy (booster entropy_coef=-0.01)
    bound_coef=1.0,
    only_positive_rewards=True,
    num_learning_epochs=20,  # booster mini_epochs
    num_mini_batches=1,  # full-batch (BoosterPPO ignores this; kept for clarity)
    max_grad_norm=1.0,
    actor_learning_rate=1e-5,
    critic_learning_rate=1e-5,
    weight_decay=0.0,
    desired_kl=0.01,
    schedule="adaptive",
    obs_normalization=False,
    symmetry_loss_coef=1.0,
    symmetry_critic_coef=0.0,
    symmetry_data_augmentation=True,
    actor=PpoActorConfig(
      hidden_layers=[256, 128, 128],
      activation="elu",
      init_noise_std=math.exp(-2.0),  # booster logstd init = -2.0
      min_noise_std=0.01,
    ),
    critic=PpoCriticConfig(hidden_layers=[256, 256, 128], activation="elu"),
  )
