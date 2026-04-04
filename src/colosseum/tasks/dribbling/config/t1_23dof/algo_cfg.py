"""Algorithm configurations for Booster T1 dribbling task."""

from colosseum.config.types.algorithm import PpoConfig
from colosseum.config.types.networks import PpoActorConfig, PpoCriticConfig

# Ensure DribblingPPO is registered before config references it.
import colosseum.tasks.dribbling.dribbling_ppo  # noqa: F401


def booster_t1_dribbling_ppo_cfg() -> PpoConfig:
  return PpoConfig(
    name="PPO",
    target="colosseum.tasks.dribbling.dribbling_ppo:DribblingPPO",
    learning_steps=500_000_000,
    num_steps_per_env=24,
    gamma=0.99,
    lam=0.95,
    clip_param=0.2,
    use_clipped_value_loss=True,
    value_loss_coef=1.0,
    entropy_coef=0.01,
    num_learning_epochs=5,
    num_mini_batches=4,
    max_grad_norm=1.0,
    learning_rate=1e-3,
    desired_kl=0.01,
    schedule="adaptive",
    obs_normalization=True,
    actor=PpoActorConfig(hidden_layers=[512, 256, 128], activation="elu"),
    critic=PpoCriticConfig(hidden_layers=[512, 256, 128], activation="elu"),
  )
