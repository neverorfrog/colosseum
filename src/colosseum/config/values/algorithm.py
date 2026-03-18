from colosseum.config.types.algorithm import PpoConfig

PPO_DEFAULT = PpoConfig(
  name="PPO",
  target="colosseum.algorithm.ppo:PPO",
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
  use_rich_logging=True,
  seed=42,
)

DEFAULTS = {
  "ppo": PPO_DEFAULT,
}
