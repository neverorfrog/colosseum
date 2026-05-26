"""Algorithm configurations for Booster T1 dribbling task."""

from colosseum.config.types.algorithm import DaggerPpoConfig, RmaPPOConfig
from colosseum.config.types.networks import PpoActorConfig, PpoCriticConfig


def booster_t1_dribbling_ppo_cfg() -> RmaPPOConfig:
  return RmaPPOConfig(
    learning_steps=300_000_000,
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
    actor_learning_rate=1e-3,
    critic_learning_rate=1e-3,
    desired_kl=0.01,
    schedule="adaptive",
    obs_normalization=True,
    actor=PpoActorConfig(hidden_layers=[512, 256, 128], activation="elu"),
    critic=PpoCriticConfig(hidden_layers=[512, 256, 128], activation="elu"),
  )


def booster_t1_dribbling_dagger_ppo_cfg(
  teacher_checkpoint: str,
) -> DaggerPpoConfig:
  return DaggerPpoConfig(
    name="DaggerRmaPPO",
    target="colosseum.algorithm.dagger_rma_ppo:DaggerRmaPPO",
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
    teacher_checkpoint=teacher_checkpoint,
    teacher_obs_dim=0,  # Infer directly from the teacher checkpoint.
    teacher_obs_start_idx=0,  # Dribbling teacher and student share the same actor obs layout.
    teacher_actor_hidden_layers=(512, 256, 128),
    teacher_actor_activation="elu",
    imitation_coef=0.5,
    imitation_annealing_steps=400_000_000,
  )
