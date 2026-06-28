"""Algorithm config for t1-dribbling-residual-obstacles: ResidualPPO.

Same single-residual stack as the base dribbling task -- a frozen RMA walk skill
(reads `loco_actor`) + one trainable residual blended by an orchestrator -- but
the residual and orchestrator read obstacle-aware groups so the one residual can
both dribble and steer the ball around a path-blocking obstacle. Asymmetric
critic over the privileged `critic` group (which gets clean obstacle GT).
"""

from colosseum.config.types.algorithm import (
  PretrainedSkillConfig,
  ResidualActorCfg,
  ResidualPpoConfig,
)
from colosseum.config.types.networks import (
  OrchestratorConfig,
  PpoActorConfig,
  PpoCriticConfig,
)

WALK_CHECKPOINT = "models/t1-velocity-rma/jun26_1/t1-velocity-rma_rmappo_jun26_1.pt"


def booster_t1_dribbling_residual_obstacles_ppo_cfg() -> ResidualPpoConfig:
  return ResidualPpoConfig(
    name="ResidualPPO",
    target="colosseum.algorithm.residual_ppo:ResidualPPO",
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
    max_actor_learning_rate=1e-4,
    max_critic_learning_rate=1e-4,
    weight_decay=0.001,
    desired_kl=0.01,
    schedule="adaptive",
    obs_normalization=True,
    symmetry_loss_coef=5.0,
    symmetry_critic_coef=0.0,
    symmetry_data_augmentation=True,
    critic=PpoCriticConfig(hidden_layers=[512, 256, 128], activation="elu"),
    residual_actor=ResidualActorCfg(
      base_skills={
        "walk": PretrainedSkillConfig(
          checkpoint=WALK_CHECKPOINT,
          kind="rma",
          latent_dim=8,
          window_size=50,
          term_name="env_params",
          actor=PpoActorConfig(
            hidden_layers=[512, 256, 128],
            activation="elu",
            init_noise_std=0.8,
            min_noise_std=0.01,
          ),
          obs_group="loco_actor",
        ),
      },
      residual_actor=PpoActorConfig(
        hidden_layers=[512, 256, 128],
        activation="elu",
        init_noise_std=0.8,
        min_noise_std=0.01,
      ),
      residual_obs_group="obstacle_residual",
      orchestrator_obs_group="obstacle_orchestrator",
      orchestrator=OrchestratorConfig(hidden_layers=[512, 256], activation="elu"),
      init_favored_logit=4.0,
    ),
    freeze_base_normalizers=True,
    residual_action_penalty_coef=0.01,
    residual_weight_penalty_coef=0.05,
  )
