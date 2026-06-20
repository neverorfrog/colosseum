"""Algorithm config for t1-kick: ResidualAMPPPO.

Frozen walk skill (reads `loco_actor`, steered toward the ball via the twist
trick) + a trainable residual (reads `kick_actor`) blended by an orchestrator
(reads `orchestrator`). Asymmetric critic over the privileged `critic` group.
On top of ResidualPPO, an AMP discriminator scores the policy's motion against
the reference kick clip (`models/trajectories/kick_1.npz`) and contributes a
style reward blended with the task reward (see ResidualAMPPPO + the `amp` obs
group). The discriminator replaces the hand-tuned gait/pose shaping stack.

Per-group symmetry (data augmentation + symmetry loss) matches the walk skill's
training.

NOTE: WALK_CHECKPOINT must match the `loco_actor` obs layout and the action
layout this task declares (actuators-branch velocity conventions: per-axis
tracking, action scale 0.25, arms fixed). jun18_1 is the latest velocity model
on this branch; swap it if you retrain the walk.
"""

from colosseum.config.types.algorithm import (
  PretrainedSkillConfig,
  ResidualActorCfg,
  ResidualAmpPpoConfig,
)
from colosseum.config.types.networks import (
  OrchestratorConfig,
  PpoActorConfig,
  PpoCriticConfig,
)

WALK_CHECKPOINT = "models/t1-velocity/jun18_1/t1-velocity_ppo_jun18_1.pt"


def booster_t1_kick_amp_ppo_cfg() -> ResidualAmpPpoConfig:
  return ResidualAmpPpoConfig(
    name="ResidualAMPPPO",
    target="colosseum.algorithm.residual_amp_ppo:ResidualAMPPPO",
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
      residual_obs_group="kick_actor",
      orchestrator=OrchestratorConfig(hidden_layers=[512, 256], activation="elu"),
      init_favored_logit=4.0,
    ),
    freeze_base_normalizers=True,
    residual_action_penalty_coef=0.01,
    residual_weight_penalty_coef=0.05,
    # ----------------------------- AMP ----------------------------- #
    amp_motion_files=["models/trajectories/kick_1.npz"],
    amp_obs_group="amp",
    amp_anchor_body="Trunk",
    amp_reward_coef=0.5,
    amp_task_reward_lerp=0.3,  # 70% AMP / 30% task
    amp_discr_hidden_dims=[256, 256],
    amp_replay_buffer_size=100_000,
    amp_learning_rate=1e-3,
    amp_grad_pen_lambda=10.0,
  )
