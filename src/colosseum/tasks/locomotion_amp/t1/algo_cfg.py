"""Algorithm config for t1-velocity-amp: AmpPPO (BeyondAMP-style walking).

Plain PPO learns velocity-tracked walking from scratch; an AMP discriminator
scores the policy's motion against the reference walk clip
(``models/trajectories/walk_1.npz``) and contributes a style reward blended with
the task reward (see AmpPPO + the ``amp`` obs group). The discriminator replaces
the hand-tuned gait/pose shaping stack of the base velocity task.
"""

from colosseum.config.types.algorithm import AmpPpoConfig
from colosseum.config.types.networks import PpoActorConfig, PpoCriticConfig
from colosseum.mdp.amp_obs import AMP_BODY_JOINTS


def booster_t1_amp_ppo_cfg() -> AmpPpoConfig:
  return AmpPpoConfig(
    name="AmpPPO",
    target="colosseum.algorithm.amp_ppo:AmpPPO",
    learning_steps=1_000_000_000,
    num_steps_per_env=24,
    gamma=0.995,
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
    # ----------------------------- AMP ----------------------------- #
    amp_motion_files=["models/trajectories/walk_3.npz"],
    amp_obs_group="amp",
    amp_include_base_vel=False,  # joints-only → speed-agnostic discriminator
    amp_joint_names=list(AMP_BODY_JOINTS),  # head excluded; mirrors the amp obs group
    amp_anchor_body="Trunk",
    amp_reward_coef=0.5,
    amp_task_reward_lerp=0.5,  # 70% AMP / 30% task
    amp_discr_hidden_dims=[256, 256],
    amp_replay_buffer_size=100_000,
    amp_learning_rate=1e-4,
    amp_grad_pen_lambda=20.0,
  )
