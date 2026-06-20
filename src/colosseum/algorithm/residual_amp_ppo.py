"""ResidualAMPPPO — ResidualPPO + an Adversarial Motion Prior.

Adds a style prior to the residual kick: an extra ``amp`` observation group feeds a
discriminator trained to tell the policy's motion transitions apart from a reference
clip. The discriminator's score becomes a per-step reward, blended with the task
reward. Everything that makes ResidualPPO (frozen walk + residual + orchestrator +
symmetry + ONNX export) is inherited untouched; AMP plugs into exactly two seams:

  1. Rollout — ``_blend_amp_reward`` (overriding the base no-op hook): turn the
     discriminator score into a reward, blend it in, and stash clean policy
     transitions into the AMP replay buffer.
  2. Learning — ``_update_discriminator``, run after the inherited PPO update: train
     the discriminator (LSGAN + gradient penalty) on policy vs. expert transitions.

The expert data only ever enters the discriminator loss; the reward path only feeds
the discriminator policy transitions. The two are coupled solely through the
discriminator's weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import torch.optim as optim
from mjlab.envs import ManagerBasedRlEnv

from colosseum.algorithm.amp import AMPDiscriminator, AMPReplayBuffer, AmpMotionDataset
from colosseum.algorithm.residual_ppo import ResidualPPO
from colosseum.algorithm.utils.normalization import EmpiricalNormalization
from colosseum.config.types.algorithm import ResidualAmpPpoConfig, register_algorithm


@register_algorithm("residual_amp_ppo", config_class=ResidualAmpPpoConfig)
class ResidualAMPPPO(ResidualPPO):
  """ResidualPPO with an AMP discriminator providing an adversarial style reward."""

  def __init__(
    self,
    config: ResidualAmpPpoConfig,
    env: ManagerBasedRlEnv,
    device: str | torch.device,
    log_fn: Callable[[dict[str, float], int], None],
    log_interval: int,
  ) -> None:
    super().__init__(
      config=config, env=env, device=device, log_fn=log_fn, log_interval=log_interval
    )
    assert isinstance(self.config, ResidualAmpPpoConfig)
    cfg = self.config

    amp_dim = self.env.observation_manager.group_obs_dim[cfg.amp_obs_group][0]

    # Fix A: the dataset subtracts these defaults so its joint angles match the
    # env's `joint_pos_rel` term. Robot joint order == npz joint column order.
    default_joint_pos = self.env.scene["robot"].data.default_joint_pos[0].detach()

    self.amp_motion_dataset = AmpMotionDataset(
      motion_files=cfg.amp_motion_files,
      default_joint_pos=default_joint_pos,
      device=self.device,
      anchor_body=cfg.amp_anchor_body,
    )
    assert self.amp_motion_dataset.observation_dim == amp_dim, (
      f"AMP obs dim mismatch: env '{cfg.amp_obs_group}' group is {amp_dim}, "
      f"dataset emits {self.amp_motion_dataset.observation_dim}. The amp group and "
      f"the dataset terms must match element-for-element."
    )

    self.discriminator = AMPDiscriminator(
      input_dim=2 * amp_dim,
      amp_reward_coef=cfg.amp_reward_coef,
      hidden_layer_sizes=cfg.amp_discr_hidden_dims,
      device=self.device,
      task_reward_lerp=cfg.amp_task_reward_lerp,
    )
    self.amp_normalizer: EmpiricalNormalization | None = (
      EmpiricalNormalization(shape=amp_dim, device=self.device)
      if cfg.obs_normalization
      else None
    )
    self.amp_replay = AMPReplayBuffer(amp_dim, cfg.amp_replay_buffer_size, self.device)

    # Separate optimizer: the discriminator is a distinct learning signal from the
    # policy. Per-group weight decay mirrors beyondAMP (trunk light, head heavy).
    self.amp_optimizer = optim.Adam(
      [
        {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-3},
        {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 1e-2},
      ],
      lr=cfg.amp_learning_rate,
    )

    self._last_amp_reward = 0.0

  # ------------------------------------------------------------------ #
  # Seam 1: rollout reward blend
  # ------------------------------------------------------------------ #

  def _blend_amp_reward(
    self,
    prev_obs_dict: dict[str, torch.Tensor],
    obs_dict: dict[str, torch.Tensor],
    rewards: torch.Tensor,
    dones: torch.Tensor,
  ) -> torch.Tensor:
    assert isinstance(self.config, ResidualAmpPpoConfig)
    group = self.config.amp_obs_group
    amp_obs = prev_obs_dict[group]
    next_amp_obs = obs_dict[group]

    blended, _d, amp_reward = self.discriminator.predict_amp_reward(
      amp_obs, next_amp_obs, rewards, normalizer=self.amp_normalizer
    )

    # Store only transitions that don't straddle a reset: on a done step the
    # post-step amp obs is already the reset observation, so (s, s') is invalid.
    keep = dones < 0.5
    if keep.any():
      self.amp_replay.insert(amp_obs[keep], next_amp_obs[keep])

    self._last_amp_reward = float(amp_reward.mean().item())
    return blended

  # ------------------------------------------------------------------ #
  # Seam 2: discriminator update (after the inherited PPO update)
  # ------------------------------------------------------------------ #

  def _learning_step(self) -> dict[str, float]:
    loss_dict = super()._learning_step()
    loss_dict.update(self._update_discriminator())
    loss_dict["amp_reward"] = self._last_amp_reward
    return loss_dict

  def _update_discriminator(self) -> dict[str, float]:
    assert isinstance(self.config, ResidualAmpPpoConfig)
    cfg = self.config
    num_batches = cfg.num_learning_epochs * cfg.num_mini_batches
    mini_batch_size = self.env.num_envs * cfg.num_steps_per_env // cfg.num_mini_batches

    # Warmup: nothing to train on until the replay buffer can fill a minibatch.
    if self.amp_replay.num_samples < mini_batch_size:
      return {}

    policy_gen = self.amp_replay.feed_forward_generator(num_batches, mini_batch_size)
    expert_gen = self.amp_motion_dataset.feed_forward_generator(
      num_batches, mini_batch_size
    )

    tot_loss = tot_gp = tot_pol = tot_exp = 0.0
    n = 0
    for (ps, psp), (es, esp) in zip(policy_gen, expert_gen):
      if self.amp_normalizer is not None:
        with torch.no_grad():
          ps_n, psp_n = self.amp_normalizer(ps), self.amp_normalizer(psp)
          es_n, esp_n = self.amp_normalizer(es), self.amp_normalizer(esp)
      else:
        ps_n, psp_n, es_n, esp_n = ps, psp, es, esp

      policy_d = self.discriminator(torch.cat([ps_n, psp_n], dim=-1))
      expert_d = self.discriminator(torch.cat([es_n, esp_n], dim=-1))
      expert_loss = F.mse_loss(expert_d, torch.ones_like(expert_d))
      policy_loss = F.mse_loss(policy_d, -torch.ones_like(policy_d))
      amp_loss = 0.5 * (expert_loss + policy_loss)
      grad_pen = self.discriminator.compute_grad_pen(
        es_n, esp_n, lambda_=cfg.amp_grad_pen_lambda
      )

      self.amp_optimizer.zero_grad()
      (amp_loss + grad_pen).backward()
      self.amp_optimizer.step()

      # Update the running stats on RAW states (both policy and expert).
      if self.amp_normalizer is not None:
        self.amp_normalizer.update(ps)
        self.amp_normalizer.update(es)

      tot_loss += amp_loss.item()
      tot_gp += grad_pen.item()
      tot_pol += policy_d.mean().item()
      tot_exp += expert_d.mean().item()
      n += 1

    denom = max(n, 1)
    return {
      "amp_disc_loss": tot_loss / denom,
      "amp_grad_pen": tot_gp / denom,
      "amp_policy_pred": tot_pol / denom,
      "amp_expert_pred": tot_exp / denom,
    }

  # ------------------------------------------------------------------ #
  # Checkpoint (extend base with discriminator + amp normalizer)
  # ------------------------------------------------------------------ #

  def save(self, path: str | Path, **extra_state: Any) -> None:
    extra_state["amp_discriminator_state_dict"] = self.discriminator.state_dict()
    extra_state["amp_optimizer_state_dict"] = self.amp_optimizer.state_dict()
    if self.amp_normalizer is not None:
      extra_state["amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
    super().save(path, **extra_state)

  def load(self, path: str | Path) -> dict[str, Any]:
    result = super().load(path)
    checkpoint = self._load_checkpoint(path)
    self.discriminator.load_state_dict(checkpoint["amp_discriminator_state_dict"])
    self.amp_optimizer.load_state_dict(checkpoint["amp_optimizer_state_dict"])
    if self.amp_normalizer is not None and "amp_normalizer_state_dict" in checkpoint:
      self.amp_normalizer.load_state_dict(checkpoint["amp_normalizer_state_dict"])
    return result
