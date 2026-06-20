"""Proximal Policy Optimization (PPO) implementation.

On-policy actor-critic algorithm following RSL-RL patterns:
- Collect fixed-horizon rollout from all envs
- Compute GAE advantages
- Multiple epochs of mini-batch gradient updates with clipped surrogate loss
- Adaptive KL-based learning rate scheduling (RSL-RL)
- Single joint optimizer for actor+critic (RSL-RL)
- Store raw obs in buffer, normalize during learning (RSL-RL)

Integrates with BaseAlgorithm for checkpoint management, logging, and
episode tracking.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from mjlab.envs import ManagerBasedRlEnv

from colosseum.algorithm.base_algorithm import BaseAlgorithm, ObsType
from colosseum.algorithm.networks.ppo_networks import PpoActor, PpoValueNet
from colosseum.algorithm.utils.normalization import (
  EmpiricalNormalization,
  IdentityNormalizer,
)
from colosseum.algorithm.utils.rollout_buffer import RolloutBuffer
from colosseum.config.types.algorithm import PpoConfig, register_algorithm
from colosseum.mdp.symmetry import (
  TermMirrorSpec,
  augment_actions,
  augment_obs,
  build_symmetry_spec,
  mirror_obs,
)
from colosseum.utils.logger import extract_episode_metrics
from colosseum.utils.torch import get_obs_dims


@register_algorithm("ppo", config_class=PpoConfig)
class PPO(BaseAlgorithm):
  """On-policy PPO (RSL-RL/holosoma/CleanRL-inspired).

  Training loop per iteration:
  1. Collect num_steps_per_env steps from all envs -> rollout buffer
  2. Compute GAE returns and advantages
  3. For num_learning_epochs, shuffle and iterate mini-batches:
     a. Re-evaluate actor log_prob and entropy for stored (obs, action)
     b. Compute KL divergence -> adapt learning rate (if desired_kl set)
     c. Compute surrogate clip loss + value clip loss + entropy bonus
     d. Backward + grad clip + step (separate actor/critic optimizers)
  4. Clear rollout buffer
  5. Log metrics
  6. Maybe save checkpoint
  """

  def __init__(
    self,
    config: PpoConfig,
    env: ManagerBasedRlEnv,
    device: str | torch.device,
    log_fn: Callable[[dict[str, float], int], None],
    log_interval: int,
  ) -> None:
    device = (
      device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    super().__init__(
      config=config,
      env=env,
      device=device,
      log_fn=log_fn,
      log_interval=log_interval,
    )

    self._build_networks()
    self._synchronize_model_weights(self.actor, self.value_net)

    # Separate adaptive LRs for actor and critic (holosoma style)
    # Must be set before _build_optimizers which uses them.
    self.actor_learning_rate = float(config.actor_learning_rate)
    self.critic_learning_rate = float(config.critic_learning_rate)
    self.max_actor_learning_rate = (
      config.max_actor_learning_rate
      if config.max_actor_learning_rate is not None
      else max(self.actor_learning_rate, 1e-2)
    )
    self.min_actor_learning_rate = (
      config.min_actor_learning_rate
      if config.min_actor_learning_rate is not None
      else min(self.actor_learning_rate, 1e-5)
    )
    self.max_critic_learning_rate = (
      config.max_critic_learning_rate
      if config.max_critic_learning_rate is not None
      else max(self.critic_learning_rate, 1e-2)
    )
    self.min_critic_learning_rate = (
      config.min_critic_learning_rate
      if config.min_critic_learning_rate is not None
      else min(self.critic_learning_rate, 1e-5)
    )

    self._build_optimizers()
    self._build_rollout_buffer()
    self._build_normalizer()

    # Symmetry (holosoma-style left-right mirror equivariance)
    self._setup_symmetry()

    self.episode_length_buf = torch.zeros(self.env.num_envs, device=self.device)

    # Rolling mean episode return (RSL-RL style): accumulate per-env reward each step,
    # push the episode total to a deque when the episode ends.
    self.rewbuffer: deque[float] = deque(maxlen=50)
    self.cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)

  def _build_networks(self) -> None:
    assert isinstance(self.config, PpoConfig)

    self.obs_dim = get_obs_dims(self.env)
    self.actor_obs_dim = self.obs_dim["actor"]
    self.critic_obs_dim = self.obs_dim["critic"]
    self.action_dim = int(np.prod(self.env.single_action_space.shape))

    self.actor: PpoActor = PpoActor(
      self.actor_obs_dim,
      self.action_dim,
      self.config.actor,
    ).to(self.device)

    self.value_net = PpoValueNet(
      self.critic_obs_dim,
      self.config.critic,
    ).to(self.device)

  def _build_optimizers(self) -> None:
    """Separate AdamW optimizers for actor and critic (holosoma style)."""
    assert isinstance(self.config, PpoConfig)

    self.actor_optimizer = optim.AdamW(
      self.actor.parameters(),
      lr=self.actor_learning_rate,
      weight_decay=self.config.weight_decay,
    )
    self.critic_optimizer = optim.AdamW(
      self.value_net.parameters(),
      lr=self.critic_learning_rate,
      weight_decay=self.config.weight_decay,
    )

  def _build_rollout_buffer(self) -> None:
    assert isinstance(self.config, PpoConfig)
    self.rollout_buffer = RolloutBuffer(
      num_envs=self.env.num_envs,
      num_steps=self.config.num_steps_per_env,
      actor_obs_dim=self.actor_obs_dim,
      critic_obs_dim=self.critic_obs_dim,
      action_dim=self.action_dim,
      device=self.device,
    )

  def _setup_symmetry(self) -> None:
    """Build the actor/critic mirror specs (single "actor" group).

    Overridden by ResidualPPO, which has multiple per-skill actor groups
    instead of one "actor" group.
    """
    assert isinstance(self.config, PpoConfig)
    self._actor_sym_spec: list[TermMirrorSpec] | None = None
    self._critic_sym_spec: list[TermMirrorSpec] | None = None
    self._action_mirror_fn = None
    self._use_symmetry = False

    if (
      self.config.symmetry_loss_coef > 0.0
      or self.config.symmetry_critic_coef > 0.0
      or self.config.symmetry_data_augmentation
    ):
      self._use_symmetry = True
      obs_manager = self.env.observation_manager
      self._actor_sym_spec = build_symmetry_spec(obs_manager, "actor")
      self._critic_sym_spec = build_symmetry_spec(obs_manager, "critic")
      if "actions" in obs_manager.active_terms.get("actor", []):
        actions_cfg = obs_manager.get_term_cfg("actor", "actions")
        self._action_mirror_fn = getattr(actions_cfg, "mirror_fn", None)

  def _build_normalizer(self) -> None:
    assert isinstance(self.config, PpoConfig)

    if not self.config.obs_normalization:
      self.actor_obs_normalizer = IdentityNormalizer()
      self.critic_obs_normalizer = IdentityNormalizer()
    else:
      self.actor_obs_normalizer = EmpiricalNormalization(
        shape=self.actor_obs_dim,
        device=self.device,
      )
      self.critic_obs_normalizer = EmpiricalNormalization(
        shape=self.critic_obs_dim,
        device=self.device,
      )

  def train(self) -> None:
    """Main PPO training loop."""
    self._ppo_loop()
    self.env.close()

  def _ppo_loop(
    self, title: str = "PPO Training", total_steps: int | None = None
  ) -> None:
    """Inner PPO training loop without env.close(). Called by train() and pipeline scripts."""
    assert isinstance(self.config, PpoConfig)

    self.start_time = time.time()

    losses_buffer: dict[str, list[float]] = defaultdict(list)
    collection_time_sum = 0.0
    learning_time_sum = 0.0

    # Set networks to training mode
    self.actor.train()
    self.value_net.train()

    total_timesteps = (
      total_steps if total_steps is not None else self.config.learning_steps
    )
    steps_per_iter = self.config.num_steps_per_env * self.env.num_envs
    num_iterations = total_timesteps // steps_per_iter
    display_total = self.global_step + total_timesteps

    log_interval_iters = max(1, self.log_interval // steps_per_iter)

    logger.info("=" * 80)
    logger.info(f"Starting {title}")
    logger.info(f"Step range:   {self.global_step} → {display_total}")
    logger.info(f"Additional:   {total_timesteps}")
    logger.info(f"Steps per iteration: {steps_per_iter}")
    logger.info(f"Num iterations: {num_iterations}")
    logger.info(
      f"Log interval: {self.log_interval} env steps ({log_interval_iters} iterations)"
    )
    logger.info("=" * 80)

    obs_dict, _ = self.env.reset(seed=self.seed)
    current_actor_obs = self.get_actor_obs(obs_dict)
    current_critic_obs = self.get_critic_obs(obs_dict)
    current_dones = torch.zeros(self.env.num_envs, device=self.device)

    # Pre-warm normalizer with initial observations so stats aren't (0,1)
    # during the first rollout. Without this, the normalizer shifts
    # dramatically during the first rollout, inflating KL artificially
    # and crashing the adaptive learning rate.
    if self.config.obs_normalization:
      prewarm_actor_obs = self._prewarm_actor_obs(current_actor_obs)
      self.actor_obs_normalizer.update(prewarm_actor_obs)
      self.critic_obs_normalizer.update(current_critic_obs)

    # Randomize initial episode lengths (RSL-RL pattern) so environments
    # reset at different times, providing more diverse early data
    if hasattr(self.env, "episode_length_buf"):
      max_ep_len = int(self.env.max_episode_length)
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=max_ep_len
      )

    for iteration in range(1, num_iterations + 1):
      # ============================================
      # COLLECTION PHASE
      # ============================================
      start_collect = time.perf_counter()

      current_actor_obs, current_critic_obs, current_dones, obs_dict = (
        self._collect_rollout(
          current_actor_obs, current_critic_obs, current_dones, obs_dict
        )
      )

      collection_time = time.perf_counter() - start_collect
      collection_time_sum += collection_time

      # ============================================
      # LEARNING PHASE
      # ============================================
      start_learn = time.perf_counter()

      loss_dict = self._learning_step()
      for k, v in loss_dict.items():
        losses_buffer[k].append(v)

      learning_time = time.perf_counter() - start_learn
      learning_time_sum += learning_time

      self.global_step += steps_per_iter
      self._maybe_save_checkpoint(self.global_step)
      self._maybe_evaluate(self.global_step)

      # ============================================
      # LOGGING PHASE
      # ============================================
      if iteration % log_interval_iters == 0:
        # Rolling mean episode return (RSL-RL style)
        if len(self.rewbuffer) > 0:
          losses_buffer["mean_reward"].append(statistics.mean(self.rewbuffer))

        self._log_training_metrics(
          step=self.global_step,
          losses_buffer=losses_buffer,
          collection_time=collection_time_sum,
          learning_time=learning_time_sum,
          log_interval=log_interval_iters,
          total_timesteps=display_total,
          title=title,
          use_rich=self.config.use_rich_logging,
          steps_per_log_step=self.config.num_steps_per_env,
        )
        losses_buffer.clear()
        collection_time_sum = 0.0
        learning_time_sum = 0.0

    # Flush remaining metrics at end of loop (handles short runs where the
    # log interval exceeds the total step count).
    if losses_buffer:
      if len(self.rewbuffer) > 0:
        losses_buffer["mean_reward"].append(statistics.mean(self.rewbuffer))
      self._log_training_metrics(
        step=self.global_step,
        losses_buffer=losses_buffer,
        collection_time=collection_time_sum,
        learning_time=learning_time_sum,
        log_interval=num_iterations,
        total_timesteps=display_total,
        title=title,
        use_rich=self.config.use_rich_logging,
        steps_per_log_step=self.config.num_steps_per_env,
      )

  def _blend_amp_reward(
    self,
    prev_obs_dict: ObsType,
    obs_dict: ObsType,
    rewards: torch.Tensor,
    dones: torch.Tensor,
  ) -> torch.Tensor:
    """Hook to fold an AMP style reward into the per-step reward (no-op here)."""
    return rewards

  def _collect_rollout(
    self,
    current_actor_obs: torch.Tensor,
    current_critic_obs: torch.Tensor,
    current_dones: torch.Tensor,
    obs_dict: ObsType,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ObsType]:
    """Collect num_steps_per_env steps and fill rollout buffer."""
    assert isinstance(self.config, PpoConfig)

    self.rollout_buffer.clear()

    with torch.no_grad():
      for _step in range(self.config.num_steps_per_env):
        norm_actor_obs = self.actor_obs_normalizer(current_actor_obs)
        norm_critic_obs = self.critic_obs_normalizer(current_critic_obs)

        # Get action, log_prob, value, distribution params (single forward pass)
        actions, log_probs, action_means, action_stds = self.actor.act_with_log_prob(
          norm_actor_obs
        )
        values = self.value_net(norm_critic_obs)

        # Step environment (action clipping handled by vecenv_wrapper)
        prev_obs_dict = obs_dict
        obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
        if terminated.is_floating_point():
          dones = torch.clamp(terminated + truncated.float(), 0.0, 1.0)
        else:
          dones = (terminated | truncated).float()

        # Extract next observations
        next_actor_obs = self.get_actor_obs(obs_dict)
        next_critic_obs = self.get_critic_obs(obs_dict)

        # Update normalizer AFTER env.step on new obs (RSL-RL pattern)
        if self.config.obs_normalization:
          self.actor_obs_normalizer.update(next_actor_obs)
          self.critic_obs_normalizer.update(next_critic_obs)

        # Accumulate RAW rewards for episode-return logging (RSL-RL pattern):
        # logger.process_env_step in RSL-RL receives the original rewards
        # BEFORE bootstrapping inflates them at truncation boundaries.
        self.cur_reward_sum += rewards

        # AMP hook: blend a discriminator style reward into `rewards` and stash
        # policy transitions. No-op in base PPO; overridden by AmpPPO. Placed
        # AFTER cur_reward_sum (so episodic logging stays task-only) and BEFORE
        # timeout bootstrapping (so the bootstrap applies on the blend).
        rewards = self._blend_amp_reward(prev_obs_dict, obs_dict, rewards, dones)

        # Timeout bootstrapping (RSL-RL/holosoma pattern)
        # For infinite-horizon tasks, bootstrap value at truncation.
        # Done AFTER logging accumulation so rewbuffer sees true rewards.
        if not getattr(self.env.cfg, "is_finite_horizon", True):
          truncated_mask = truncated.float()
          if truncated_mask.any():
            norm_next_critic = self.critic_obs_normalizer(next_critic_obs)
            truncated_values = self.value_net(norm_next_critic).squeeze(-1)
            rewards = rewards + self.config.gamma * truncated_values * truncated_mask
        self.episode_length_buf += 1
        episode_done_ids = (dones >= 1.0).nonzero(as_tuple=False).squeeze(-1)
        if len(episode_done_ids) > 0:
          self.rewbuffer.extend(
            self.cur_reward_sum[episode_done_ids].cpu().numpy().tolist()
          )
          self.cur_reward_sum[episode_done_ids] = 0.0
          self.episode_lengths.extend(
            self.episode_length_buf[episode_done_ids].cpu().numpy().tolist()
          )
          self.episode_length_buf[episode_done_ids] = 0

        # Update episode tracking
        hard_terminated = (
          terminated >= 1.0 if terminated.is_floating_point() else terminated
        )
        self.update_episode_counts(hard_terminated, truncated)

        # Only update episode metrics when episodes actually ended
        if "log" in infos and (dones >= 1.0).any():
          self.latest_episode_metrics = extract_episode_metrics(infos["log"])

        # Store RAW observations in buffer (RSL-RL pattern)
        # Normalization happens during learning, not storage.
        self.rollout_buffer.add(
          actor_obs=current_actor_obs,
          critic_obs=current_critic_obs,
          actions=actions,
          rewards=rewards,
          dones=dones,
          values=values,
          log_probs=log_probs,
          action_means=action_means,
          action_stds=action_stds,
        )

        # Advance
        current_actor_obs = next_actor_obs
        current_critic_obs = next_critic_obs
        current_dones = dones

      # Bootstrap value for last observation
      norm_last_critic = self.critic_obs_normalizer(current_critic_obs)
      last_values = self.value_net(norm_last_critic)

    # Compute GAE (without normalization in the buffer)
    self.rollout_buffer.compute_returns_and_advantages(
      last_values=last_values,
      gamma=self.config.gamma,
      lam=self.config.lam,
      normalize_advantage=False,
    )

    # Multi-GPU advantage normalization (holosoma style).
    # Global mean/std across all GPUs ensures consistent advantage scaling.
    if not self.config.normalize_advantage_per_mini_batch:
      self.rollout_buffer.advantages = self._normalize_advantages_multi_gpu(
        self.rollout_buffer.advantages
      )

    return current_actor_obs, current_critic_obs, current_dones, obs_dict

  def _learning_step(self) -> dict[str, float]:
    """Run PPO update epochs over mini-batches. Returns averaged losses."""
    assert isinstance(self.config, PpoConfig)

    total_surrogate_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_symmetry_actor_loss = 0.0
    total_symmetry_critic_loss = 0.0
    num_updates = 0

    generator = self.rollout_buffer.mini_batch_generator(
      num_mini_batches=self.config.num_mini_batches,
      num_epochs=self.config.num_learning_epochs,
      normalize_advantage_per_mini_batch=self.config.normalize_advantage_per_mini_batch,
    )

    for batch in generator:
      # Get RAW observations from buffer
      actor_obs_raw = batch["actor_obs"]
      critic_obs_raw = batch["critic_obs"]
      actions = batch["actions"]
      returns = batch["returns"]
      advantages = batch["advantages"]
      old_log_probs = batch["old_log_probs"].squeeze(-1)
      old_action_means = batch["old_action_means"]
      old_action_stds = batch["old_action_stds"]
      target_values = batch["values"]

      actor_obs_norm = self.actor_obs_normalizer(actor_obs_raw)
      critic_obs = self.critic_obs_normalizer(critic_obs_raw)

      # --- Symmetry data augmentation (holosoma style) ---
      # Augment BEFORE compose so only proprio is mirrored, not any appended
      # encoder latents (subclasses like RmaPPO append latents in _compose_actor_input).
      original_batch_size = actor_obs_norm.shape[0]
      privileged_obs = batch.get("privileged_obs", {})
      adaptation_obs = batch.get("adaptation_obs", {})
      if (
        self._use_symmetry
        and self.config.symmetry_data_augmentation
        and self._action_mirror_fn is not None
      ):
        actor_obs_norm = augment_obs(actor_obs_norm, self._actor_sym_spec)
        actions = augment_actions(actions, self._action_mirror_fn)
        critic_obs = augment_obs(critic_obs, self._critic_sym_spec)
        old_log_probs = old_log_probs.repeat(2)
        target_values = target_values.repeat(2, 1)
        advantages = advantages.repeat(2, 1)
        returns = returns.repeat(2, 1)
        old_action_means = old_action_means.repeat(2, 1)
        old_action_stds = old_action_stds.repeat(2, 1)
        # Repeat privileged obs so _compose_actor_input sees a consistent batch.
        # Latent z is left-right symmetric (physics scalars don't flip under mirroring).
        privileged_obs = {
          k: v.repeat(2, *([1] * (v.dim() - 1))) for k, v in privileged_obs.items()
        }
        adaptation_obs = {
          k: v.repeat(2, *([1] * (v.dim() - 1))) for k, v in adaptation_obs.items()
        }

      # Compose actor input (identity in PPO; RmaPPO overrides to append encoder latents)
      actor_obs = self._compose_actor_input(
        actor_obs_norm, privileged_obs, adaptation_obs
      )

      # Re-evaluate actions with current policy
      new_log_probs, entropy_all = self.actor.evaluate(actor_obs, actions)
      new_values = self.value_net(critic_obs)

      # Entropy: only from original batch (holosoma convention)
      if (
        self._use_symmetry
        and self.config.symmetry_data_augmentation
        and self._action_mirror_fn is not None
      ):
        entropy = entropy_all[:original_batch_size]
      else:
        entropy = entropy_all

      # --- KL divergence (RSL-RL analytical formula) ---
      # Computed on original batch only, even when augmented (holosoma convention).
      with torch.inference_mode():
        mu_batch = self.actor.forward(actor_obs)
        _old_means = old_action_means
        _old_stds = old_action_stds
        if (
          self._use_symmetry
          and self.config.symmetry_data_augmentation
          and self._action_mirror_fn is not None
        ):
          mu_batch = mu_batch[:original_batch_size]
          _old_means = _old_means[:original_batch_size]
          _old_stds = _old_stds[:original_batch_size]
        sigma_batch = torch.clamp(
          self.actor.std, min=self.actor.min_noise_std
        ).expand_as(mu_batch)
        kl = torch.sum(
          torch.log(sigma_batch / _old_stds + 1e-5)
          + (_old_stds.pow(2) + (_old_means - mu_batch).pow(2))
          / (2.0 * sigma_batch.pow(2))
          - 0.5,
          dim=-1,
        )
        kl_mean = self._distributed_mean_scalar(float(kl.mean().item()))

      # Adaptive KL LR scheduling (holosoma pattern: separate actor/critic LRs)
      if self.config.schedule == "adaptive" and self.config.desired_kl is not None:
        if kl_mean > self.config.desired_kl * 2.0:
          self.actor_learning_rate = max(
            self.min_actor_learning_rate, self.actor_learning_rate / 1.5
          )
          self.critic_learning_rate = max(
            self.min_critic_learning_rate, self.critic_learning_rate / 1.5
          )
        elif kl_mean < self.config.desired_kl / 2.0 and kl_mean > 0.0:
          self.actor_learning_rate = min(
            self.max_actor_learning_rate, self.actor_learning_rate * 1.5
          )
          self.critic_learning_rate = min(
            self.max_critic_learning_rate, self.critic_learning_rate * 1.5
          )
        for g in self.actor_optimizer.param_groups:
          g["lr"] = self.actor_learning_rate
        for g in self.critic_optimizer.param_groups:
          g["lr"] = self.critic_learning_rate

      # --- Surrogate loss (PPO-clip) ---
      advantages_squeezed = advantages.squeeze(-1)
      ratio = torch.exp(new_log_probs - old_log_probs)
      surrogate = -advantages_squeezed * ratio
      surrogate_clipped = -advantages_squeezed * torch.clamp(
        ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      # --- Value loss ---
      if self.config.use_clipped_value_loss:
        value_clipped = target_values + torch.clamp(
          new_values - target_values,
          -self.config.clip_param,
          self.config.clip_param,
        )
        value_loss_unclipped = (new_values - returns).pow(2)
        value_loss_clipped = (value_clipped - returns).pow(2)
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
      else:
        value_loss = (returns - new_values).pow(2).mean()

      # --- Symmetry loss (holosoma style) ---
      symmetry_actor_loss = torch.tensor(0.0, device=self.device)
      symmetry_critic_loss = torch.tensor(0.0, device=self.device)
      if self._use_symmetry and self._action_mirror_fn is not None:
        if self.config.symmetry_loss_coef > 0.0:
          if self.config.symmetry_data_augmentation:
            mu_full = self.actor.forward(actor_obs.detach())
            mu_original = mu_full[:original_batch_size]
            mu_mirrored = mu_full[original_batch_size:]
            symmetry_actor_loss = torch.nn.functional.mse_loss(
              mu_mirrored, self._action_mirror_fn(mu_original)
            )
          else:
            mu_original = self.actor.forward(actor_obs.detach())
            mirrored_actor_obs = self._mirror_actor_input(
              actor_obs_norm.detach(), privileged_obs, adaptation_obs
            )
            mu_mirrored = self.actor.forward(mirrored_actor_obs)
            symmetry_actor_loss = torch.nn.functional.mse_loss(
              mu_mirrored, self._action_mirror_fn(mu_original)
            )

        if self.config.symmetry_critic_coef > 0.0:
          if self.config.symmetry_data_augmentation:
            val_original = new_values[:original_batch_size]
            val_mirrored = new_values[original_batch_size:]
            symmetry_critic_loss = torch.nn.functional.mse_loss(
              val_original, val_mirrored
            )
          else:
            mirrored_critic_obs = mirror_obs(critic_obs.detach(), self._critic_sym_spec)
            val_mirrored = self.value_net(mirrored_critic_obs)
            symmetry_critic_loss = torch.nn.functional.mse_loss(
              new_values, val_mirrored
            )

      # --- Total loss (single combined, RSL-RL style) ---
      loss = (
        surrogate_loss
        + self.config.value_loss_coef * value_loss
        - self.config.entropy_coef * entropy.mean()
        + self.config.symmetry_loss_coef * symmetry_actor_loss
        + self.config.symmetry_critic_coef * symmetry_critic_loss
      )

      # --- Gradient step (separate optimizers, holosoma style) ---
      self.actor_optimizer.zero_grad()
      self.critic_optimizer.zero_grad()

      # Guard: NaN/Inf loss (from NaN rewards/advantages) would corrupt params.
      # Skip this minibatch instead of propagating NaN through the network.
      if not torch.isfinite(loss):
        continue

      loss.backward()
      self._distributed_average_optimizer_grads(self.actor_optimizer)
      self._distributed_average_optimizer_grads(self.critic_optimizer)
      # Second-layer guard: sanitize any residual NaN gradients before clipping.
      for p in self.actor.parameters():
        if p.grad is not None:
          p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
      for p in self.value_net.parameters():
        if p.grad is not None:
          p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
      torch.nn.utils.clip_grad_norm_(
        self.actor.parameters(), max_norm=self.config.max_grad_norm
      )
      torch.nn.utils.clip_grad_norm_(
        self.value_net.parameters(), max_norm=self.config.max_grad_norm
      )
      self.actor_optimizer.step()
      self.critic_optimizer.step()

      # Accumulate for logging
      total_surrogate_loss += surrogate_loss.item()
      total_value_loss += value_loss.item()
      total_entropy += entropy.mean().item()
      total_kl += kl_mean
      total_symmetry_actor_loss += symmetry_actor_loss.item()
      total_symmetry_critic_loss += symmetry_critic_loss.item()
      num_updates += 1

    if self.is_distributed:
      totals = self._distributed_sum_vector(
        [
          total_surrogate_loss,
          total_value_loss,
          total_entropy,
          total_kl,
          total_symmetry_actor_loss,
          total_symmetry_critic_loss,
          float(num_updates),
        ]
      )
      (
        total_surrogate_loss,
        total_value_loss,
        total_entropy,
        total_kl,
        total_symmetry_actor_loss,
        total_symmetry_critic_loss,
      ) = totals[:6]
      num_updates = int(totals[6])

    self.rollout_buffer.clear()

    loss_dict: dict[str, float] = {
      "surrogate_loss": total_surrogate_loss / max(num_updates, 1),
      "value_loss": total_value_loss / max(num_updates, 1),
      "entropy": total_entropy / max(num_updates, 1),
      "kl": total_kl / max(num_updates, 1),
      "actor_learning_rate": self.actor_learning_rate,
      "critic_learning_rate": self.critic_learning_rate,
    }
    if self._use_symmetry:
      loss_dict["symmetry_actor_loss"] = total_symmetry_actor_loss / max(num_updates, 1)
      loss_dict["symmetry_critic_loss"] = total_symmetry_critic_loss / max(
        num_updates, 1
      )
    return loss_dict

  def _compose_actor_input(
    self,
    actor_obs: torch.Tensor,
    privileged_obs: dict[str, torch.Tensor],
    adaptation_obs: dict[str, torch.Tensor] | None = None,
  ) -> torch.Tensor:
    """Build the full actor input. Identity in PPO; RmaPPO overrides to append encoder latents."""
    return actor_obs

  def _mirror_actor_input(
    self,
    actor_obs_norm: torch.Tensor,
    privileged_obs: dict[str, torch.Tensor],
    adaptation_obs: dict[str, torch.Tensor] | None = None,
  ) -> torch.Tensor:
    """Mirror actor_obs_norm then recompose. Used by the non-augmentation symmetry loss path."""
    mirrored_norm = mirror_obs(actor_obs_norm, self._actor_sym_spec)
    return self._compose_actor_input(mirrored_norm, privileged_obs, adaptation_obs)

  def _prewarm_actor_obs(self, actor_obs: torch.Tensor) -> torch.Tensor:
    """Transform actor obs for normalizer pre-warming. Override in subclasses."""
    return actor_obs

  def _eval_get_action(self, normalized_obs: torch.Tensor) -> torch.Tensor:
    """Deterministic action for PPO evaluation."""
    return self.actor.act_inference(normalized_obs)

  def save(self, path: str | Path, **extra_state: Any) -> None:
    """Save PPO checkpoint to disk."""
    if "global_step" not in extra_state:
      raise ValueError("global_step must be provided in extra_state")

    state_dict = {
      "actor_state_dict": self.actor.state_dict(),
      "value_net_state_dict": self.value_net.state_dict(),
      "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
      "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
      "actor_obs_normalizer_state_dict": self.actor_obs_normalizer.state_dict(),
      "critic_obs_normalizer_state_dict": self.critic_obs_normalizer.state_dict(),
      "global_step": extra_state["global_step"],
      "actor_learning_rate": self.actor_learning_rate,
      "critic_learning_rate": self.critic_learning_rate,
      "config": self.config,
    }

    for key, value in extra_state.items():
      if key not in state_dict:
        state_dict[key] = value

    self._save_checkpoint(path, state_dict)

  def load(self, path: str | Path) -> dict[str, Any]:
    """Load PPO checkpoint from disk."""
    checkpoint = self._load_checkpoint(path)

    self.actor.load_state_dict(checkpoint["actor_state_dict"])
    self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
    self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    self.actor_obs_normalizer.load_state_dict(
      checkpoint["actor_obs_normalizer_state_dict"]
    )
    self.critic_obs_normalizer.load_state_dict(
      checkpoint["critic_obs_normalizer_state_dict"]
    )

    self.global_step = checkpoint["global_step"]
    self.actor_learning_rate = checkpoint.get(
      "actor_learning_rate", self.actor_learning_rate
    )
    self.critic_learning_rate = checkpoint.get(
      "critic_learning_rate", self.critic_learning_rate
    )
    self._restore_env_step_counter()

    return {
      "global_step": checkpoint["global_step"],
      "metadata": checkpoint.get("metadata", {}),
      "config": checkpoint.get("config"),
    }
