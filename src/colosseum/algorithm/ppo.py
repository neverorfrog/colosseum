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

from colosseum.algorithm.base_algorithm import BaseAlgorithm
from colosseum.algorithm.normalization import EmpiricalNormalization, IdentityNormalizer
from colosseum.algorithm.ppo_networks import PpoActor, PpoValueNet
from colosseum.algorithm.rollout_buffer import RolloutBuffer
from colosseum.config.types.algorithm import PpoConfig, register_algorithm
from colosseum.utils.logger import extract_episode_metrics
from colosseum.utils.torch import get_obs_dims
from colosseum.algorithm.base_algorithm import ObsType


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
    self._build_optimizers()
    self._build_rollout_buffer()
    self._build_normalizer()

    self.episode_length_buf = torch.zeros(self.env.num_envs, device=self.device)

    # Adaptive LR state (single LR for joint optimizer, RSL-RL style)
    self.learning_rate = float(config.learning_rate)

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
    """Single joint optimizer for actor+critic (RSL-RL style)."""
    assert isinstance(self.config, PpoConfig)
    from itertools import chain

    self.optimizer = optim.Adam(
      chain(self.actor.parameters(), self.value_net.parameters()),
      lr=self.config.learning_rate,
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
    assert isinstance(self.config, PpoConfig)

    self.start_time = time.time()

    losses_buffer: dict[str, list[float]] = defaultdict(list)
    collection_time_sum = 0.0
    learning_time_sum = 0.0

    # Set networks to training mode
    self.actor.train()
    self.value_net.train()

    total_timesteps = self.config.learning_steps
    steps_per_iter = self.config.num_steps_per_env * self.env.num_envs
    num_iterations = total_timesteps // steps_per_iter

    # Convert log_interval from env steps → PPO iterations (same unit as SAC's step counter)
    # This makes the W&B x-axis (global_step = total env transitions) consistent with SAC.
    log_interval_iters = max(1, self.log_interval // steps_per_iter)

    logger.info("=" * 80)
    logger.info("Starting PPO training")
    logger.info(f"Total steps: {total_timesteps}")
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

      current_actor_obs, current_critic_obs, current_dones, obs_dict = self._collect_rollout(
        current_actor_obs, current_critic_obs, current_dones, obs_dict
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
      self._maybe_evaluate(self.global_step)

      # ============================================
      # LOGGING PHASE
      # ============================================
      if iteration % log_interval_iters == 0:
        # Save checkpoint at log intervals (not every iteration)
        self._maybe_save_checkpoint(self.global_step)
        # Rolling mean episode return (RSL-RL style)
        if len(self.rewbuffer) > 0:
          losses_buffer["mean_reward"].append(statistics.mean(self.rewbuffer))

        self._log_training_metrics(
          step=self.global_step,
          losses_buffer=losses_buffer,
          collection_time=collection_time_sum,
          learning_time=learning_time_sum,
          log_interval=log_interval_iters,
          total_timesteps=total_timesteps,
          title="PPO Training",
          use_rich=self.config.use_rich_logging,
          steps_per_log_step=self.config.num_steps_per_env,
        )
        losses_buffer.clear()
        collection_time_sum = 0.0
        learning_time_sum = 0.0

    self.env.close()

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
        obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
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
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
          self.rewbuffer.extend(self.cur_reward_sum[done_ids].cpu().numpy().tolist())
          self.cur_reward_sum[done_ids] = 0.0
          self.episode_lengths.extend(
            self.episode_length_buf[done_ids].cpu().numpy().tolist()
          )
          self.episode_length_buf[done_ids] = 0

        # Update episode tracking
        self.update_episode_counts(terminated, truncated)

        # Only update episode metrics when episodes actually ended
        if "log" in infos and dones.any():
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

    # Compute GAE
    normalize_globally = not self.config.normalize_advantage_per_mini_batch
    self.rollout_buffer.compute_returns_and_advantages(
      last_values=last_values,
      gamma=self.config.gamma,
      lam=self.config.lam,
      normalize_advantage=normalize_globally,
    )

    return current_actor_obs, current_critic_obs, current_dones, obs_dict

  def _learning_step(self) -> dict[str, float]:
    """Run PPO update epochs over mini-batches. Returns averaged losses."""
    assert isinstance(self.config, PpoConfig)

    total_surrogate_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
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

      # Compose actor input (identity in PPO; RmaPPO overrides to append encoder latents)
      privileged_obs = batch.get("privileged_obs", {})
      actor_obs = self._compose_actor_input(actor_obs_norm, privileged_obs)

      # Re-evaluate actions with current policy
      new_log_probs, entropy = self.actor.evaluate(actor_obs, actions)
      new_values = self.value_net(critic_obs)

      # --- KL divergence (RSL-RL analytical formula) ---
      with torch.inference_mode():
        mu_batch = self.actor.forward(actor_obs)
        sigma_batch = torch.clamp(
          self.actor.std, min=self.actor.min_noise_std
        ).expand_as(mu_batch)
        kl = torch.sum(
          torch.log(sigma_batch / old_action_stds + 1e-5)
          + (old_action_stds.pow(2) + (old_action_means - mu_batch).pow(2))
          / (2.0 * sigma_batch.pow(2))
          - 0.5,
          dim=-1,
        )
        kl_mean = self._distributed_mean_scalar(float(kl.mean().item()))

      # Adaptive KL LR scheduling (RSL-RL pattern: single LR)
      if self.config.schedule == "adaptive" and self.config.desired_kl is not None:
        if kl_mean > self.config.desired_kl * 2.0:
          self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < self.config.desired_kl / 2.0 and kl_mean > 0.0:
          self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for g in self.optimizer.param_groups:
          g["lr"] = self.learning_rate

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

      # --- Total loss (single combined, RSL-RL style) ---
      loss = (
        surrogate_loss
        + self.config.value_loss_coef * value_loss
        - self.config.entropy_coef * entropy.mean()
      )

      # --- Gradient step (single optimizer, RSL-RL style) ---
      self.optimizer.zero_grad()
      loss.backward()
      self._distributed_average_optimizer_grads(self.optimizer)
      torch.nn.utils.clip_grad_norm_(
        self.actor.parameters(), max_norm=self.config.max_grad_norm
      )
      torch.nn.utils.clip_grad_norm_(
        self.value_net.parameters(), max_norm=self.config.max_grad_norm
      )
      self.optimizer.step()

      # Accumulate for logging
      total_surrogate_loss += surrogate_loss.item()
      total_value_loss += value_loss.item()
      total_entropy += entropy.mean().item()
      total_kl += kl_mean
      num_updates += 1

    if self.is_distributed:
      totals = self._distributed_sum_vector(
        [
          total_surrogate_loss,
          total_value_loss,
          total_entropy,
          total_kl,
          float(num_updates),
        ]
      )
      total_surrogate_loss, total_value_loss, total_entropy, total_kl = totals[:4]
      num_updates = int(totals[4])

    self.rollout_buffer.clear()

    return {
      "surrogate_loss": total_surrogate_loss / max(num_updates, 1),
      "value_loss": total_value_loss / max(num_updates, 1),
      "entropy": total_entropy / max(num_updates, 1),
      "kl": total_kl / max(num_updates, 1),
      "learning_rate": self.learning_rate,
    }

  def _compose_actor_input(
    self,
    actor_obs: torch.Tensor,
    privileged_obs: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    """Build the full actor input. Identity in PPO; RmaPPO overrides to append encoder latents."""
    return actor_obs

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
      "optimizer_state_dict": self.optimizer.state_dict(),
      "actor_obs_normalizer_state_dict": self.actor_obs_normalizer.state_dict(),
      "critic_obs_normalizer_state_dict": self.critic_obs_normalizer.state_dict(),
      "global_step": extra_state["global_step"],
      "learning_rate": self.learning_rate,
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
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.actor_obs_normalizer.load_state_dict(
      checkpoint["actor_obs_normalizer_state_dict"]
    )
    self.critic_obs_normalizer.load_state_dict(
      checkpoint["critic_obs_normalizer_state_dict"]
    )

    self.global_step = checkpoint["global_step"]
    self.learning_rate = checkpoint.get("learning_rate", self.learning_rate)
    self._restore_env_step_counter()

    return {
      "global_step": checkpoint["global_step"],
      "metadata": checkpoint.get("metadata", {}),
      "config": checkpoint.get("config"),
    }
