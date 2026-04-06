"""DAgger-PPO: PPO with teacher-student imitation loss.

Extends PPO with a frozen pre-trained teacher policy. At each rollout step the
teacher's deterministic actions are stored alongside the standard transition
data. During the learning phase an imitation loss term is added to the PPO
objective:

    L = L_PPO + λ · MSE(student_mean, teacher_action)

λ is annealed linearly from imitation_coef → 0 over imitation_annealing_steps
global steps, so the policy eventually acts on pure RL signal.

The teacher's observation is a contiguous suffix of the student's observation:

    student_obs[:, teacher_obs_start_idx:] == teacher_obs

For the maze → velocity transfer this is indices [7:] (78-dim velocity obs).

Reference: DAgger (Ross et al., 2011) + teacher-student RL (arxiv:2512.06571).
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from mjlab.envs import ManagerBasedRlEnv

from colosseum.algorithm.base_algorithm import BaseAlgorithm
from colosseum.algorithm.normalization import EmpiricalNormalization
from colosseum.algorithm.ppo import PPO
from colosseum.algorithm.ppo_networks import PpoActor, PpoValueNet
from colosseum.algorithm.rollout_buffer import RolloutBuffer
from colosseum.algorithm.teacher_policy import TeacherPolicy
from colosseum.config.types.algorithm import DaggerPpoConfig, PpoConfig, register_algorithm
from colosseum.config.types.networks import PpoActorConfig, PpoCriticConfig
from colosseum.utils.logger import extract_episode_metrics
from colosseum.utils.torch import get_obs_dims


@register_algorithm("dagger_ppo", config_class=DaggerPpoConfig)
class DaggerPPO(PPO):
  """PPO extended with DAgger-style teacher-student imitation loss."""

  def __init__(
    self,
    config: DaggerPpoConfig,
    env: ManagerBasedRlEnv,
    device: str | torch.device,
    log_fn: Callable[[dict[str, float], int], None],
    log_interval: int,
  ) -> None:
    # super().__init__ calls _build_networks → _build_rollout_buffer (overridden
    # below via MRO, so extras are registered before any data is collected).
    super().__init__(config, env, device, log_fn, log_interval)

    teacher_actor_cfg = PpoActorConfig(
      hidden_layers=list(config.teacher_actor_hidden_layers),
      activation=config.teacher_actor_activation,
    )
    self.teacher = TeacherPolicy(
      checkpoint_path=config.teacher_checkpoint,
      obs_dim=config.teacher_obs_dim,
      action_dim=self.action_dim,
      actor_cfg=teacher_actor_cfg,
      obs_start_idx=config.teacher_obs_start_idx,
      device=self.device,
    )

  # ------------------------------------------------------------------
  # Rollout buffer: register teacher_actions extra slot
  # ------------------------------------------------------------------

  def _build_rollout_buffer(self) -> None:
    assert isinstance(self.config, DaggerPpoConfig)
    self.rollout_buffer = RolloutBuffer(
      num_envs=self.env.num_envs,
      num_steps=self.config.num_steps_per_env,
      actor_obs_dim=self.actor_obs_dim,
      critic_obs_dim=self.critic_obs_dim,
      action_dim=self.action_dim,
      device=self.device,
      extras={"teacher_actions": self.action_dim},
    )

  # ------------------------------------------------------------------
  # Rollout collection: store teacher actions alongside each transition
  # ------------------------------------------------------------------

  def _collect_rollout(
    self,
    current_actor_obs: torch.Tensor,
    current_critic_obs: torch.Tensor,
    current_dones: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert isinstance(self.config, DaggerPpoConfig)

    self.rollout_buffer.clear()

    with torch.no_grad():
      for _step in range(self.config.num_steps_per_env):
        norm_actor_obs = self.actor_obs_normalizer(current_actor_obs)
        norm_critic_obs = self.critic_obs_normalizer(current_critic_obs)

        actions, log_probs, action_means, action_stds = self.actor.act_with_log_prob(
          norm_actor_obs
        )
        values = self.value_net(norm_critic_obs)

        # Query teacher on raw (un-normalized) student obs — teacher normalizes
        # internally using its own saved normalizer.
        teacher_actions = self.teacher.get_actions(current_actor_obs)

        next_obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = (terminated | truncated).float()

        next_actor_obs = self.get_actor_obs(next_obs_dict)
        next_critic_obs = self.get_critic_obs(next_obs_dict)

        if self.config.obs_normalization:
          self.actor_obs_normalizer.update(next_actor_obs)
          self.critic_obs_normalizer.update(next_critic_obs)

        self.cur_reward_sum += rewards

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

        self.update_episode_counts(terminated, truncated)

        if "log" in infos and dones.any():
          self.latest_episode_metrics = extract_episode_metrics(infos["log"])

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
          extras={"teacher_actions": teacher_actions},
        )

        current_actor_obs = next_actor_obs
        current_critic_obs = next_critic_obs
        current_dones = dones

      norm_last_critic = self.critic_obs_normalizer(current_critic_obs)
      last_values = self.value_net(norm_last_critic)

    normalize_globally = not self.config.normalize_advantage_per_mini_batch
    self.rollout_buffer.compute_returns_and_advantages(
      last_values=last_values,
      gamma=self.config.gamma,
      lam=self.config.lam,
      normalize_advantage=normalize_globally,
    )

    return current_actor_obs, current_critic_obs, current_dones

  # ------------------------------------------------------------------
  # Learning step: PPO losses + annealed imitation loss
  # ------------------------------------------------------------------

  def _learning_step(self) -> dict[str, float]:
    assert isinstance(self.config, DaggerPpoConfig)

    total_surrogate_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_imitation_loss = 0.0
    num_updates = 0

    # Annealing coefficient: linear decay from imitation_coef → 0
    progress = min(
      self.global_step / max(self.config.imitation_annealing_steps, 1), 1.0
    )
    lam = self.config.imitation_coef * (1.0 - progress)

    generator = self.rollout_buffer.mini_batch_generator(
      num_mini_batches=self.config.num_mini_batches,
      num_epochs=self.config.num_learning_epochs,
      normalize_advantage_per_mini_batch=self.config.normalize_advantage_per_mini_batch,
    )

    for batch in generator:
      actor_obs_raw = batch["actor_obs"]
      critic_obs_raw = batch["critic_obs"]
      actions = batch["actions"]
      returns = batch["returns"]
      advantages = batch["advantages"]
      old_log_probs = batch["old_log_probs"].squeeze(-1)
      old_action_means = batch["old_action_means"]
      old_action_stds = batch["old_action_stds"]
      target_values = batch["values"]
      teacher_actions = batch["teacher_actions"]

      actor_obs = self.actor_obs_normalizer(actor_obs_raw)
      critic_obs = self.critic_obs_normalizer(critic_obs_raw)

      new_log_probs, entropy = self.actor.evaluate(actor_obs, actions)
      new_values = self.value_net(critic_obs)

      # --- KL divergence ---
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
        kl_mean = kl.mean().item()

      if self.config.schedule == "adaptive" and self.config.desired_kl is not None:
        if kl_mean > self.config.desired_kl * 2.0:
          self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < self.config.desired_kl / 2.0 and kl_mean > 0.0:
          self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for g in self.optimizer.param_groups:
          g["lr"] = self.learning_rate

      # --- Surrogate loss ---
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
        value_loss = torch.max(
          (new_values - returns).pow(2),
          (value_clipped - returns).pow(2),
        ).mean()
      else:
        value_loss = (returns - new_values).pow(2).mean()

      # --- Imitation loss: MSE between student mean and stored teacher actions ---
      student_means = self.actor.forward(actor_obs)
      imitation_loss = F.mse_loss(student_means, teacher_actions)

      # --- Total loss ---
      loss = (
        surrogate_loss
        + self.config.value_loss_coef * value_loss
        - self.config.entropy_coef * entropy.mean()
        + lam * imitation_loss
      )

      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(
        self.actor.parameters(), max_norm=self.config.max_grad_norm
      )
      torch.nn.utils.clip_grad_norm_(
        self.value_net.parameters(), max_norm=self.config.max_grad_norm
      )
      self.optimizer.step()

      total_surrogate_loss += surrogate_loss.item()
      total_value_loss += value_loss.item()
      total_entropy += entropy.mean().item()
      total_kl += kl_mean
      total_imitation_loss += imitation_loss.item()
      num_updates += 1

    self.rollout_buffer.clear()

    return {
      "surrogate_loss": total_surrogate_loss / max(num_updates, 1),
      "value_loss": total_value_loss / max(num_updates, 1),
      "entropy": total_entropy / max(num_updates, 1),
      "kl": total_kl / max(num_updates, 1),
      "imitation_loss": total_imitation_loss / max(num_updates, 1),
      "imitation_coef": lam,
      "learning_rate": self.learning_rate,
    }
