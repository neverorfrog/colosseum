"""PPO with auxiliary depth encoder supervision for the dribbling task.

Extends the base PPO to jointly train the depth encoder via an auxiliary
projection loss: L_visual = MSE(proj_head(z_enc), gt_nxny) * in_front_mask.

The encoder and projection head live on the environment (DribblingEnv).
Their parameters are added to a separate optimizer with a lower learning rate.
The encoder is trained online: at the end of each rollout collection, a fresh
forward pass through encoder + projection head is computed with gradients,
and the auxiliary loss is backpropagated immediately.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any, Callable

import torch
import torch.optim as optim
from mjlab.envs import ManagerBasedRlEnv

from colosseum.algorithm.ppo import PPO
from colosseum.config.types.algorithm import PpoConfig, register_algorithm
from colosseum.tasks.dribbling.mdp.rewards import _project_ball_to_camera

# Camera constants for GT projection (must match the D455 in the scene).
_CAMERA_NAME = "robot/d455_color"
_ASPECT_RATIO = 1280.0 / 720.0


@register_algorithm("dribbling-ppo", config_class=PpoConfig)
class DribblingPPO(PPO):
  """PPO with auxiliary depth encoder loss for visual ball tracking."""

  def __init__(
    self,
    config: PpoConfig,
    env: ManagerBasedRlEnv,
    device: str | torch.device,
    log_fn: Callable[[dict[str, float], int], None],
    log_interval: int,
  ) -> None:
    super().__init__(config, env, device, log_fn, log_interval)

    # Auxiliary loss config
    self.visual_loss_coef = getattr(config, "visual_loss_coef", 1.0)
    self.encoder_lr_scale = getattr(config, "encoder_lr_scale", 0.01)

    # Separate optimizer for encoder + projection head (lower LR).
    dribbling_env = self.env.unwrapped
    self.encoder_optimizer = optim.Adam(
      chain(
        dribbling_env.depth_encoder.parameters(),
        dribbling_env.projection_head.parameters(),
      ),
      lr=self.config.learning_rate * self.encoder_lr_scale,
    )

    # Visual loss from online encoder training (updated during collection)
    self._visual_loss = 0.0

  def _collect_rollout(
    self,
    current_actor_obs: torch.Tensor,
    current_critic_obs: torch.Tensor,
    current_dones: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect rollout, storing z_enc and GT projection in extras."""
    assert isinstance(self.config, PpoConfig)

    self.rollout_buffer.clear()
    dribbling_env = self.env.unwrapped
    visual_loss_sum = 0.0
    visual_loss_count = 0

    with torch.no_grad():
      for _step in range(self.config.num_steps_per_env):
        norm_actor_obs = self.actor_obs_normalizer(current_actor_obs)
        norm_critic_obs = self.critic_obs_normalizer(current_critic_obs)

        actions, log_probs, action_means, action_stds = self.actor.act_with_log_prob(
          norm_actor_obs
        )
        values = self.value_net(norm_critic_obs)

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
          from colosseum.utils.logger import extract_episode_metrics
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
        )

        current_actor_obs = next_actor_obs
        current_critic_obs = next_critic_obs
        current_dones = dones

    # --- Online encoder training (outside no_grad, with gradients) ---
    # Re-forward the current depth buffer through the encoder with
    # gradients enabled, compute the auxiliary projection loss, and
    # backprop through encoder + projection head.
    if dribbling_env._depth_buffer is not None:
      z_enc = dribbling_env.depth_encoder(dribbling_env._depth_buffer)
      predicted_nxny = dribbling_env.projection_head(z_enc)

      nx, ny, in_front = _project_ball_to_camera(
        dribbling_env, _CAMERA_NAME, _ASPECT_RATIO
      )
      gt_nxny = torch.stack(
        [nx.clamp(-2.0, 2.0), ny.clamp(-2.0, 2.0)], dim=-1
      ).detach()

      visual_loss_raw = (predicted_nxny - gt_nxny).pow(2).mean(dim=-1)
      visual_loss = (visual_loss_raw * in_front.float()).mean()

      self.encoder_optimizer.zero_grad()
      (self.visual_loss_coef * visual_loss).backward()
      torch.nn.utils.clip_grad_norm_(
        dribbling_env.depth_encoder.parameters(),
        max_norm=self.config.max_grad_norm,
      )
      torch.nn.utils.clip_grad_norm_(
        dribbling_env.projection_head.parameters(),
        max_norm=self.config.max_grad_norm,
      )
      self.encoder_optimizer.step()

      visual_loss_sum += visual_loss.item()
      visual_loss_count += 1

    # Store visual loss for logging
    self._visual_loss = (
      visual_loss_sum / visual_loss_count if visual_loss_count > 0 else 0.0
    )

    with torch.no_grad():
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

  def _learning_step(self) -> dict[str, float]:
    """PPO update (encoder is trained online during rollout collection)."""
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
      actor_obs_raw = batch["actor_obs"]
      critic_obs_raw = batch["critic_obs"]
      actions = batch["actions"]
      returns = batch["returns"]
      advantages = batch["advantages"]
      old_log_probs = batch["old_log_probs"].squeeze(-1)
      old_action_means = batch["old_action_means"]
      old_action_stds = batch["old_action_stds"]
      target_values = batch["values"]

      actor_obs = self.actor_obs_normalizer(actor_obs_raw)
      critic_obs = self.critic_obs_normalizer(critic_obs_raw)

      new_log_probs, entropy = self.actor.evaluate(actor_obs, actions)
      new_values = self.value_net(critic_obs)

      # KL divergence
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

      # Adaptive KL LR scheduling
      if self.config.schedule == "adaptive" and self.config.desired_kl is not None:
        if kl_mean > self.config.desired_kl * 2.0:
          self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < self.config.desired_kl / 2.0 and kl_mean > 0.0:
          self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for g in self.optimizer.param_groups:
          g["lr"] = self.learning_rate
        for g in self.encoder_optimizer.param_groups:
          g["lr"] = self.learning_rate * self.encoder_lr_scale

      # Surrogate loss
      advantages_squeezed = advantages.squeeze(-1)
      ratio = torch.exp(new_log_probs - old_log_probs)
      surrogate = -advantages_squeezed * ratio
      surrogate_clipped = -advantages_squeezed * torch.clamp(
        ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      # Value loss
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

      # PPO loss (actor + critic)
      ppo_loss = (
        surrogate_loss
        + self.config.value_loss_coef * value_loss
        - self.config.entropy_coef * entropy.mean()
      )

      # PPO gradient step
      self.optimizer.zero_grad()
      ppo_loss.backward()
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
      num_updates += 1

    self.rollout_buffer.clear()

    n = max(num_updates, 1)
    return {
      "surrogate_loss": total_surrogate_loss / n,
      "value_loss": total_value_loss / n,
      "entropy": total_entropy / n,
      "kl": total_kl / n,
      "learning_rate": self.learning_rate,
      "visual_loss": self._visual_loss,
    }

  def save(self, path: str | Path, **extra_state: Any) -> None:
    """Save checkpoint including encoder and projection head."""
    dribbling_env = self.env.unwrapped
    extra_state["depth_encoder_state_dict"] = dribbling_env.depth_encoder.state_dict()
    extra_state["projection_head_state_dict"] = dribbling_env.projection_head.state_dict()
    extra_state["encoder_optimizer_state_dict"] = self.encoder_optimizer.state_dict()
    super().save(path, **extra_state)

  def load(self, path: str | Path) -> dict[str, Any]:
    """Load checkpoint including encoder and projection head."""
    result = super().load(path)

    checkpoint = self._load_checkpoint(path)
    dribbling_env = self.env.unwrapped

    if "depth_encoder_state_dict" in checkpoint:
      dribbling_env.depth_encoder.load_state_dict(checkpoint["depth_encoder_state_dict"])
    if "projection_head_state_dict" in checkpoint:
      dribbling_env.projection_head.load_state_dict(checkpoint["projection_head_state_dict"])
    if "encoder_optimizer_state_dict" in checkpoint:
      self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state_dict"])

    return result
