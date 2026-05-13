"""DAgger-RMA-PPO: RmaPPO with teacher-student imitation loss.

This variant is tailored for RMA-style policies whose actor input is:

    concat(normalized_actor_obs, privileged_latent)

The teacher is a frozen stage-0 policy head loaded from checkpoint. During
rollout and learning it is queried with:

    concat(teacher_normalized_actor_obs, current_student_latent)

This keeps the nominal stage-0 action style as a regularizer while PPO still
learns obstacle-specific behavior through the live RMA encoders and rewards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from mjlab.envs import ManagerBasedRlEnv

from colosseum.algorithm.networks.ppo_networks import PpoActor
from colosseum.algorithm.rma_ppo import RmaPPO
from colosseum.algorithm.utils.normalization import EmpiricalNormalization
from colosseum.algorithm.utils.rollout_buffer import RolloutBuffer
from colosseum.config.types.algorithm import DaggerPpoConfig, register_algorithm
from colosseum.config.types.networks import PpoActorConfig


@register_algorithm("dagger_rma_ppo", config_class=DaggerPpoConfig)
class DaggerRmaPPO(RmaPPO):
  """RmaPPO extended with a DAgger-style imitation loss."""

  def __init__(
    self,
    config: DaggerPpoConfig,
    env: ManagerBasedRlEnv,
    device: str | torch.device,
    log_fn: Callable[[dict[str, float], int], None],
    log_interval: int,
  ) -> None:
    super().__init__(config, env, device, log_fn, log_interval)

    teacher_actor_cfg = PpoActorConfig(
      hidden_layers=list(config.teacher_actor_hidden_layers),
      activation=config.teacher_actor_activation,
    )
    self.teacher_actor = PpoActor(
      self.actor_obs_dim + self.rma_manager.total_latent_dim,
      self.action_dim,
      teacher_actor_cfg,
    ).to(self.device)
    self.teacher_actor_obs_normalizer = EmpiricalNormalization(self.actor_obs_dim).to(
      self.device
    )

    checkpoint = torch.load(
      Path(config.teacher_checkpoint), map_location="cpu", weights_only=False
    )
    self.teacher_actor.load_state_dict(checkpoint["actor_state_dict"])
    self.teacher_actor_obs_normalizer.load_state_dict(
      checkpoint["actor_obs_normalizer_state_dict"]
    )
    self.teacher_actor.eval()
    self.teacher_actor_obs_normalizer.eval()
    for param in self.teacher_actor.parameters():
      param.requires_grad_(False)
    for param in self.teacher_actor_obs_normalizer.parameters():
      param.requires_grad_(False)

  def _build_rollout_buffer(self) -> None:
    config = self.config
    assert isinstance(config, DaggerPpoConfig)

    obs_mgr = self.env.observation_manager
    privileged_obs_dims = {
      group: obs_mgr.group_obs_dim[group][0]
      for group in self.rma_manager.privileged_group_names
    }

    self.rollout_buffer = RolloutBuffer(
      num_envs=self.env.num_envs,
      num_steps=config.num_steps_per_env,
      actor_obs_dim=self.actor_obs_dim,
      critic_obs_dim=self.critic_obs_dim,
      action_dim=self.action_dim,
      device=self.device,
      extras={"teacher_actions": self.action_dim},
      privileged_obs_dims=privileged_obs_dims,
    )

  @torch.no_grad()
  def _get_teacher_actions(
    self,
    actor_obs_raw: torch.Tensor,
    privileged_obs: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    """Teacher action means using teacher normalizer and current privileged latents."""
    teacher_actor_obs = self.teacher_actor_obs_normalizer(actor_obs_raw)
    z = self.rma_manager.encode(privileged_obs)
    teacher_input = torch.cat([teacher_actor_obs, z], dim=-1)
    return self.teacher_actor.forward(teacher_input)

  def _collect_rollout(
    self,
    current_actor_obs: torch.Tensor,
    current_critic_obs: torch.Tensor,
    current_dones: torch.Tensor,
    obs_dict: dict[str, Any] | torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any] | torch.Tensor]:
    config = self.config
    assert isinstance(config, DaggerPpoConfig)

    self.rollout_buffer.clear()

    with torch.no_grad():
      for _step in range(config.num_steps_per_env):
        norm_actor_obs_base = self.actor_obs_normalizer(current_actor_obs)
        norm_critic_obs = self.critic_obs_normalizer(current_critic_obs)
        current_privileged_obs = self.get_privileged_obs(obs_dict)
        norm_actor_obs = self._compose_actor_input(
          norm_actor_obs_base, current_privileged_obs
        )

        actions, log_probs, action_means, action_stds = self.actor.act_with_log_prob(
          norm_actor_obs
        )
        values = self.value_net(norm_critic_obs)
        teacher_actions = self._get_teacher_actions(
          current_actor_obs, current_privileged_obs
        )

        obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = (terminated | truncated).float()

        next_actor_obs = self.get_actor_obs(obs_dict)
        next_critic_obs = self.get_critic_obs(obs_dict)

        if config.obs_normalization:
          self.actor_obs_normalizer.update(next_actor_obs)
          self.critic_obs_normalizer.update(next_critic_obs)

        self.cur_reward_sum += rewards

        if not getattr(self.env.cfg, "is_finite_horizon", True):
          truncated_mask = truncated.float()
          if truncated_mask.any():
            norm_next_critic = self.critic_obs_normalizer(next_critic_obs)
            truncated_values = self.value_net(norm_next_critic).squeeze(-1)
            rewards = rewards + config.gamma * truncated_values * truncated_mask

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

        from colosseum.utils.logger import extract_episode_metrics

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
          privileged_obs=current_privileged_obs,
        )

        current_actor_obs = next_actor_obs
        current_critic_obs = next_critic_obs
        current_dones = dones

      norm_last_critic = self.critic_obs_normalizer(current_critic_obs)
      last_values = self.value_net(norm_last_critic)

    self.rollout_buffer.compute_returns_and_advantages(
      last_values=last_values,
      gamma=config.gamma,
      lam=config.lam,
      normalize_advantage=False,
    )

    if not config.normalize_advantage_per_mini_batch:
      self.rollout_buffer.advantages = self._normalize_advantages_multi_gpu(
        self.rollout_buffer.advantages
      )

    return current_actor_obs, current_critic_obs, current_dones, obs_dict

  def _learning_step(self) -> dict[str, float]:
    config = self.config
    assert isinstance(config, DaggerPpoConfig)

    total_surrogate_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_imitation_loss = 0.0
    num_updates = 0

    progress = min(self.global_step / max(config.imitation_annealing_steps, 1), 1.0)
    lam = config.imitation_coef * (1.0 - progress)

    generator = self.rollout_buffer.mini_batch_generator(
      num_mini_batches=config.num_mini_batches,
      num_epochs=config.num_learning_epochs,
      normalize_advantage_per_mini_batch=config.normalize_advantage_per_mini_batch,
    )

    for batch in generator:
      actor_obs_raw = batch["actor_obs"]
      critic_obs_raw = batch["critic_obs"]
      privileged_obs = batch["privileged_obs"]
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
      actor_input = self._compose_actor_input(actor_obs, privileged_obs)

      new_log_probs, entropy = self.actor.evaluate(actor_input, actions)
      new_values = self.value_net(critic_obs)

      with torch.inference_mode():
        mu_batch = self.actor.forward(actor_input)
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

      if config.schedule == "adaptive" and config.desired_kl is not None:
        if kl_mean > config.desired_kl * 2.0:
          self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < config.desired_kl / 2.0 and kl_mean > 0.0:
          self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for g in self.optimizer.param_groups:
          g["lr"] = self.learning_rate

      advantages_squeezed = advantages.squeeze(-1)
      ratio = torch.exp(new_log_probs - old_log_probs)
      surrogate = -advantages_squeezed * ratio
      surrogate_clipped = -advantages_squeezed * torch.clamp(
        ratio, 1.0 - config.clip_param, 1.0 + config.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if config.use_clipped_value_loss:
        value_clipped = target_values + torch.clamp(
          new_values - target_values,
          -config.clip_param,
          config.clip_param,
        )
        value_loss = torch.max(
          (new_values - returns).pow(2),
          (value_clipped - returns).pow(2),
        ).mean()
      else:
        value_loss = (returns - new_values).pow(2).mean()

      student_means = self.actor.forward(actor_input)
      imitation_loss = F.mse_loss(student_means, teacher_actions)

      loss = (
        surrogate_loss
        + config.value_loss_coef * value_loss
        - config.entropy_coef * entropy.mean()
        + lam * imitation_loss
      )

      self.optimizer.zero_grad()
      loss.backward()
      self._distributed_average_optimizer_grads(self.optimizer)
      torch.nn.utils.clip_grad_norm_(
        self.actor.parameters(), max_norm=config.max_grad_norm
      )
      torch.nn.utils.clip_grad_norm_(
        self.value_net.parameters(), max_norm=config.max_grad_norm
      )
      self.optimizer.step()

      total_surrogate_loss += surrogate_loss.item()
      total_value_loss += value_loss.item()
      total_entropy += entropy.mean().item()
      total_kl += kl_mean
      total_imitation_loss += imitation_loss.item()
      num_updates += 1

    if self.is_distributed:
      totals = self._distributed_sum_vector(
        [
          total_surrogate_loss,
          total_value_loss,
          total_entropy,
          total_kl,
          total_imitation_loss,
          float(num_updates),
        ]
      )
      total_surrogate_loss = totals[0]
      total_value_loss = totals[1]
      total_entropy = totals[2]
      total_kl = totals[3]
      total_imitation_loss = totals[4]
      num_updates = int(totals[5])

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
