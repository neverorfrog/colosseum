"""BoosterPPO: PPO faithfully matching booster_gym's runner.py.

Subclasses the colosseum PPO and reuses all of its infrastructure (networks,
rollout buffer, checkpointing, logging, episode tracking) but overrides the
collection and learning phases to match booster's train() exactly:

- only_positive_rewards: total per-step reward clipped at >= 0 (booster clips it
  in the env; here it is clipped as it enters the buffer).
- Full-batch update: every epoch uses the entire rollout (no mini-batches), with
  the value estimate AND GAE recomputed each epoch.
- Timeout bootstrap implemented as reward[timeout] = V(timeout).
- Combined single loss: value + surrogate + bound_coef*bound + entropy bonus.
  No value clipping.
- Adaptive-KL LR (booster bounds 1e-5..1e-2), one LR shared by actor and critic.

booster's runner.py does not use its config's ``symmetric_coef`` — there is no
symmetry term in its loss. As a deliberate divergence, BoosterPPO honors the same
three symmetry flags as the base PPO:
- ``symmetry_data_augmentation`` doubles the rollout along the env dimension with
  left-right mirrored copies before the epoch loop, so the per-env GAE, per-epoch
  value recompute, and all losses run unchanged on 2N envs.
- ``symmetry_loss_coef`` / ``symmetry_critic_coef`` add actor/critic equivariance
  MSE terms to the combined loss (reusing the augmented halves when augmentation is
  on, else computed via a separate mirrored forward).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from colosseum.algorithm.base_algorithm import ObsType
from colosseum.algorithm.ppo import PPO
from colosseum.algorithm.utils.rollout_buffer import RolloutBuffer
from colosseum.config.types.algorithm import BoosterPpoConfig, register_algorithm
from colosseum.mdp.symmetry import mirror_obs
from colosseum.utils.logger import extract_episode_metrics


@register_algorithm("booster_ppo", config_class=BoosterPpoConfig)
class BoosterPPO(PPO):
  """PPO with booster_gym's training mechanics."""

  def _build_rollout_buffer(self) -> None:
    assert isinstance(self.config, BoosterPpoConfig)
    # Store truncation flags alongside the standard fields so the learning step
    # can bootstrap reward = V at timeouts (booster's _check_termination split).
    self.rollout_buffer = RolloutBuffer(
      num_envs=self.env.num_envs,
      num_steps=self.config.num_steps_per_env,
      actor_obs_dim=self.actor_obs_dim,
      critic_obs_dim=self.critic_obs_dim,
      action_dim=self.action_dim,
      device=self.device,
      extras={"time_outs": 1},
    )

  def _collect_rollout(
    self,
    current_actor_obs: torch.Tensor,
    current_critic_obs: torch.Tensor,
    current_dones: torch.Tensor,
    obs_dict: ObsType,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ObsType]:
    assert isinstance(self.config, BoosterPpoConfig)
    cfg = self.config
    self.rollout_buffer.clear()

    with torch.no_grad():
      for _step in range(cfg.num_steps_per_env):
        norm_actor = self.actor_obs_normalizer(current_actor_obs)
        norm_critic = self.critic_obs_normalizer(current_critic_obs)

        actions, log_probs, means, stds = self.actor.act_with_log_prob(norm_actor)
        values = self.value_net(norm_critic)

        obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
        if terminated.is_floating_point():
          dones = torch.clamp(terminated + truncated.float(), 0.0, 1.0)
          term_bool = terminated >= 1.0
        else:
          dones = (terminated | truncated).float()
          term_bool = terminated
        truncated_f = truncated.float()

        next_actor = self.get_actor_obs(obs_dict)
        next_critic = self.get_critic_obs(obs_dict)

        if cfg.obs_normalization:
          self.actor_obs_normalizer.update(next_actor)
          self.critic_obs_normalizer.update(next_critic)

        # Log RAW episode return; clip only what is stored for learning.
        self.cur_reward_sum += rewards
        stored_rewards = rewards.clamp(min=0.0) if cfg.only_positive_rewards else rewards

        self.episode_length_buf += 1
        done_ids = (dones >= 1.0).nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
          self.rewbuffer.extend(
            self.cur_reward_sum[done_ids].cpu().numpy().tolist()
          )
          self.cur_reward_sum[done_ids] = 0.0
          self.episode_lengths.extend(
            self.episode_length_buf[done_ids].cpu().numpy().tolist()
          )
          self.episode_length_buf[done_ids] = 0

        self.update_episode_counts(term_bool, truncated)
        if "log" in infos and (dones >= 1.0).any():
          self.latest_episode_metrics = extract_episode_metrics(infos["log"])

        self.rollout_buffer.add(
          actor_obs=current_actor_obs,
          critic_obs=current_critic_obs,
          actions=actions,
          rewards=stored_rewards,
          dones=dones,
          values=values,
          log_probs=log_probs,
          action_means=means,
          action_stds=stds,
          extras={"time_outs": truncated_f.unsqueeze(-1)},
        )

        current_actor_obs = next_actor
        current_critic_obs = next_critic
        current_dones = dones

    # Bootstrap obs for the final step's value (recomputed each epoch).
    self._last_critic_obs = current_critic_obs
    return current_actor_obs, current_critic_obs, current_dones, obs_dict

  @staticmethod
  def _gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
  ) -> torch.Tensor:
    """booster_gym discount_values: GAE over [T, N, 1] with a [T, N, 1] done mask."""
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(values[0])
    num_steps = rewards.shape[0]
    for t in reversed(range(num_steps)):
      next_nonterminal = 1.0 - dones[t]
      next_values = last_values if t == num_steps - 1 else values[t + 1]
      delta = rewards[t] + gamma * next_nonterminal * next_values - values[t]
      last_adv = delta + gamma * lam * next_nonterminal * last_adv
      advantages[t] = last_adv
    return advantages

  def _learning_step(self) -> dict[str, float]:
    assert isinstance(self.config, BoosterPpoConfig)
    cfg = self.config
    buf = self.rollout_buffer

    actor_obs = self.actor_obs_normalizer(buf.actor_obs)
    critic_obs = self.critic_obs_normalizer(buf.critic_obs)
    last_critic_obs = self.critic_obs_normalizer(self._last_critic_obs)
    actions = buf.actions
    rewards0 = buf.rewards
    dones = buf.dones
    time_outs = buf._extras["time_outs"]

    # Symmetry support. ``augmented`` doubles the rollout along the env dimension
    # with left-right mirrored copies (rewards/dones/time_outs are mirror-invariant
    # scalars, so they are simply repeated); everything downstream (per-env GAE,
    # per-epoch value recompute, surrogate/bound/entropy) then runs on 2N envs.
    # ``n_orig`` is the pre-augmentation env count, used to split the two halves
    # for the equivariance loss terms in the epoch loop below.
    has_symmetry = self._use_symmetry and self._action_mirror_fn is not None
    augmented = has_symmetry and cfg.symmetry_data_augmentation
    n_orig = actor_obs.shape[1]
    if augmented:
      actor_obs = torch.cat(
        [actor_obs, mirror_obs(actor_obs, self._actor_sym_spec)], dim=1
      )
      critic_obs = torch.cat(
        [critic_obs, mirror_obs(critic_obs, self._critic_sym_spec)], dim=1
      )
      last_critic_obs = torch.cat(
        [last_critic_obs, mirror_obs(last_critic_obs, self._critic_sym_spec)], dim=0
      )
      actions = torch.cat([actions, self._action_mirror_fn(actions)], dim=1)
      rewards0 = rewards0.repeat(1, 2, 1)
      dones = dones.repeat(1, 2, 1)
      time_outs = time_outs.repeat(1, 2, 1)

    # Old policy reference (computed once, as in booster's runner).
    with torch.no_grad():
      old_dist = self.actor.get_distribution(actor_obs)
      old_log_probs = old_dist.log_prob(actions).sum(dim=-1)
      old_means = old_dist.loc
      old_stds = old_dist.scale

    sums = {
      "surrogate_loss": 0.0,
      "value_loss": 0.0,
      "bound_loss": 0.0,
      "entropy": 0.0,
      "symmetry_actor_loss": 0.0,
      "symmetry_critic_loss": 0.0,
    }
    kl_mean = 0.0
    n_updates = 0

    for _epoch in range(cfg.num_learning_epochs):
      values = self.value_net(critic_obs)

      with torch.no_grad():
        last_values = self.value_net(last_critic_obs)
        rewards = rewards0.clone()
        to_mask = time_outs.bool()
        rewards[to_mask] = values[to_mask]
        done_mask = (dones.bool() | to_mask).float()
        advantages = self._gae(
          rewards, done_mask, values, last_values, cfg.gamma, cfg.lam
        )
        returns = values + advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

      value_loss = F.mse_loss(values, returns)

      dist = self.actor.get_distribution(actor_obs)
      new_log_probs = dist.log_prob(actions).sum(dim=-1)
      ratio = torch.exp(new_log_probs - old_log_probs)
      adv_s = advantages.squeeze(-1)
      surrogate = torch.max(
        -adv_s * ratio,
        -adv_s * torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param),
      ).mean()

      bound_loss = (
        torch.clamp(dist.loc - 1.0, min=0.0).square().mean()
        + torch.clamp(dist.loc + 1.0, max=0.0).square().mean()
      )
      entropy = dist.entropy().sum(dim=-1)

      # Left-right equivariance penalties (parity with base PPO). When augmented,
      # the two halves are already split at n_orig along the env dim; otherwise a
      # separate mirrored forward is run. Actor: pi(mirror(o)) == mirror(pi(o)).
      # Critic: V(o) == V(mirror(o)).
      symmetry_actor_loss = values.new_zeros(())
      symmetry_critic_loss = values.new_zeros(())
      if has_symmetry:
        if cfg.symmetry_loss_coef > 0.0:
          if augmented:
            mu = dist.loc
            symmetry_actor_loss = F.mse_loss(
              mu[:, n_orig:], self._action_mirror_fn(mu[:, :n_orig])
            )
          else:
            mu_mirror = self.actor.get_distribution(
              mirror_obs(actor_obs, self._actor_sym_spec)
            ).loc
            symmetry_actor_loss = F.mse_loss(
              mu_mirror, self._action_mirror_fn(dist.loc)
            )
        if cfg.symmetry_critic_coef > 0.0:
          if augmented:
            symmetry_critic_loss = F.mse_loss(values[:, :n_orig], values[:, n_orig:])
          else:
            val_mirror = self.value_net(mirror_obs(critic_obs, self._critic_sym_spec))
            symmetry_critic_loss = F.mse_loss(values, val_mirror)

      loss = (
        value_loss
        + surrogate
        + cfg.bound_coef * bound_loss
        - cfg.entropy_coef * entropy.mean()
        + cfg.symmetry_loss_coef * symmetry_actor_loss
        + cfg.symmetry_critic_coef * symmetry_critic_loss
      )

      self.actor_optimizer.zero_grad()
      self.critic_optimizer.zero_grad()
      if not torch.isfinite(loss):
        continue
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
      torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), cfg.max_grad_norm)
      self.actor_optimizer.step()
      self.critic_optimizer.step()

      # Adaptive-KL LR (booster: one LR, bounds 1e-5..1e-2), KL on pre-step dist.
      with torch.no_grad():
        kl = torch.sum(
          torch.log(dist.scale / old_stds + 1e-5)
          + 0.5 * (old_stds.pow(2) + (dist.loc - old_means).pow(2)) / dist.scale.pow(2)
          - 0.5,
          dim=-1,
        )
        kl_mean = float(kl.mean().item())
        if cfg.schedule == "adaptive" and cfg.desired_kl is not None:
          if kl_mean > cfg.desired_kl * 2.0:
            self.actor_learning_rate = max(
              self.min_actor_learning_rate, self.actor_learning_rate / 1.5
            )
          elif 0.0 < kl_mean < cfg.desired_kl / 2.0:
            self.actor_learning_rate = min(
              self.max_actor_learning_rate, self.actor_learning_rate * 1.5
            )
          self.critic_learning_rate = self.actor_learning_rate
          for g in self.actor_optimizer.param_groups:
            g["lr"] = self.actor_learning_rate
          for g in self.critic_optimizer.param_groups:
            g["lr"] = self.critic_learning_rate

      sums["surrogate_loss"] += surrogate.item()
      sums["value_loss"] += value_loss.item()
      sums["bound_loss"] += bound_loss.item()
      sums["entropy"] += entropy.mean().item()
      sums["symmetry_actor_loss"] += symmetry_actor_loss.item()
      sums["symmetry_critic_loss"] += symmetry_critic_loss.item()
      n_updates += 1

    self.rollout_buffer.clear()
    d = max(n_updates, 1)
    out = {
      "surrogate_loss": sums["surrogate_loss"] / d,
      "value_loss": sums["value_loss"] / d,
      "bound_loss": sums["bound_loss"] / d,
      "entropy": sums["entropy"] / d,
      "kl": kl_mean,
      "actor_learning_rate": self.actor_learning_rate,
      "critic_learning_rate": self.critic_learning_rate,
    }
    if has_symmetry:
      out["symmetry_actor_loss"] = sums["symmetry_actor_loss"] / d
      out["symmetry_critic_loss"] = sums["symmetry_critic_loss"] / d
    return out
