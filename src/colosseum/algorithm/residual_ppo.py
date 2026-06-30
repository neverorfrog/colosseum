"""ResidualPPO — PPO with frozen base skills + a trainable residual + orchestrator.

Subclass of PPO. The actor is a ResidualActor (composite): one or more frozen
base skills, a trainable residual branch, and a gating orchestrator that blends
them via PoE-style fusion. See `residual_ppo_plan.md`.

Differences from plain PPO:
  - The actor reads multiple named observation groups (one per base skill, one
    for the residual, plus "orchestrator"), not a single "actor" group. Per-skill
    raw obs are stored in the rollout buffer's privileged-obs channel and
    re-normalized at learning time.
  - Per-skill EmpiricalNormalization. Base-skill normalizers are loaded from the
    base checkpoints and (by default) frozen.
  - The actor optimizer trains only the residual branch + orchestrator; the
    frozen base branches are excluded.
  - KL is computed from the combined distribution (ResidualActor has no .std /
    .forward). Two extra penalties keep the residual small and the orchestrator
    biased toward the frozen base.

Symmetry is per-group: one mirror spec per actor-side obs group (each base skill
group + residual + orchestrator), with the symmetry loss on the COMBINED action
mean. Toggled by the same symmetry_* config fields as base PPO.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from mjlab.envs import ManagerBasedRlEnv

from colosseum.algorithm.base_algorithm import ObsType
from colosseum.algorithm.networks.ppo_networks import PpoValueNet
from colosseum.algorithm.networks.residual_ppo_networks import (
  BaseSkill,
  MlpBaseSkill,
  ResidualActor,
  ResidualBaseSkill,
  RmaBaseSkill,
)
from colosseum.algorithm.ppo import PPO
from colosseum.algorithm.utils.normalization import (
  EmpiricalNormalization,
  IdentityNormalizer,
)
from colosseum.algorithm.utils.rollout_buffer import RolloutBuffer
from colosseum.config.types.algorithm import ResidualPpoConfig, register_algorithm
from colosseum.mdp.symmetry import (
  TermMirrorSpec,
  augment_actions,
  augment_obs,
  build_symmetry_spec,
  mirror_obs,
)
from colosseum.utils.logger import extract_episode_metrics


@register_algorithm("residual_ppo", config_class=ResidualPpoConfig)
class ResidualPPO(PPO):
  """PPO with frozen base skills blended with a trainable residual by an orchestrator."""

  # Hardcoded group consumed by the orchestrator (mirrors PPO hardcoding "critic").
  _ORCH_GROUP = "orchestrator"

  def __init__(
    self,
    config: ResidualPpoConfig,
    env: ManagerBasedRlEnv,
    device: str | torch.device,
    log_fn: Callable[[dict[str, float], int], None],
    log_interval: int,
  ) -> None:
    super().__init__(
      config=config, env=env, device=device, log_fn=log_fn, log_interval=log_interval
    )
    # Lifecycle is separate from structure: load + freeze the base skills, then
    # bias the orchestrator toward them (both after the networks are built).
    self._load_frozen_skills()
    self.residual_actor.init_orchestrator_bias(config.residual_actor.init_favored_logit)

  # ------------------------------------------------------------------ #
  # Construction
  # ------------------------------------------------------------------ #

  def _all_skill_groups(self) -> set[str]:
    """All actor-side obs groups: every base skill group + residual + orchestrator."""
    groups: set[str] = set()
    for skill in self.residual_actor.base_branches.values():
      groups.update(skill.obs_groups)
    groups.add(self._residual_group)
    groups.add(self._orch_group)
    return groups

  def _build_networks(self) -> None:
    assert isinstance(self.config, ResidualPpoConfig)
    ra = self.config.residual_actor
    obs_dim = self.env.observation_manager.group_obs_dim

    def gdim(group: str) -> int:
      return obs_dim[group][0]

    self._base_skill_groups = {n: s.obs_group for n, s in ra.base_skills.items()}
    self._residual_group = ra.residual_obs_group
    self._orch_group = ra.orchestrator_obs_group

    self.action_dim = int(np.prod(self.env.single_action_space.shape))
    self.critic_obs_dim = gdim("critic")
    self.actor_obs_dim = 1

    self._base_skill_names = list(ra.base_skills)
    self._feed_dim = sum(
      (ra.base_skills[n].latent_dim if ra.base_skills[n].kind == "rma" else 0)
      for n in ra.latent_feed_skills
    )

    base_skills: list[BaseSkill] = []
    for name in self._base_skill_names:
      skill_cfg = ra.base_skills[name]
      group = skill_cfg.obs_group
      if skill_cfg.kind == "rma":
        s = RmaBaseSkill(
          actor_obs_group=group,
          actor_obs_dim=gdim(group),
          latent_dim=skill_cfg.latent_dim,
          window_size=skill_cfg.window_size,
          action_dim=self.action_dim,
          actor_config=skill_cfg.actor,
          term_name=skill_cfg.term_name,
          device=self.device,
        )
      elif skill_cfg.kind == "residual":
        s = self._build_residual_base_skill(name, skill_cfg)
      else:
        s = MlpBaseSkill(
          obs_group=group,
          obs_dim=gdim(group),
          action_dim=self.action_dim,
          actor_config=skill_cfg.actor,
          device=self.device,
        )
      base_skills.append(s)

    self.residual_actor = ResidualActor(
      base_skills=base_skills,
      base_skill_names=self._base_skill_names,
      residual_obs_dim=gdim(self._residual_group),
      residual_config=ra.residual_actor,
      action_dim=self.action_dim,
      orchestrator_obs_dim=gdim(self._orch_group),
      orchestrator_hidden_layers=ra.orchestrator.hidden_layers,
      orchestrator_activation=ra.orchestrator.activation,
      latent_feed_skills=ra.latent_feed_skills,
    ).to(self.device)
    self.actor = self.residual_actor

    self.value_net = PpoValueNet(self.critic_obs_dim, self.config.critic).to(self.device)

  def _build_residual_base_skill(self, name: str, skill_cfg) -> BaseSkill:
    from colosseum.algorithm.utils.normalization import EmpiricalNormalization

    ckpt = torch.load(skill_cfg.checkpoint, map_location=self.device, weights_only=False)
    inner_cfg = ckpt["config"]
    obs_dim = self.env.observation_manager.group_obs_dim

    inner_base_skill_groups: dict[str, str] = {}
    for bname, bsc in inner_cfg.residual_actor.base_skills.items():
      inner_base_skill_groups[bname] = bsc.obs_group
    inner_residual_group = inner_cfg.residual_actor.residual_obs_group
    inner_orch_group = skill_cfg.inner_orch_obs_group

    inner_base_skill_names = list(inner_base_skill_groups)
    inner_action_dim = int(np.prod(self.env.single_action_space.shape))
    inner_num_skills = len(inner_base_skill_names) + 1

    inner_base_skills: list[BaseSkill] = []
    for bname in inner_base_skill_names:
      bsc = inner_cfg.residual_actor.base_skills[bname]
      bg = bsc.obs_group
      inner_s = MlpBaseSkill(
        obs_group=bg,
        obs_dim=obs_dim[bg][0],
        action_dim=inner_action_dim,
        actor_config=bsc.actor,
        device=self.device,
      )
      inner_base_skills.append(inner_s)

    inner_norms: dict[str, EmpiricalNormalization] = {}
    all_skill_groups = set(inner_base_skill_groups.values())
    all_skill_groups.add(inner_residual_group)
    all_skill_groups.add(inner_orch_group)
    for g in all_skill_groups:
      inner_norms[g] = EmpiricalNormalization(shape=obs_dim[g][0], device=self.device)

    obs_groups = tuple(all_skill_groups)

    inner_actor = ResidualActor(
      base_skills=inner_base_skills,
      base_skill_names=inner_base_skill_names,
      residual_obs_dim=obs_dim[inner_residual_group][0],
      residual_config=inner_cfg.residual_actor.residual_actor,
      action_dim=inner_action_dim,
      orchestrator_obs_dim=obs_dim[inner_orch_group][0],
      orchestrator_hidden_layers=inner_cfg.residual_actor.orchestrator.hidden_layers,
      orchestrator_activation=inner_cfg.residual_actor.orchestrator.activation,
      latent_feed_skills=(),
    ).to(self.device)

    state_dict = ckpt["residual_actor_state_dict"]
    orch_w_key = "orchestrator.weight_head.weight"
    if orch_w_key in state_dict:
      w = state_dict[orch_w_key]
      if w.shape[0] == inner_num_skills:
        inner_actor.orchestrator.weight_head = nn.Linear(
          int(inner_actor.orchestrator.weight_head.in_features), inner_num_skills
        ).to(self.device)
        inner_actor.orchestrator.num_skills = inner_num_skills
        def _old_forward(obs):
          logits = inner_actor.orchestrator.weight_head(
            inner_actor.orchestrator.backbone(obs)
          )
          return nn.functional.softmax(logits, dim=-1)
        inner_actor.orchestrator.forward = _old_forward

    return ResidualBaseSkill(
      inner=inner_actor,
      inner_norms=inner_norms,
      obs_groups=obs_groups,
      residual_group=inner_residual_group,
      orch_group=inner_orch_group,
      base_group_for=inner_base_skill_groups,
    )

  def _build_optimizers(self) -> None:
    """Actor optimizer over residual + orchestrator only; critic unchanged."""
    self.actor_optimizer = optim.AdamW(
      self.residual_actor.trainable_parameters(),
      lr=self.actor_learning_rate,
      weight_decay=self.config.weight_decay,
    )
    self.critic_optimizer = optim.AdamW(
      self.value_net.parameters(),
      lr=self.critic_learning_rate,
      weight_decay=self.config.weight_decay,
    )

  def _build_rollout_buffer(self) -> None:
    assert isinstance(self.config, ResidualPpoConfig)
    obs_dim = self.env.observation_manager.group_obs_dim

    privileged_obs_dims = {
      g: obs_dim[g][0] for g in self._all_skill_groups()
    }

    extras: dict[str, int] = {}
    for name in self._base_skill_names:
      extras[f"base_mean_{name}"] = self.action_dim
      extras[f"base_std_{name}"] = self.action_dim

    if self._feed_dim > 0:
      extras["z_feed"] = self._feed_dim

    cfg = self.config
    aug = (
      cfg.symmetry_loss_coef > 0.0
      or cfg.symmetry_critic_coef > 0.0
      or cfg.symmetry_data_augmentation
    )
    if aug and cfg.symmetry_data_augmentation:
      for name in self._base_skill_names:
        extras[f"base_mean_mirror_{name}"] = self.action_dim
        extras[f"base_std_mirror_{name}"] = self.action_dim
      if self._feed_dim > 0:
        extras["z_feed_mirror"] = self._feed_dim

    self.rollout_buffer = RolloutBuffer(
      num_envs=self.env.num_envs,
      num_steps=self.config.num_steps_per_env,
      actor_obs_dim=self.actor_obs_dim,
      critic_obs_dim=self.critic_obs_dim,
      action_dim=self.action_dim,
      device=self.device,
      privileged_obs_dims=privileged_obs_dims,
      extras=extras,
    )

  def _build_normalizer(self) -> None:
    assert isinstance(self.config, ResidualPpoConfig)
    obs_dim = self.env.observation_manager.group_obs_dim

    self.skill_normalizers: dict[str, EmpiricalNormalization | IdentityNormalizer] = {}

    if not self.config.obs_normalization:
      self.skill_normalizers[self._residual_group] = IdentityNormalizer()
      self.skill_normalizers[self._orch_group] = IdentityNormalizer()
      self.critic_obs_normalizer = IdentityNormalizer()
      self.actor_obs_normalizer = IdentityNormalizer()
      return

    self.skill_normalizers[self._residual_group] = EmpiricalNormalization(
      shape=obs_dim[self._residual_group][0], device=self.device
    )
    self.skill_normalizers[self._orch_group] = EmpiricalNormalization(
      shape=obs_dim[self._orch_group][0], device=self.device
    )
    self.critic_obs_normalizer = EmpiricalNormalization(
      shape=self.critic_obs_dim, device=self.device
    )
    self.actor_obs_normalizer = IdentityNormalizer()

  def _setup_symmetry(self) -> None:
    """Build one mirror spec per actor-side group (residual mode has no 'actor' group)."""
    assert isinstance(self.config, ResidualPpoConfig)
    # Unused singular specs from the base contract; residual uses per-group specs.
    self._actor_sym_spec: list[TermMirrorSpec] | None = None
    self._critic_sym_spec: list[TermMirrorSpec] | None = None
    self._action_mirror_fn = None
    self._use_symmetry = False
    self._actor_sym_specs: dict[str, list[TermMirrorSpec]] = {}

    cfg = self.config
    if not (
      cfg.symmetry_loss_coef > 0.0
      or cfg.symmetry_critic_coef > 0.0
      or cfg.symmetry_data_augmentation
    ):
      return

    self._use_symmetry = True
    obs_mgr = self.env.observation_manager
    for group in self._all_skill_groups():
      self._actor_sym_specs[group] = build_symmetry_spec(obs_mgr, group)
    self._critic_sym_spec = build_symmetry_spec(obs_mgr, "critic")
    # Action mirror fn from any actor-side group carrying the "actions" term.
    for group in self._all_skill_groups():
      if "actions" in obs_mgr.active_terms.get(group, []):
        actions_cfg = obs_mgr.get_term_cfg(group, "actions")
        self._action_mirror_fn = getattr(actions_cfg, "mirror_fn", None)
        break

  def _load_frozen_skills(self) -> None:
    assert isinstance(self.config, ResidualPpoConfig)
    for name, skill_cfg in self.config.residual_actor.base_skills.items():
      checkpoint = torch.load(
        skill_cfg.checkpoint, map_location=self.device, weights_only=False
      )
      self.residual_actor.base_branches[name].load_pretrained(checkpoint)
    self.residual_actor.freeze_base_skills()

  # ------------------------------------------------------------------ #
  # Observation routing helpers
  # ------------------------------------------------------------------ #

  def _normalized_skill_inputs(
    self, obs: dict[str, torch.Tensor]
  ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return (base_obs dict-of-dicts per skill, normalized residual, normalized orch)."""
    base_obs = {
      name: {g: obs[g] for g in skill.obs_groups}
      for name, skill in self.residual_actor.base_branches.items()
    }
    residual_obs = self.skill_normalizers[self._residual_group](obs[self._residual_group])
    orch_obs = self.skill_normalizers[self._orch_group](obs[self._orch_group])
    return base_obs, residual_obs, orch_obs

  def _raw_skill_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Raw per-skill obs for the rollout buffer (used by symmetry)."""
    return {g: obs[g] for g in self._all_skill_groups()}

  def _update_skill_normalizers(self, obs: dict[str, torch.Tensor]) -> None:
    for g in (self._residual_group, self._orch_group):
      if g in self.skill_normalizers:
        self.skill_normalizers[g].update(obs[g])

  # ------------------------------------------------------------------
  # Symmetry helpers for residual + orch only (base params are cached)
  # ------------------------------------------------------------------

  def _augment_skill_inputs(
    self,
    residual_raw: torch.Tensor,
    orch_raw: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    # Mirror in raw obs space (where the mirror specs are defined), then
    # normalize — matching how base skills mirror their obs.
    rnorm = self.skill_normalizers[self._residual_group]
    onorm = self.skill_normalizers[self._orch_group]
    residual_aug = rnorm(augment_obs(residual_raw, self._actor_sym_specs[self._residual_group]))
    orch_aug = onorm(augment_obs(orch_raw, self._actor_sym_specs[self._orch_group]))
    return residual_aug, orch_aug

  def _mirror_skill_inputs(
    self,
    residual_raw: torch.Tensor,
    orch_raw: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    rnorm = self.skill_normalizers[self._residual_group]
    onorm = self.skill_normalizers[self._orch_group]
    residual_m = rnorm(mirror_obs(residual_raw, self._actor_sym_specs[self._residual_group]))
    orch_m = onorm(mirror_obs(orch_raw, self._actor_sym_specs[self._orch_group]))
    return residual_m, orch_m

  def get_actor_obs(self, obs: ObsType) -> torch.Tensor:
    """Residual mode has no single 'actor' group.

    Cache the full obs dict so per-skill routing (and prewarm) can read every
    group, and return the orchestrator obs as the threaded placeholder the
    inherited loop expects.
    """
    assert isinstance(obs, dict)
    self._cached_obs_dict = obs
    return obs[self._orch_group]

  def _prewarm_actor_obs(self, actor_obs: torch.Tensor) -> torch.Tensor:
    """Prewarm the fresh (non-frozen) per-skill normalizers from initial obs."""
    if self.config.obs_normalization:
      self._update_skill_normalizers(self._cached_obs_dict)
    return actor_obs

  # ------------------------------------------------------------------ #
  # Rollout collection
  # ------------------------------------------------------------------ #

  def _collect_rollout(
    self,
    current_actor_obs: torch.Tensor,
    current_critic_obs: torch.Tensor,
    current_dones: torch.Tensor,
    obs_dict: ObsType,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ObsType]:
    assert isinstance(self.config, ResidualPpoConfig)
    assert isinstance(obs_dict, dict)

    self.rollout_buffer.clear()
    dummy_actor_obs = torch.zeros(self.env.num_envs, 1, device=self.device)

    with torch.no_grad():
      for skill in self.residual_actor.base_branches.values():
        skill.update_state({g: obs_dict[g] for g in skill.obs_groups})

    with torch.no_grad():
      for _step in range(self.config.num_steps_per_env):
        raw_skill_obs = self._raw_skill_obs(obs_dict)
        base_obs, residual_obs, orch_obs = self._normalized_skill_inputs(obs_dict)
        norm_critic_obs = self.critic_obs_normalizer(current_critic_obs)

        actions, log_probs, action_means, action_stds = (
          self.residual_actor.act_with_log_prob(base_obs, residual_obs, orch_obs)
        )
        values = self.value_net(norm_critic_obs)

        base_means = [m.detach().clone() for m in self.residual_actor.last_base_means]
        base_stds = [s.detach().clone() for s in self.residual_actor.last_base_stds]
        z_feed = self.residual_actor.last_z_feed.detach().clone()

        # Snapshot each RMA skill's pre-step window now: the symmetry mirror block
        # below runs after update_state() rolls in the post-step obs, so reading
        # skill._window there would use the next step's window (misaligned with the
        # pre-step base_means / z_feed captured above).
        prestep_windows = {
          name: self.residual_actor.base_branches[name]._window.clone()
          for name in self._base_skill_names
          if isinstance(self.residual_actor.base_branches[name], RmaBaseSkill)
        }

        prev_obs_dict = obs_dict
        obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
        assert isinstance(obs_dict, dict)
        if terminated.is_floating_point():
          dones = torch.clamp(terminated + truncated.float(), 0.0, 1.0)
        else:
          dones = (terminated | truncated).float()

        next_critic_obs = self.get_critic_obs(obs_dict)

        if self.config.obs_normalization:
          self._update_skill_normalizers(obs_dict)
          self.critic_obs_normalizer.update(next_critic_obs)

        self.cur_reward_sum += rewards

        rewards = self._blend_amp_reward(prev_obs_dict, obs_dict, rewards, dones)

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

        hard_terminated = (
          terminated >= 1.0 if terminated.is_floating_point() else terminated
        )
        self.update_episode_counts(hard_terminated, truncated)

        if "log" in infos and (dones >= 1.0).any():
          self.latest_episode_metrics = extract_episode_metrics(infos["log"])

        for skill in self.residual_actor.base_branches.values():
          skill.update_state({g: obs_dict[g] for g in skill.obs_groups})

        episode_done_ids_for_skill = episode_done_ids if len(episode_done_ids) > 0 else None
        for skill in self.residual_actor.base_branches.values():
          skill.reset_state(episode_done_ids_for_skill)

        extras: dict[str, torch.Tensor] = {}
        for i, name in enumerate(self._base_skill_names):
          extras[f"base_mean_{name}"] = base_means[i]
          extras[f"base_std_{name}"] = base_stds[i]
        if self._feed_dim > 0:
          extras["z_feed"] = z_feed

        if self._use_symmetry and self.config.symmetry_data_augmentation:
          # Mirror of each RMA skill's pre-step window (used for both the base
          # mirror-mean latent and the residual's z_feed_mirror).
          mirrored_windows = {
            name: mirror_obs(
              w.reshape(-1, w.shape[-1]),
              self._actor_sym_specs[self._base_skill_groups[name]],
            ).reshape_as(w)
            for name, w in prestep_windows.items()
          }
          for name, g in self._base_skill_groups.items():
            skill = self.residual_actor.base_branches[name]
            mirrored = {grp: mirror_obs(prev_obs_dict[grp], self._actor_sym_specs[grp]) for grp in skill.obs_groups}
            mean_m, std_m = skill.get_distribution_params(
              mirrored, window=mirrored_windows.get(name)
            )
            extras[f"base_mean_mirror_{name}"] = mean_m.detach()
            extras[f"base_std_mirror_{name}"] = std_m.detach()

          if self._feed_dim > 0:
            latents_m: list[torch.Tensor] = []
            for name in self.config.residual_actor.latent_feed_skills:
              skill = self.residual_actor.base_branches[name]
              if isinstance(skill, RmaBaseSkill):
                latents_m.append(skill.latent_for(mirrored_windows[name]))
            if latents_m:
              extras["z_feed_mirror"] = torch.cat(latents_m, dim=-1).detach()

        self.rollout_buffer.add(
          actor_obs=dummy_actor_obs,
          critic_obs=current_critic_obs,
          actions=actions,
          rewards=rewards,
          dones=dones,
          values=values,
          log_probs=log_probs,
          action_means=action_means,
          action_stds=action_stds,
          privileged_obs=raw_skill_obs,
          extras=extras,
        )

        current_critic_obs = next_critic_obs
        current_dones = dones

      norm_last_critic = self.critic_obs_normalizer(current_critic_obs)
      last_values = self.value_net(norm_last_critic)

    self.rollout_buffer.compute_returns_and_advantages(
      last_values=last_values,
      gamma=self.config.gamma,
      lam=self.config.lam,
      normalize_advantage=False,
    )

    if not self.config.normalize_advantage_per_mini_batch:
      self.rollout_buffer.advantages = self._normalize_advantages_multi_gpu(
        self.rollout_buffer.advantages
      )

    return self.get_actor_obs(obs_dict), current_critic_obs, current_dones, obs_dict

  def _blend_amp_reward(
    self,
    prev_obs_dict: dict[str, torch.Tensor],
    obs_dict: dict[str, torch.Tensor],
    rewards: torch.Tensor,
    dones: torch.Tensor,
  ) -> torch.Tensor:
    """Hook to fold an AMP style reward into the per-step reward (no-op here)."""
    return rewards

  # ------------------------------------------------------------------ #
  # Learning
  # ------------------------------------------------------------------ #

  def _learning_step(self) -> dict[str, float]:
    assert isinstance(self.config, ResidualPpoConfig)

    total_surrogate_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_residual_magnitude = 0.0
    total_residual_weight = 0.0
    total_symmetry_actor_loss = 0.0
    total_symmetry_critic_loss = 0.0
    num_updates = 0

    generator = self.rollout_buffer.mini_batch_generator(
      num_mini_batches=self.config.num_mini_batches,
      num_epochs=self.config.num_learning_epochs,
      normalize_advantage_per_mini_batch=self.config.normalize_advantage_per_mini_batch,
    )

    augment = (
      self._use_symmetry
      and self.config.symmetry_data_augmentation
      and self._action_mirror_fn is not None
    )

    for batch in generator:
      raw_skill_obs = batch["privileged_obs"]
      actions = batch["actions"]
      returns = batch["returns"]
      advantages = batch["advantages"]
      old_log_probs = batch["old_log_probs"].squeeze(-1)
      old_action_means = batch["old_action_means"]
      old_action_stds = batch["old_action_stds"]
      target_values = batch["values"]

      base_means = [batch[f"base_mean_{name}"] for name in self._base_skill_names]
      base_stds = [batch[f"base_std_{name}"] for name in self._base_skill_names]
      z_feed = batch["z_feed"] if self._feed_dim > 0 else torch.zeros(1, 0, device=self.device)

      residual_obs_raw = raw_skill_obs[self._residual_group]
      orch_obs_raw = raw_skill_obs[self._orch_group]

      residual_obs = self.skill_normalizers[self._residual_group](residual_obs_raw)
      orch_obs = self.skill_normalizers[self._orch_group](orch_obs_raw)

      critic_obs = self.critic_obs_normalizer(batch["critic_obs"])

      original_batch_size = actions.shape[0]
      if augment:
        residual_obs, orch_obs = self._augment_skill_inputs(residual_obs_raw, orch_obs_raw)
        critic_obs = self.critic_obs_normalizer(
          augment_obs(batch["critic_obs"], self._critic_sym_spec)
        )
        actions = augment_actions(actions, self._action_mirror_fn)
        old_log_probs = old_log_probs.repeat(2)
        target_values = target_values.repeat(2, 1)
        advantages = advantages.repeat(2, 1)
        returns = returns.repeat(2, 1)
        old_action_means = old_action_means.repeat(2, 1)
        old_action_stds = old_action_stds.repeat(2, 1)

        for i, name in enumerate(self._base_skill_names):
          base_means[i] = torch.cat([base_means[i], batch[f"base_mean_mirror_{name}"]], dim=0)
          base_stds[i] = torch.cat([base_stds[i], batch[f"base_std_mirror_{name}"]], dim=0)
        z_feed = torch.cat([z_feed, batch["z_feed_mirror"]], dim=0) if self._feed_dim > 0 else z_feed

      new_log_probs, entropy_all = self.residual_actor.evaluate_cached(
        base_means, base_stds, z_feed, residual_obs, orch_obs, actions
      )
      residual_magnitude = self.residual_actor.residual_action_magnitude
      residual_weight = self.residual_actor.residual_weights_
      combined_mean = self.residual_actor.distribution.mean
      new_values = self.value_net(critic_obs)

      entropy = entropy_all[:original_batch_size] if augment else entropy_all

      with torch.no_grad():
        mu_batch = self.residual_actor.distribution.mean
        sigma_batch = self.residual_actor.distribution.stddev
        if augment:
          mu_batch = mu_batch[:original_batch_size]
          sigma_batch = sigma_batch[:original_batch_size]
          old_means_kl = old_action_means[:original_batch_size]
          old_stds_kl = old_action_stds[:original_batch_size]
        else:
          old_means_kl = old_action_means
          old_stds_kl = old_action_stds
        kl = torch.sum(
          torch.log(sigma_batch / old_stds_kl + 1e-5)
          + (old_stds_kl.pow(2) + (old_means_kl - mu_batch).pow(2))
          / (2.0 * sigma_batch.pow(2))
          - 0.5,
          dim=-1,
        )
        kl_mean = self._distributed_mean_scalar(float(kl.mean().item()))

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

      advantages_squeezed = advantages.squeeze(-1)
      ratio = torch.exp(new_log_probs - old_log_probs)
      surrogate = -advantages_squeezed * ratio
      surrogate_clipped = -advantages_squeezed * torch.clamp(
        ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.config.use_clipped_value_loss:
        value_clipped = target_values + torch.clamp(
          new_values - target_values, -self.config.clip_param, self.config.clip_param
        )
        value_loss_unclipped = (new_values - returns).pow(2)
        value_loss_clipped = (value_clipped - returns).pow(2)
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
      else:
        value_loss = (returns - new_values).pow(2).mean()

      symmetry_actor_loss = torch.zeros((), device=self.device)
      symmetry_critic_loss = torch.zeros((), device=self.device)
      if self._use_symmetry and self._action_mirror_fn is not None:
        if self.config.symmetry_loss_coef > 0.0:
          if augment:
            mu_original = combined_mean[:original_batch_size]
            mu_mirrored = combined_mean[original_batch_size:]
            symmetry_actor_loss = torch.nn.functional.mse_loss(
              mu_mirrored, self._action_mirror_fn(mu_original)
            )
          else:
            residual_m, orch_m = self._mirror_skill_inputs(residual_obs_raw, orch_obs_raw)
            mu_mirrored, _ = self.residual_actor.get_distribution_params(
              base_obs=(
                {n: {g: mirror_obs(raw_skill_obs[g], self._actor_sym_specs[g])
                     for g in skill.obs_groups}
                 for n, skill in self.residual_actor.base_branches.items()}
              ),
              residual_obs=residual_m,
              orch_obs=orch_m,
            )
            symmetry_actor_loss = torch.nn.functional.mse_loss(
              mu_mirrored, self._action_mirror_fn(combined_mean)
            )

        if self.config.symmetry_critic_coef > 0.0:
          if augment:
            val_original = new_values[:original_batch_size]
            val_mirrored = new_values[original_batch_size:]
            symmetry_critic_loss = torch.nn.functional.mse_loss(
              val_original, val_mirrored
            )
          else:
            mirrored_critic = mirror_obs(critic_obs.detach(), self._critic_sym_spec)
            val_mirrored = self.value_net(mirrored_critic)
            symmetry_critic_loss = torch.nn.functional.mse_loss(new_values, val_mirrored)

      if self.config.residual_weight_penalty_ramp_transitions > 0:
        ramp_fraction = min(
          1.0, self.global_step / self.config.residual_weight_penalty_ramp_transitions
        )
        rwp_coef = (
          self.config.residual_weight_penalty_coef_min
          + ramp_fraction
          * (self.config.residual_weight_penalty_coef - self.config.residual_weight_penalty_coef_min)
        )
      else:
        rwp_coef = self.config.residual_weight_penalty_coef

      loss = (
        surrogate_loss
        + self.config.value_loss_coef * value_loss
        - self.config.entropy_coef * entropy.mean()
        + self.config.symmetry_loss_coef * symmetry_actor_loss
        + self.config.symmetry_critic_coef * symmetry_critic_loss
        + self.config.residual_action_penalty_coef * residual_magnitude
        + rwp_coef * residual_weight
      )

      self.actor_optimizer.zero_grad()
      self.critic_optimizer.zero_grad()

      if not torch.isfinite(loss):
        continue

      loss.backward()
      self._distributed_average_optimizer_grads(self.actor_optimizer)
      self._distributed_average_optimizer_grads(self.critic_optimizer)
      for p in self.residual_actor.trainable_parameters():
        if p.grad is not None:
          p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
      for p in self.value_net.parameters():
        if p.grad is not None:
          p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
      torch.nn.utils.clip_grad_norm_(
        list(self.residual_actor.trainable_parameters()),
        max_norm=self.config.max_grad_norm,
      )
      torch.nn.utils.clip_grad_norm_(
        self.value_net.parameters(), max_norm=self.config.max_grad_norm
      )
      self.actor_optimizer.step()
      self.critic_optimizer.step()

      total_surrogate_loss += surrogate_loss.item()
      total_value_loss += value_loss.item()
      total_entropy += entropy.mean().item()
      total_kl += kl_mean
      total_residual_magnitude += residual_magnitude.item()
      total_residual_weight += residual_weight.item()
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
          total_residual_magnitude,
          total_residual_weight,
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
        total_residual_magnitude,
        total_residual_weight,
        total_symmetry_actor_loss,
        total_symmetry_critic_loss,
      ) = totals[:8]
      num_updates = int(totals[8])

    self.rollout_buffer.clear()

    denom = max(num_updates, 1)
    loss_dict = {
      "surrogate_loss": total_surrogate_loss / denom,
      "value_loss": total_value_loss / denom,
      "entropy": total_entropy / denom,
      "kl": total_kl / denom,
      "residual_magnitude": total_residual_magnitude / denom,
      "residual_weight": total_residual_weight / denom,
      "actor_learning_rate": self.actor_learning_rate,
      "critic_learning_rate": self.critic_learning_rate,
    }
    if self._use_symmetry:
      loss_dict["symmetry_actor_loss"] = total_symmetry_actor_loss / denom
      loss_dict["symmetry_critic_loss"] = total_symmetry_critic_loss / denom
    return loss_dict

  # ------------------------------------------------------------------ #
  # Evaluation
  # ------------------------------------------------------------------ #

  def _eval_get_action(self, normalized_obs: torch.Tensor) -> torch.Tensor:
    """Deterministic combined action. Routes from the cached obs dict."""
    # Roll the RMA skills' proprio windows with the current obs (mirrors the
    # collection loop). The first call sizes each window from the batch; without
    # it RMA branches keep an empty (0, W, D) buffer and the latent cat fails.
    for skill in self.residual_actor.base_branches.values():
      skill.update_state({g: self._cached_obs_dict[g] for g in skill.obs_groups})
    base_obs, residual_obs, orch_obs = self._normalized_skill_inputs(
      self._cached_obs_dict
    )
    return self.residual_actor.act_inference(base_obs, residual_obs, orch_obs)

  # ------------------------------------------------------------------ #
  # Checkpoint
  # ------------------------------------------------------------------ #

  def save(self, path: str | Path, **extra_state: Any) -> None:
    if "global_step" not in extra_state:
      raise ValueError("global_step must be provided in extra_state")

    state_dict = {
      "residual_actor_state_dict": self.residual_actor.state_dict(),
      "value_net_state_dict": self.value_net.state_dict(),
      "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
      "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
      "skill_normalizer_state_dicts": {
        g: n.state_dict() for g, n in self.skill_normalizers.items()
      },
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
    checkpoint = self._load_checkpoint(path)

    self.residual_actor.load_state_dict(checkpoint["residual_actor_state_dict"])
    self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
    self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    for g, norm in self.skill_normalizers.items():
      norm.load_state_dict(checkpoint["skill_normalizer_state_dicts"][g])
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

  def _orch_indices_for_group(self, group: str) -> list[int]:
    """Return the indices into orch obs that correspond to each term in `group`.

    The orchestrator obs is the union of all skill groups (deduplicated). For any
    sub-group, we can find which orch dims correspond to each of its terms by
    matching term names and accumulating offsets within the orch group.
    """
    obs_manager = self.env.observation_manager
    orch_group = self._orch_group
    orch_term_names = obs_manager._group_obs_term_names[orch_group]
    orch_term_dims = obs_manager._group_obs_term_dim[orch_group]

    # Build term → (start, end) offset map within orch obs.
    orch_offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, dims in zip(orch_term_names, orch_term_dims):
      dim = dims[0] if isinstance(dims, (tuple, list)) else int(dims)
      orch_offsets[name] = (offset, offset + dim)
      offset += dim

    branch_term_names = obs_manager._group_obs_term_names[group]
    indices: list[int] = []
    for term in branch_term_names:
      start, end = orch_offsets[term]
      indices.extend(range(start, end))
    return indices

  def export_onnx(self, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    was_training = self.residual_actor.training
    self.residual_actor.eval()
    for norm in self.skill_normalizers.values():
      norm.eval()

    obs_dim = self.env.observation_manager.group_obs_dim
    base_skills = list(self._base_skill_groups.items())  # [(name, group), ...]
    residual_group = self._residual_group
    orch_group = self._orch_group
    orch_dim = obs_dim[orch_group][0]

    # Per-group index arrays into the orch obs vector (computed once, baked in).
    base_indices = {name: self._orch_indices_for_group(g) for name, g in base_skills}
    residual_indices = self._orch_indices_for_group(residual_group)

    # RMA base skills carry a stateful proprio window: each becomes an extra
    # ONNX input/output pair so the deployment runtime feeds it back every step
    # (same convention as RmaPPO.export_onnx: "<name>_window" / "<name>_window_out").
    rma_skills = [
      (name, self.residual_actor.base_branches[name].window_size, len(base_indices[name]))
      for name, _ in base_skills
      if isinstance(self.residual_actor.base_branches[name], RmaBaseSkill)
    ]

    class _Wrapper(nn.Module):
      def __init__(self, actor, normalizers):
        super().__init__()
        self.actor = actor
        self.norms = nn.ModuleDict(normalizers)
        for name, idxs in base_indices.items():
          self.register_buffer(f"idx_{name}", torch.tensor(idxs, dtype=torch.long))
        self.register_buffer("idx_residual", torch.tensor(residual_indices, dtype=torch.long))
        self.rma_names = [name for name, _, _ in rma_skills]

      def forward(self, obs: torch.Tensor, *windows_in: torch.Tensor):
        # obs is raw orch obs (superset). Base branches normalize internally,
        # so feed them raw per-group slices as {name: {group: slice}}.
        base_obs = {
          name: {g: obs[:, getattr(self, f"idx_{name}")]}
          for name, g in base_skills
        }
        # Roll each RMA window with the current (raw) base-obs slice; the encoder
        # consumes the rolled window, the runtime feeds it back next step.
        precomputed_windows: dict[str, torch.Tensor] = {}
        windows_out: list[torch.Tensor] = []
        for i, name in enumerate(self.rma_names):
          base_slice = obs[:, getattr(self, f"idx_{name}")]
          w_out = torch.cat([windows_in[i][:, 1:, :], base_slice.unsqueeze(1)], dim=1)
          precomputed_windows[name] = w_out
          windows_out.append(w_out)
        res_obs = self.norms[residual_group](obs[:, self.idx_residual])
        orch_obs = self.norms[orch_group](obs)
        actions = self.actor.act_inference(
          base_obs, res_obs, orch_obs, precomputed_windows=precomputed_windows or None
        )
        return (actions, *windows_out)

    wrapper = _Wrapper(self.residual_actor, self.skill_normalizers).cpu()
    wrapper.eval()

    window_dummies = tuple(torch.zeros(1, w, d) for _, w, d in rma_skills)
    state_names = [f"{name}_window" for name, _, _ in rma_skills]
    torch.onnx.export(
      wrapper,
      (torch.zeros(1, orch_dim), *window_dummies),
      str(path),
      export_params=True,
      opset_version=18,
      input_names=["obs"] + state_names,
      output_names=["actions"] + [n + "_out" for n in state_names],
      # Embed weights in the single .onnx (no sidecar .onnx.data file).
      external_data=False,
    )

    if was_training:
      self.residual_actor.train()
      for norm in self.skill_normalizers.values():
        norm.train()
    self.residual_actor.to(self.device)
    for norm in self.skill_normalizers.values():
      norm.to(self.device)

    logger.success(f"ONNX exported: {path}")
    return path
