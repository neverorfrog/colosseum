"""Residual PPO networks: orchestrator gating net + composite residual actor.

See `residual_ppo_plan.md` §4-5 for the design. The orchestrator observes the
deduplicated union of all skill observations (NOT privileged critic info) and
outputs softmax blending weights. The residual actor blends frozen base skills
with a trainable residual via product-of-experts-style fusion.

Base skills are polymorphic (BaseSkill ABC). Three implementations:
  - MlpBaseSkill: plain PpoActor + normalizer (today's behaviour, latent_dim=0).
  - RmaBaseSkill: RMA-trained adaptation policy whose encoder latent ẑ feeds the
    residual branch.
  - ResidualBaseSkill: a frozen ResidualActor composite (residual-of-residual).
"""

from __future__ import annotations

import abc
from typing import Iterator, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from colosseum.algorithm.networks import Network
from colosseum.algorithm.networks.ppo_networks import PpoActor
from colosseum.algorithm.networks.proprio_encoder import ProprioWindowEncoder
from colosseum.algorithm.utils.normalization import EmpiricalNormalization
from colosseum.config.types.networks import PpoActorConfig


class _OdomHeadContainer(nn.Module):
  def forward(self, x: Tensor) -> Tensor:
    return self.net(x)


# ============================================================================ #
# BaseSkill ABC
# ============================================================================ #


class BaseSkill(nn.Module, abc.ABC):
  """A frozen pretrained skill: obs dict → action Gaussian.

  Owns its normalizer(s) and any internal state. May expose a latent for
  downstream consumers (e.g. the residual branch)."""

  obs_groups: tuple[str, ...]
  latent_dim: int = 0
  last_latent: Tensor | None = None

  @abc.abstractmethod
  def get_distribution_params(
    self, obs: dict[str, Tensor], window: Tensor | None = None
  ) -> tuple[Tensor, Tensor]: ...

  @abc.abstractmethod
  def freeze(self) -> None: ...

  @abc.abstractmethod
  def load_pretrained(self, checkpoint: dict) -> None: ...

  def update_state(self, obs: dict[str, Tensor]) -> None:
    pass

  def reset_state(self, env_ids: Tensor) -> None:
    pass


# ============================================================================ #
# MlpBaseSkill
# ============================================================================ #


class MlpBaseSkill(BaseSkill):
  """A plain frozen PpoActor with its own EmpiricalNormalization.

  Wraps today's behaviour: one obs group, no latent, no state."""

  def __init__(
    self,
    obs_group: str,
    obs_dim: int,
    action_dim: int,
    actor_config: PpoActorConfig,
    device: str | torch.device,
  ):
    super().__init__()
    self.obs_groups = (obs_group,)
    self.latent_dim = 0
    self._group = obs_group
    self.actor = PpoActor(obs_dim, action_dim, actor_config)
    self.norm = EmpiricalNormalization(shape=obs_dim, device=device)

  def get_distribution_params(
    self, obs: dict[str, Tensor], window: Tensor | None = None
  ) -> tuple[Tensor, Tensor]:
    return self.actor.get_distribution_params(self.norm(obs[self._group]))

  def freeze(self) -> None:
    for p in self.parameters():
      p.requires_grad_(False)
    self.eval()
    self.norm.eval()

  def load_pretrained(self, checkpoint: dict) -> None:
    if "actor_state_dict" in checkpoint:
      self.actor.load_state_dict(checkpoint["actor_state_dict"])
      self.norm.load_state_dict(checkpoint["actor_obs_normalizer_state_dict"])
    elif "residual_actor_state_dict" in checkpoint:
      import re

      r_state = checkpoint["residual_actor_state_dict"]
      base_prefix = None
      for k in r_state:
        m = re.match(r"base_branches\.([^.]+)\.actor\.", k)
        if m:
          base_prefix = f"base_branches.{m.group(1)}."
          break
      if base_prefix is None:
        raise KeyError(
          "Could not find any base_branches.*.actor key in residual_actor_state_dict"
        )
      remapped = {}
      for key, value in r_state.items():
        if key.startswith(base_prefix):
          m = re.match(r"base_branches\.[^.]+\.(actor\..+)", key)
          if m:
            remapped[m.group(1)] = value
      self.actor.load_state_dict(remapped, strict=False)
      norms = checkpoint.get("skill_normalizer_state_dicts", {})
      if self._group in norms:
        self.norm.load_state_dict(norms[self._group])
    else:
      raise KeyError(
        "Checkpoint has neither 'actor_state_dict' (plain PPO) nor "
        f"'residual_actor_state_dict' (ResidualPPO). Keys: {list(checkpoint.keys())[:10]}"
      )


# ============================================================================ #
# RmaBaseSkill
# ============================================================================ #


class RmaBaseSkill(BaseSkill):
  """RMA-trained adaptation policy whose encoder latent ẑ feeds the residual.

  Maintains a rolling proprioceptive window; the adaptation encoder consumes the
  raw window to produce ẑ, and the actor MLP consumes normalized obs ⊕ ẑ."""

  def __init__(
    self,
    actor_obs_group: str,
    actor_obs_dim: int,
    latent_dim: int,
    window_size: int,
    action_dim: int,
    actor_config: PpoActorConfig,
    term_name: str,
    device: str | torch.device,
  ):
    super().__init__()
    self.obs_groups = (actor_obs_group,)
    self.latent_dim = latent_dim
    self._group = actor_obs_group
    self._term_name = term_name
    self.window_size = window_size

    self.actor = PpoActor(actor_obs_dim + latent_dim, action_dim, actor_config)
    self.adapt_enc = ProprioWindowEncoder(actor_obs_dim, latent_dim, window_size)
    self.norm = EmpiricalNormalization(shape=actor_obs_dim, device=device)
    self.odom_head: nn.Module | None = None

    self.register_buffer(
      "_window",
      torch.zeros(0, window_size, actor_obs_dim, device=device),
      persistent=False,
    )

  def get_distribution_params(
    self, obs: dict[str, Tensor], window: Tensor | None = None
  ) -> tuple[Tensor, Tensor]:
    if window is not None:
      z = self.adapt_enc(window)
    else:
      z = self.adapt_enc(self._window)
    self.last_latent = z
    return self.actor.get_distribution_params(
      torch.cat([self.norm(obs[self._group]), z], -1)
    )

  def latent_for(self, window: Tensor) -> Tensor:
    return self.adapt_enc(window)

  def update_state(self, obs: dict[str, Tensor]) -> None:
    x = obs[self._group].detach()
    if self._window.shape[0] != x.shape[0]:
      self._window = x.unsqueeze(1).expand(-1, self.window_size, -1).clone()
      return
    self._window = torch.roll(self._window, -1, dims=1)
    self._window[:, -1, :] = x

  def reset_state(self, env_ids: Tensor) -> None:
    if env_ids is None or len(env_ids) == 0:
      return
    x = self._window[env_ids, -1, :]
    self._window[env_ids] = x.unsqueeze(1).expand(-1, self.window_size, -1)

  def freeze(self) -> None:
    for p in self.parameters():
      p.requires_grad_(False)
    self.eval()
    self.norm.eval()

  def load_pretrained(self, checkpoint: dict) -> None:
    self.actor.load_state_dict(checkpoint["actor_state_dict"])
    self.norm.load_state_dict(checkpoint["actor_obs_normalizer_state_dict"])
    term = checkpoint["rma_manager_state_dict"][self._term_name]["adaptation"]
    enc = {k[2:]: v for k, v in term.items() if k.startswith("0.")}
    self.adapt_enc.load_state_dict(enc)

    odom_state = {k[2:]: v for k, v in term.items() if k.startswith("1.")}
    if odom_state:
      if self.odom_head is None:
        hidden_dim = odom_state["net.0.weight"].shape[0]
        odom_net = nn.Sequential(
          nn.Linear(self.latent_dim, hidden_dim),
          nn.ELU(),
          nn.Linear(hidden_dim, 2),
        )
        self.odom_head = _OdomHeadContainer()
        self.odom_head.add_module("net", odom_net)
      self.odom_head.load_state_dict(odom_state)


# ============================================================================ #
# ResidualBaseSkill
# ============================================================================ #


class ResidualBaseSkill(BaseSkill):
  """A whole frozen ResidualActor (e.g. trained dribbling_residual) as one base
  skill. Reads its inner obs groups, owns its inner per-group normalizers, and
  outputs a single combined (mean, std). latent_dim=0 (no latent feed)."""

  def __init__(
    self,
    inner: "ResidualActor",
    inner_norms: dict[str, "EmpiricalNormalization"],
    obs_groups: tuple[str, ...],
    residual_group: str,
    orch_group: str,
    base_group_for: dict[str, str],
  ):
    super().__init__()
    self.obs_groups = obs_groups
    self.latent_dim = 0
    self.inner = inner
    self.norms = nn.ModuleDict(inner_norms)
    self._residual_group = residual_group
    self._orch_group = orch_group
    self._base_group_for = base_group_for

  def get_distribution_params(
    self, obs: dict[str, Tensor], window: Tensor | None = None
  ) -> tuple[Tensor, Tensor]:
    inner_base = {n: {g: obs[g]} for n, g in self._base_group_for.items()}
    residual = self.norms[self._residual_group](obs[self._residual_group])
    orch = self.norms[self._orch_group](obs[self._orch_group])
    return self.inner.get_distribution_params(inner_base, residual, orch)

  def freeze(self) -> None:
    for p in self.parameters():
      p.requires_grad_(False)
    self.eval()
    for n in self.norms.values():
      n.eval()

  def load_pretrained(self, checkpoint: dict) -> None:
    import re

    state_dict = checkpoint["residual_actor_state_dict"]
    remapped = {}
    for key, value in state_dict.items():
      new_key = key
      m = re.match(
        r"(base_branches\.[^.]+)\.(backbone|mean_head|std|min_noise_std)(\..*|$)", key
      )
      if m:
        new_key = f"{m.group(1)}.actor.{m.group(2)}{m.group(3)}"
      remapped[new_key] = value

    self.inner.load_state_dict(remapped, strict=False)
    for g, n in self.norms.items():
      n.load_state_dict(checkpoint["skill_normalizer_state_dicts"][g])
    for name, inner_skill in self.inner.base_branches.items():
      g = self._base_group_for[name]
      inner_skill.norm.load_state_dict(checkpoint["skill_normalizer_state_dicts"][g])


# ============================================================================ #
# Orchestrator
# ============================================================================ #


class Orchestrator(Network):
  """MLP gating network: observes the union of all skill observations, outputs blending weights.

  Input: the orchestrator observation group (deduplicated union of all skill
  observations, no privileged info — so it is deployable).
  Output: softmax-normalized weights [w_0, ..., w_{num_skills-1}] over all skills.

  Architecture:
      obs -> backbone (MLP) -> weight_head -> softmax -> [B, num_skills]
  """

  def __init__(
    self,
    input_dim: int,
    num_skills: int,
    hidden_layers: Sequence[int],
    activation: str,
  ):
    super().__init__(
      input_dim=input_dim,
      hidden_layers=hidden_layers,
      activation_name=activation,
    )
    self.num_skills = num_skills
    self.weight_head = nn.Linear(int(self.last_dim), num_skills)

    nn.init.orthogonal_(self.weight_head.weight, gain=1.0)
    nn.init.zeros_(self.weight_head.bias)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    """Return softmax blending weights [B, num_skills]."""
    logits = self.weight_head(self.backbone(obs))
    return nn.functional.softmax(logits, dim=-1)


# ============================================================================ #
# ResidualActor
# ============================================================================ #


class ResidualActor(nn.Module):
  """Composite actor: frozen base skills + trainable residual, blended by an orchestrator.

  A composite (not a single MLP), so it extends nn.Module directly. Each base
  skill and the residual is a participant; the orchestrator outputs per-skill
  weights that fuse their Gaussians into one action distribution (PoE-style).

  Base skills are polymorphic (BaseSkill ABC). Each may optionally expose an
  adaptation-encoder latent ẑ that feeds the residual branch's input (via
  ``latent_feed_skills``).

  Skill-ordering invariant: base branches (ModuleDict, insertion-ordered) come
  first, the residual is appended *last*. The orchestrator's output columns must
  follow the same order, so the residual is always the last weight column.
  Checkpoint loading and freezing are done by ResidualPPO after construction.
  """

  def __init__(
    self,
    base_skills: Sequence[BaseSkill],
    base_skill_names: Sequence[str],
    residual_obs_dim: int,
    residual_config: PpoActorConfig,
    action_dim: int,
    orchestrator_obs_dim: int,
    orchestrator_hidden_layers: Sequence[int],
    orchestrator_activation: str,
    latent_feed_skills: tuple[str, ...] = (),
  ):
    super().__init__()

    self.base_branches = nn.ModuleDict(
      {name: skill for name, skill in zip(base_skill_names, base_skills)}
    )

    feed_dim = sum(self.base_branches[n].latent_dim for n in latent_feed_skills)
    self.residual_branch = PpoActor(
      residual_obs_dim + feed_dim, action_dim, residual_config
    )

    self._latent_feed_skills = latent_feed_skills
    self._feed_dim = feed_dim

    self.num_skills = len(self.base_branches) + 1
    self.orchestrator = Orchestrator(
      input_dim=orchestrator_obs_dim,
      num_skills=self.num_skills,
      hidden_layers=orchestrator_hidden_layers,
      activation=orchestrator_activation,
    )

    self.distribution: torch.distributions.Normal | None = None
    self.residual_action_magnitude: torch.Tensor = torch.zeros(())
    self.residual_weights_: torch.Tensor = torch.zeros(())

    self._last_base_means: list[Tensor] = []
    self._last_base_stds: list[Tensor] = []
    self._last_z_feed: Tensor = torch.zeros(0)

  # ----------------------------------------------------------------------- #
  # Forward / distribution
  # ----------------------------------------------------------------------- #

  @staticmethod
  def combine_skills(
    means: torch.Tensor, stds: torch.Tensor, weights: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted-blend fusion of per-skill Gaussians (ported from rsl_rl CCRL).

    Scales each skill by w_i / sigma_i (NOT 1/sigma^2 — this is a weighted blend
    of stds, not a strict precision-weighted product of experts).

    Args:
      means: [B, N, A] per-skill action means.
      stds:  [B, N, A] per-skill action stds.
      weights: [B, N] per-skill blend weights (softmax, sum to 1).

    Returns:
      (combined_mean [B, A], combined_std [B, A]).
    """
    stds = stds + 1e-2
    sw = weights.unsqueeze(-1) / stds
    sw_sum = sw.sum(dim=1)
    combined_std = 1.0 / sw_sum
    combined_mean = combined_std * (means * sw).sum(dim=1)
    return combined_mean, combined_std

  def _collect_latents(self) -> Tensor:
    """Collect latents from fed skills into a concatenated [B, feed_dim]."""
    latents: list[Tensor] = []
    for name in self._latent_feed_skills:
      z = self.base_branches[name].last_latent
      if z is None:
        raise RuntimeError(
          f"RmaBaseSkill '{name}' has no last_latent — call "
          "get_distribution_params first."
        )
      latents.append(z)
    return (
      torch.cat(latents, dim=-1)
      if latents
      else torch.zeros(1, 0, device=self.orchestrator.weight_head.weight.device)
    )

  def update_distribution(
    self,
    base_obs: dict[str, torch.Tensor],
    residual_obs: torch.Tensor,
    orch_obs: torch.Tensor,
    precomputed_windows: dict[str, Tensor] | None = None,
  ) -> None:
    """Central forward pass (collection time): run all branches + orchestrator, set self.distribution.

    precomputed_windows: if provided, each RMA skill uses this external window
    instead of its internal buffer (used for ONNX export)."""
    means: list[torch.Tensor] = []
    stds: list[torch.Tensor] = []

    for name, branch in self.base_branches.items():
      window = precomputed_windows.get(name) if precomputed_windows else None
      mean, std = branch.get_distribution_params(base_obs[name], window=window)
      means.append(mean)
      stds.append(std)
    z_feed = self._collect_latents()
    self._last_base_means = list(means)
    self._last_base_stds = list(stds)
    self._last_z_feed = z_feed

    res_input = (
      residual_obs
      if self._feed_dim == 0
      else torch.cat([residual_obs, z_feed.expand(residual_obs.shape[0], -1)], dim=-1)
    )
    res_mean, res_std = self.residual_branch.get_distribution_params(res_input)
    means.append(res_mean)
    stds.append(res_std)

    weights = self.orchestrator(orch_obs)
    means_stack = torch.stack(means, dim=1)
    stds_stack = torch.stack(stds, dim=1)
    combined_mean, combined_std = self.combine_skills(means_stack, stds_stack, weights)
    self.distribution = torch.distributions.Normal(combined_mean, combined_std)

    self.residual_action_magnitude = torch.norm(res_mean, p=2, dim=-1).mean()
    self.residual_weights_ = weights[:, -1].abs().mean()

  def update_distribution_cached(
    self,
    base_means: list[Tensor],
    base_stds: list[Tensor],
    z_feed: Tensor,
    residual_obs: torch.Tensor,
    orch_obs: torch.Tensor,
  ) -> None:
    """Learning-time forward: base params + z come pre-computed from the buffer."""
    res_input = (
      residual_obs if self._feed_dim == 0 else torch.cat([residual_obs, z_feed], dim=-1)
    )
    res_mean, res_std = self.residual_branch.get_distribution_params(res_input)

    means = base_means + [res_mean]
    stds = base_stds + [res_std]

    weights = self.orchestrator(orch_obs)
    means_stack = torch.stack(means, dim=1)
    stds_stack = torch.stack(stds, dim=1)
    combined_mean, combined_std = self.combine_skills(means_stack, stds_stack, weights)
    self.distribution = torch.distributions.Normal(combined_mean, combined_std)

    self.residual_action_magnitude = torch.norm(res_mean, p=2, dim=-1).mean()
    self.residual_weights_ = weights[:, -1].abs().mean()

  def act_with_log_prob(
    self,
    base_obs: dict[str, torch.Tensor],
    residual_obs: torch.Tensor,
    orch_obs: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample actions; return (actions, log_probs, combined_mean, combined_std)."""
    self.update_distribution(base_obs, residual_obs, orch_obs)
    assert self.distribution is not None
    actions = self.distribution.sample()
    log_probs = self.distribution.log_prob(actions).sum(dim=-1)
    return actions, log_probs, self.distribution.mean, self.distribution.stddev

  def evaluate(
    self,
    base_obs: dict[str, torch.Tensor],
    residual_obs: torch.Tensor,
    orch_obs: torch.Tensor,
    actions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-evaluate stored actions for the PPO update; sets penalty side effects."""
    self.update_distribution(base_obs, residual_obs, orch_obs)
    assert self.distribution is not None
    log_prob = self.distribution.log_prob(actions).sum(dim=-1)
    entropy = self.distribution.entropy().sum(dim=-1)
    return log_prob, entropy

  def evaluate_cached(
    self,
    base_means: list[Tensor],
    base_stds: list[Tensor],
    z_feed: Tensor,
    residual_obs: torch.Tensor,
    orch_obs: torch.Tensor,
    actions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-evaluate actions from cached base params (learning-time)."""
    self.update_distribution_cached(
      base_means, base_stds, z_feed, residual_obs, orch_obs
    )
    assert self.distribution is not None
    log_prob = self.distribution.log_prob(actions).sum(dim=-1)
    entropy = self.distribution.entropy().sum(dim=-1)
    return log_prob, entropy

  def act_inference(
    self,
    base_obs: dict[str, torch.Tensor],
    residual_obs: torch.Tensor,
    orch_obs: torch.Tensor,
  ) -> torch.Tensor:
    """Deterministic combined mean for deployment."""
    self.update_distribution(base_obs, residual_obs, orch_obs)
    assert self.distribution is not None
    return self.distribution.mean

  def get_distribution_params(
    self,
    base_obs: dict[str, torch.Tensor],
    residual_obs: torch.Tensor,
    orch_obs: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return combined (mean, std) for KL divergence tracking."""
    self.update_distribution(base_obs, residual_obs, orch_obs)
    assert self.distribution is not None
    return self.distribution.mean, self.distribution.stddev

  # ----------------------------------------------------------------------- #
  # Lifecycle management (called by ResidualPPO)
  # ----------------------------------------------------------------------- #

  def freeze_base_skills(self) -> None:
    """Disable grad on all base branch parameters."""
    for branch in self.base_branches.values():
      branch.freeze()

  def init_orchestrator_bias(self, favored_logit: float) -> None:
    """Bias the orchestrator toward the frozen base skills at init.

    Base-skill columns get `favored_logit`, the residual (last column) gets 0.0,
    so the residual starts near-off (e.g. softmax([4,0]) ~= [0.98, 0.02]).
    """
    bias = torch.zeros(self.num_skills)
    bias[:-1] = favored_logit
    with torch.no_grad():
      self.orchestrator.weight_head.bias.copy_(bias)

  def trainable_parameters(self) -> Iterator[nn.Parameter]:
    """Yield residual branch + orchestrator parameters (frozen base excluded)."""
    yield from self.residual_branch.parameters()
    yield from self.orchestrator.parameters()

  @property
  def last_base_means(self) -> list[Tensor]:
    return self._last_base_means

  @property
  def last_base_stds(self) -> list[Tensor]:
    return self._last_base_stds

  @property
  def last_z_feed(self) -> Tensor:
    return self._last_z_feed
