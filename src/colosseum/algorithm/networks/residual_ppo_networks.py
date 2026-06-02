"""Residual PPO networks: orchestrator gating net + composite residual actor.

See `residual_ppo_plan.md` §4-5 for the design. The orchestrator observes the
deduplicated union of all skill observations (NOT privileged critic info) and
outputs softmax blending weights. The residual actor blends frozen base skills
with a trainable residual via product-of-experts-style fusion.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import torch
import torch.nn as nn

from colosseum.algorithm.networks import Network
from colosseum.algorithm.networks.ppo_networks import PpoActor
from colosseum.config.types.networks import PpoActorConfig


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


class ResidualActor(nn.Module):
  """Composite actor: frozen base skills + trainable residual, blended by an orchestrator.

  A composite (not a single MLP), so it extends nn.Module directly. Each base
  skill and the residual is a PpoActor; the orchestrator outputs per-skill
  weights that fuse their Gaussians into one action distribution (PoE-style).

  Skill-ordering invariant: base branches (ModuleDict, insertion-ordered) come
  first, the residual is appended *last*. The orchestrator's output columns must
  follow the same order, so the residual is always the last weight column.
  Checkpoint loading and freezing are done by ResidualPPO after construction.
  """

  def __init__(
    self,
    base_skill_obs_dims: dict[str, int],
    base_skill_configs: dict[str, PpoActorConfig],
    residual_obs_dim: int,
    residual_config: PpoActorConfig,
    action_dim: int,
    orchestrator_obs_dim: int,
    orchestrator_hidden_layers: Sequence[int],
    orchestrator_activation: str,
  ):
    super().__init__()

    # Base branches first (insertion order == orchestrator column order).
    self.base_branches = nn.ModuleDict(
      {
        name: PpoActor(base_skill_obs_dims[name], action_dim, base_skill_configs[name])
        for name in base_skill_configs
      }
    )
    self.residual_branch = PpoActor(residual_obs_dim, action_dim, residual_config)

    self.num_skills = len(self.base_branches) + 1  # base skills + residual
    self.orchestrator = Orchestrator(
      input_dim=orchestrator_obs_dim,
      num_skills=self.num_skills,
      hidden_layers=orchestrator_hidden_layers,
      activation=orchestrator_activation,
    )

    self.distribution: torch.distributions.Normal | None = None
    # Differentiable penalty side effects (set by update_distribution). Kept as
    # live tensors with grad — only .item() them when logging.
    self.residual_action_magnitude: torch.Tensor = torch.zeros(())
    self.residual_weights_: torch.Tensor = torch.zeros(())

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
    sw = weights.unsqueeze(-1) / stds  # [B, N, A]
    sw_sum = sw.sum(dim=1)  # [B, A]
    combined_std = 1.0 / sw_sum
    combined_mean = combined_std * (means * sw).sum(dim=1)
    return combined_mean, combined_std

  def update_distribution(
    self,
    base_obs: dict[str, torch.Tensor],
    residual_obs: torch.Tensor,
    orch_obs: torch.Tensor,
  ) -> None:
    """Central forward pass: run all branches + orchestrator, set self.distribution."""
    means: list[torch.Tensor] = []
    stds: list[torch.Tensor] = []
    # Base branches first, in ModuleDict insertion order.
    for name, branch in self.base_branches.items():
      mean, std = branch.get_distribution_params(base_obs[name])
      means.append(mean)
      stds.append(std)
    # Residual appended last (skill-ordering invariant).
    res_mean, res_std = self.residual_branch.get_distribution_params(residual_obs)
    means.append(res_mean)
    stds.append(res_std)

    weights = self.orchestrator(orch_obs)  # [B, num_skills], post-softmax

    means_stack = torch.stack(means, dim=1)  # [B, num_skills, A]
    stds_stack = torch.stack(stds, dim=1)  # [B, num_skills, A]
    combined_mean, combined_std = self.combine_skills(means_stack, stds_stack, weights)
    self.distribution = torch.distributions.Normal(combined_mean, combined_std)

    # Differentiable penalty quantities (residual is the last column).
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

  def load_base_skill(self, name: str, state_dict: dict) -> None:
    """Load a PPO actor checkpoint into one base branch."""
    self.base_branches[name].load_state_dict(state_dict)

  def freeze_base_skills(self) -> None:
    """Disable grad on all base branch parameters."""
    for branch in self.base_branches.values():
      for param in branch.parameters():
        param.requires_grad = False

  def init_orchestrator_bias(self, favored_logit: float) -> None:
    """Bias the orchestrator toward the frozen base skills at init.

    Base-skill columns get `favored_logit`, the residual (last column) gets 0.0,
    so the residual starts near-off (e.g. softmax([4,0]) ~= [0.98, 0.02]).
    """
    bias = torch.zeros(self.num_skills)
    bias[:-1] = favored_logit  # base skills favored; residual (last) stays 0
    with torch.no_grad():
      self.orchestrator.weight_head.bias.copy_(bias)

  def trainable_parameters(self) -> Iterator[nn.Parameter]:
    """Yield residual branch + orchestrator parameters (frozen base excluded)."""
    yield from self.residual_branch.parameters()
    yield from self.orchestrator.parameters()
