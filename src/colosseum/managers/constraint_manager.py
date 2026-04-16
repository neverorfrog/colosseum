"""Constraints-as-Terminations (CaT) manager for mjlab.

Implements the algorithm from:
  "Constraints as Terminations: Reinforcement Learning with Soft Constraints"
  https://arxiv.org/abs/2403.18765

Constraint violations are converted to stochastic episode termination
probabilities rather than Lagrangian penalties. At each step:
  1. Each constraint function returns a raw violation value (positive = violated).
  2. Violations are normalized by a Polyak-averaged running maximum.
  3. A per-env termination probability is derived from the normalized violation.
  4. Rewards are scaled by (1 - p) and episodes are terminated with Bernoulli(p).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch
from mjlab.managers.manager_base import ManagerBase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass
class ConstraintTermCfg:
  """Configuration for a single CaT constraint term.

  Example::

      joint_torque = ConstraintTermCfg(
          func=constraints.joint_torque,
          max_p=1.0,
          params={"limit": 40.0, "asset_cfg": SceneEntityCfg("robot")},
      )
  """

  func: Callable[..., torch.Tensor]
  """Constraint function. Must return a tensor of shape [num_envs] or
  [num_envs, num_dims] where positive values indicate violation magnitude."""

  max_p: float
  """Maximum termination probability for this constraint.
  Use 1.0 for hard constraints (never violate), 0.0–0.5 for soft constraints."""

  params: dict[str, Any] = field(default_factory=dict)
  """Extra keyword arguments forwarded to ``func`` at every call."""


class CaT:
  """Core CaT algorithm: converts raw constraint violations to termination probabilities.

  For each constraint i, at each step (Eq. 6 in paper):

    p_i = max_p_i * clip(c_i^+ / c_i^max, 0, 1)

  where c_i^max is updated via Polyak averaging (Eq. 7):

    c_i^max <- tau * c_i^max + (1 - tau) * max_{batch}(c_i^+)

  The per-env termination probability is the max over all constraints:

    delta = max_i p_i
  """

  def __init__(self, tau: float = 0.95, min_p: float = 0.0):
    self.tau = tau
    self.min_p = min_p
    self.running_maxes: dict[str, torch.Tensor] = {}
    self.probs: dict[str, torch.Tensor] = {}

  def reset(self) -> None:
    """Clear per-step state (call before each compute pass)."""
    self.probs.clear()

  def add(self, name: str, constraint: torch.Tensor, max_p: float) -> None:
    """Process one constraint and store its per-env termination probability.

    Args:
      name: Unique constraint name (used as dict key).
      constraint: Raw violation tensor [num_envs] or [num_envs, num_dims].
        Values > 0 are violations; values <= 0 are safe.
      max_p: Maximum termination probability for this constraint.
    """
    if not torch.is_floating_point(constraint):
      constraint = constraint.float()
    if constraint.ndim == 1:
      constraint = constraint.unsqueeze(1)  # [num_envs, 1]

    # Positive part: only violations matter
    violation = constraint.clamp(min=0.0)

    # Polyak update of running maximum (per output dimension)
    batch_max = violation.max(dim=0, keepdim=True).values.clamp(min=1e-6)
    if name in self.running_maxes:
      self.running_maxes[name].mul_(self.tau).add_((1.0 - self.tau) * batch_max)
    else:
      self.running_maxes[name] = batch_max.clone()

    # Normalized probability (Eq. 6)
    probs = torch.zeros_like(constraint)
    mask = constraint > 0.0
    if mask.any():
      normalized = violation / self.running_maxes[name].expand_as(violation)
      probs[mask] = self.min_p + normalized[mask].clamp(0.0, 1.0) * (max_p - self.min_p)
    self.probs[name] = probs

  def get_probs(self) -> torch.Tensor:
    """Return per-env max termination probability across all constraints.

    Returns:
      Float tensor of shape [num_envs] with values in [0, 1].
    """
    if not self.probs:
      raise RuntimeError("No constraints have been added yet.")
    return torch.cat(list(self.probs.values()), dim=1).max(dim=1).values


class ConstraintManager(ManagerBase):
  """Manages a collection of CaT constraint terms.

  Usage::

      # In ConstraintBasedEnvCfg:
      constraints = {
          "joint_torque": ConstraintTermCfg(
              func=constraints.joint_torque, max_p=1.0,
              params={"limit": 40.0, "asset_cfg": SceneEntityCfg("robot")},
          ),
          "base_orientation": ConstraintTermCfg(
              func=constraints.base_orientation, max_p=0.25,
              params={"limit": 0.5, "asset_cfg": SceneEntityCfg("robot")},
          ),
      }
  """

  def __init__(
    self,
    cfg: dict[str, ConstraintTermCfg],
    env: ManagerBasedRlEnv,
    tau: float = 0.95,
    min_p: float = 0.0,
  ):
    self.cfg = deepcopy(cfg)
    self._tau = tau
    self._min_p = min_p
    super().__init__(env=env)

    # Episode tracking: violation count and probability sum per env per term
    self._episode_violation_sums: dict[str, torch.Tensor] = {
      name: torch.zeros(self.num_envs, device=self.device)
      for name in self._term_names
    }
    self._episode_prob_sums: dict[str, torch.Tensor] = {
      name: torch.zeros(self.num_envs, device=self.device)
      for name in self._term_names
    }

  @property
  def active_terms(self) -> list[str]:
    return self._term_names

  def _prepare_terms(self) -> None:
    self._term_names: list[str] = []
    self._term_cfgs: list[ConstraintTermCfg] = []
    self.cat = CaT(tau=self._tau, min_p=self._min_p)

    for name, term_cfg in self.cfg.items():
      if term_cfg is None:
        continue
      if not isinstance(term_cfg, ConstraintTermCfg):
        raise TypeError(
          f"Term '{name}' must be a ConstraintTermCfg, got {type(term_cfg).__name__}."
        )
      # Resolve SceneEntityCfg params and instantiate class-based funcs
      self._resolve_common_term_cfg(name, term_cfg)
      self._term_names.append(name)
      self._term_cfgs.append(term_cfg)

  def compute(self) -> torch.Tensor:
    """Evaluate all constraint terms and return per-env termination probability.

    Returns:
      Float tensor [num_envs] in [0, 1]. Zero means no constraint is violated.
    """
    self.cat.reset()
    for name, term_cfg in zip(self._term_names, self._term_cfgs):
      raw = term_cfg.func(self._env, **term_cfg.params)
      self.cat.add(name, raw, term_cfg.max_p)

    cstr_prob = self.cat.get_probs()

    # Accumulate episode statistics for logging
    for name in self._term_names:
      per_env_max_prob = self.cat.probs[name].max(dim=1).values
      self._episode_violation_sums[name].add_(per_env_max_prob.gt(0.0).float())
      self._episode_prob_sums[name].add_(per_env_max_prob)

    return cstr_prob

  def reset(self, env_ids: torch.Tensor) -> dict[str, Any]:
    """Log per-episode constraint statistics and clear buffers for env_ids.

    Returns:
      Dict with keys ``Episode_Constraint_violation/<name>`` (% of steps violated)
      and ``Episode_Constraint_probability/<name>`` (mean termination probability).
    """
    extras: dict[str, Any] = {}
    ep_len = self._env.episode_length_buf[env_ids].float().clamp(min=1.0)

    for name in self._term_names:
      violation_rate = (self._episode_violation_sums[name][env_ids] / ep_len).mean()
      prob_mean = (self._episode_prob_sums[name][env_ids] / ep_len).mean()
      extras[f"Episode_Constraint_violation/{name}"] = violation_rate * 100.0
      extras[f"Episode_Constraint_probability/{name}"] = prob_mean
      self._episode_violation_sums[name][env_ids] = 0.0
      self._episode_prob_sums[name][env_ids] = 0.0

    return extras
