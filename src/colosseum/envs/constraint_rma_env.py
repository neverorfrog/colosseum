"""ConstraintRmaEnv — RmaBasedEnv with CaT constraint-based terminations.

Combines RMA encoder support with the Constraints-as-Terminations algorithm.
The step() override applies reward scaling and Bernoulli-sampled terminations
from the constraint manager after the RMA update, so encoder state is always
consistent with the post-constraint observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import torch
from mjlab.envs.types import VecEnvStepReturn

from colosseum.envs.rma_based_env import RmaBasedEnv, RmaBasedEnvCfg
from colosseum.managers.constraint_manager import ConstraintManager, ConstraintTermCfg


@dataclass(kw_only=True)
class ConstraintRmaEnvCfg(RmaBasedEnvCfg):
  """Configuration for ConstraintRmaEnv."""

  class_type: ClassVar[type[ConstraintRmaEnv]]  # set after class definition

  constraints: dict[str, ConstraintTermCfg] = field(default_factory=dict)
  """Constraint terms. An empty dict means no constraints are enforced."""


class ConstraintRmaEnv(RmaBasedEnv):
  """RmaBasedEnv with CaT constraint-based terminations.

  If ``cfg.constraints`` is non-empty, a ``ConstraintManager`` is created
  during ``load_managers()``. The ``step()`` override applies reward scaling
  and Bernoulli-sampled terminations from the constraint probabilities after
  the RMA manager update.
  """

  cfg: ConstraintRmaEnvCfg  # type: ignore[override]

  def load_managers(self) -> None:
    super().load_managers()
    if self.cfg.constraints:
      self.constraint_manager = ConstraintManager(
        cfg=self.cfg.constraints,
        env=self,
      )

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    super()._reset_idx(env_ids)
    if hasattr(self, "constraint_manager"):
      info = self.constraint_manager.reset(env_ids)
      self.extras["log"].update(info)

  def step(self, action: torch.Tensor) -> VecEnvStepReturn:
    obs, rewards, terminated, truncated, extras = super().step(action)

    if hasattr(self, "constraint_manager"):
      cstr_prob = self.constraint_manager.compute()
      rewards = torch.clip(rewards * (1.0 - cstr_prob), min=0.0)
      cstr_terminated = torch.bernoulli(cstr_prob).bool()
      terminated = terminated | cstr_terminated

    return obs, rewards, terminated, truncated, extras


ConstraintRmaEnvCfg.class_type = ConstraintRmaEnv
