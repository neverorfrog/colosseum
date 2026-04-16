"""Environment with Constraints-as-Terminations (CaT) support.

Extends ManagerBasedRlEnv with a ConstraintManager that converts constraint
violations into stochastic episode terminations (Option A: Bernoulli sampling).

Step behaviour:
  1. Run the normal env step (physics, terminations, rewards).
  2. Compute per-env constraint termination probability cstr_prob ∈ [0, 1].
  3. Scale rewards by (1 - cstr_prob) and clip to [0, ∞).
  4. Sample binary terminated_constraint ~ Bernoulli(cstr_prob) and OR with
     the hard terminations from the termination manager.

This keeps the (obs, rewards, terminated: bool, truncated: bool, extras)
interface unchanged so no PPO modifications are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.types import VecEnvStepReturn

from colosseum.managers.constraint_manager import ConstraintManager, ConstraintTermCfg


@dataclass(kw_only=True)
class ConstraintBasedEnvCfg(ManagerBasedRlEnvCfg):
  """Configuration for ConstraintBasedEnv."""

  class_type: ClassVar[type[ConstraintBasedEnv]]  # set after class definition

  constraints: dict[str, ConstraintTermCfg] = field(default_factory=dict)
  """Constraint terms. An empty dict means no constraints are enforced."""


class ConstraintBasedEnv(ManagerBasedRlEnv):
  """ManagerBasedRlEnv with CaT constraint-based terminations.

  If ``cfg.constraints`` is non-empty, a ``ConstraintManager`` is created
  during ``load_managers()``. The ``step()`` override applies reward scaling
  and Bernoulli-sampled terminations from the constraint probabilities.
  """

  cfg: ConstraintBasedEnvCfg  # type: ignore[override]

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
      # Termination probability for each env: float in [0, 1]
      cstr_prob = self.constraint_manager.compute()

      # Scale rewards down proportionally to violation severity (Algorithm 1, paper)
      rewards = torch.clip(rewards * (1.0 - cstr_prob), min=0.0)

      # Option B: return float terminated so GAE discount softens without resetting the env.
      # Hard resets (terminated=True) stay at 1.0; constraint probability fills the rest.
      terminated_float = cstr_prob.clone()
      terminated_float[terminated] = 1.0
      terminated = terminated_float

    return obs, rewards, terminated, truncated, extras


ConstraintBasedEnvCfg.class_type = ConstraintBasedEnv
