"""ConstraintAbstractionBasedEnv — AbstractionBasedEnv with CaT constraints.

Combines the abstraction manager (Sokoban plan tracking) with
Constraints-as-Terminations so constraint terms can be applied alongside
abstraction-driven rewards without duplicating env infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from loguru import logger
from mjlab.envs.types import VecEnvObs, VecEnvStepReturn

from colosseum.envs.constraint_based_env import ConstraintBasedEnv, ConstraintBasedEnvCfg
from colosseum.managers.abstraction_manager import (
  AbstractionManager,
  AbstractionTermCfg,
  NullAbstractionManager,
)


@dataclass(kw_only=True)
class ConstraintAbstractionBasedEnvCfg(ConstraintBasedEnvCfg):
  """Configuration for ConstraintAbstractionBasedEnv."""

  class_type: ClassVar[type[ConstraintAbstractionBasedEnv]]  # set after class definition
  abstractions: dict[str, AbstractionTermCfg]


class ConstraintAbstractionBasedEnv(ConstraintBasedEnv):
  """ConstraintBasedEnv extended with an abstraction manager.

  Execution order per step:
    1. Physics + rewards (ConstraintBasedEnv.step → ManagerBasedRlEnv.step)
    2. Constraint scaling and stochastic terminations (ConstraintBasedEnv.step)
    3. Abstraction state update for the NEXT step (this class)

  The abstraction computes after rewards so reward functions always see the
  state that was current when the action was applied, matching the behaviour
  of AbstractionBasedEnv.
  """

  cfg: ConstraintAbstractionBasedEnvCfg  # type: ignore[override]

  def get_observations(self) -> dict:
    return self.observation_manager.compute()

  def load_managers(self) -> None:
    if self.cfg.abstractions:
      self.abstraction_manager = AbstractionManager(cfg=self.cfg.abstractions, env=self)
    else:
      self.abstraction_manager = NullAbstractionManager(cfg={}, env=self)
    super().load_managers()

  def setup_manager_visualizers(self) -> None:
    super().setup_manager_visualizers()
    logger.info("Setting up abstraction manager visualizer...")
    self.manager_visualizers["abstraction_manager"] = self.abstraction_manager

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    super()._reset_idx(env_ids)
    info = self.abstraction_manager.reset(env_ids)
    self.extras["log"].update(info)

  def reset(
    self,
    *,
    seed: int | None = None,
    env_ids: torch.Tensor | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[VecEnvObs, dict]:
    obs, info = super().reset(seed=seed, options=options)
    all_env_ids = torch.arange(self.num_envs, device=self.device)
    self.abstraction_manager.reset(all_env_ids)
    return obs, info

  def step(self, action: torch.Tensor) -> VecEnvStepReturn:
    # ConstraintBasedEnv.step: physics + rewards + constraint scaling/termination
    obs, rewards, terminated, truncated, infos = super().step(action)
    # Update abstraction state for the next policy step
    self.abstraction_manager.compute(dt=self.step_dt)
    return obs, rewards, terminated, truncated, infos


ConstraintAbstractionBasedEnvCfg.class_type = ConstraintAbstractionBasedEnv
