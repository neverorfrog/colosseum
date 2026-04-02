from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from loguru import logger
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.types import VecEnvObs, VecEnvStepReturn

from colosseum.managers.abstraction_manager import (
  AbstractionManager,
  AbstractionTermCfg,
  NullAbstractionManager,
)


@dataclass(kw_only=True)
class AbstractionBasedEnvCfg(ManagerBasedRlEnvCfg):
  """Configuration for AbstractionBasedEnv."""

  class_type: ClassVar[type] = None  # set after class definition
  abstractions: dict[str, AbstractionTermCfg]


class AbstractionBasedEnv(ManagerBasedRlEnv):
  """An environment that uses abstraction-based guidance."""

  cfg: AbstractionBasedEnvCfg

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
    obs, rewards, terminated, truncated, infos = super().step(action)
    self.abstraction_manager.compute(dt=self.step_dt)
    return obs, rewards, terminated, truncated, infos


AbstractionBasedEnvCfg.class_type = AbstractionBasedEnv
