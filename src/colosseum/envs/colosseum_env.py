"""ColosseumEnv — unified environment for all colosseum tasks.

Conditionally loads an RmaManager based on config:
  - rma_manager → cfg.encoders is non-empty

Step order (per-step):
  1. Physics + rewards (ManagerBasedRlEnv.step)
  2. RMA encoder update (after obs are ready)

Reset order (_reset_idx):
  1. ManagerBasedRlEnv._reset_idx (standard reset)
  2. RMA reset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, ClassVar

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.types import VecEnvObs, VecEnvStepReturn

from colosseum.managers.rma_manager import RmaManager, RmaTermCfg


@dataclass(kw_only=True)
class ColosseumEnvCfg(ManagerBasedRlEnvCfg):
  """Configuration for ColosseumEnv."""

  class_type: ClassVar[type[ColosseumEnv]]

  encoders: dict[str, RmaTermCfg] = field(default_factory=dict)

  use_depth_camera: bool = False
  """Include depth camera sensor (only meaningful when encoders are configured)."""

  viz_callbacks: Annotated[list[tuple[str, Callable]], tyro.conf.Suppress] = field(
    default_factory=list
  )
  """(name, factory) pairs registered into manager_visualizers."""


class ColosseumEnv(ManagerBasedRlEnv):
  """Unified colosseum environment.

  Conditionally owns an RmaManager depending on whether cfg.encoders is
  non-empty.
  """

  cfg: ColosseumEnvCfg  # type: ignore[override]

  def get_observations(self) -> dict:
    return self.observation_manager.compute()

  def load_managers(self) -> None:
    super().load_managers()
    self.rma_manager = RmaManager(cfg=self.cfg.encoders, env=self)

  def setup_manager_visualizers(self) -> None:
    super().setup_manager_visualizers()
    for name, factory in getattr(self.cfg, "viz_callbacks", []):
      self.manager_visualizers[name] = factory(self)

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    super()._reset_idx(env_ids)
    if env_ids is not None and hasattr(self, "rma_manager"):
      self.rma_manager.reset(env_ids)

  def step(self, action: torch.Tensor) -> VecEnvStepReturn:
    obs, rewards, terminated, truncated, extras = super().step(action)
    if hasattr(self, "rma_manager"):
      self.rma_manager.update()
    return obs, rewards, terminated, truncated, extras


ColosseumEnvCfg.class_type = ColosseumEnv
