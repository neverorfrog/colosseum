"""ColosseumEnv — unified environment for all colosseum tasks.

Replaces the six legacy subclasses (ViewerCompatibleEnv, RmaBasedEnv,
AbstractionBasedEnv, ConstraintBasedEnv, ConstraintRmaEnv,
ConstraintAbstractionBasedEnv) with a single class that conditionally
loads managers based on config.

Managers are loaded only when their config section is non-empty:
  - rma_manager        → cfg.encoders is non-empty
  - abstraction_manager → cfg.abstractions is non-empty
  - constraint_manager  → cfg.constraints is non-empty

Step order (per-step):
  1. Physics + rewards (ManagerBasedRlEnv.step)
  2. RMA encoder update (after obs are ready)
  3. Constraint scaling and soft termination (Option B: float terminated)
  4. Abstraction state update for the NEXT step

Reset order (_reset_idx):
  1. Snapshot episode lengths (before super zeroes them)
  2. ManagerBasedRlEnv._reset_idx (standard reset)
  3. Abstraction reset
  4. Constraint reset (with pre-reset episode lengths)
  5. RMA reset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, ClassVar

import torch
import tyro
from loguru import logger
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.types import VecEnvObs, VecEnvStepReturn

from colosseum.managers.abstraction_manager import (
  AbstractionManager,
  AbstractionTermCfg,
)
from colosseum.managers.constraint_manager import ConstraintManager, ConstraintTermCfg
from colosseum.managers.rma_manager import RmaManager, RmaTermCfg


@dataclass(kw_only=True)
class ColosseumEnvCfg(ManagerBasedRlEnvCfg):
  """Configuration for ColosseumEnv."""

  class_type: ClassVar[type[ColosseumEnv]]

  encoders: dict[str, RmaTermCfg] = field(default_factory=dict)
  abstractions: dict[str, AbstractionTermCfg] = field(default_factory=dict)
  constraints: dict[str, ConstraintTermCfg] = field(default_factory=dict)

  use_depth_camera: bool = False
  """Include depth camera sensor (only meaningful when encoders are configured)."""

  viz_callbacks: Annotated[list[tuple[str, Callable]], tyro.conf.Suppress] = field(
    default_factory=list
  )
  """(name, factory) pairs registered into manager_visualizers."""


class ColosseumEnv(ManagerBasedRlEnv):
  """Unified colosseum environment.

  Conditionally owns an RmaManager, AbstractionManager, and/or
  ConstraintManager depending on which config sections are non-empty.
  """

  cfg: ColosseumEnvCfg  # type: ignore[override]

  def get_observations(self) -> dict:
    return self.observation_manager.compute()

  def load_managers(self) -> None:
    super().load_managers()
    if self.cfg.abstractions:
      self.abstraction_manager = AbstractionManager(cfg=self.cfg.abstractions, env=self)
    self.rma_manager = RmaManager(cfg=self.cfg.encoders, env=self)
    if self.cfg.constraints:
      self.constraint_manager = ConstraintManager(cfg=self.cfg.constraints, env=self)

  def setup_manager_visualizers(self) -> None:
    super().setup_manager_visualizers()
    for name, factory in getattr(self.cfg, "viz_callbacks", []):
      self.manager_visualizers[name] = factory(self)
    if hasattr(self, "abstraction_manager"):
      logger.info("Setting up abstraction manager visualizer...")
      self.manager_visualizers["abstraction_manager"] = self.abstraction_manager

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    ep_lens = (
      self.episode_length_buf[env_ids].clone()
      if env_ids is not None
      else self.episode_length_buf.clone()
    )
    super()._reset_idx(env_ids)
    if hasattr(self, "abstraction_manager"):
      info = self.abstraction_manager.reset(env_ids)
      self.extras["log"].update(info)
    if hasattr(self, "constraint_manager"):
      info = self.constraint_manager.reset(env_ids, ep_lens)
      self.extras["log"].update(info)
    if env_ids is not None and hasattr(self, "rma_manager"):
      self.rma_manager.reset(env_ids)

  def reset(
    self,
    *,
    seed: int | None = None,
    env_ids: torch.Tensor | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[VecEnvObs, dict]:
    obs, info = super().reset(seed=seed, options=options)
    if hasattr(self, "abstraction_manager"):
      all_env_ids = torch.arange(self.num_envs, device=self.device)
      self.abstraction_manager.reset(all_env_ids)
    return obs, info

  def step(self, action: torch.Tensor) -> VecEnvStepReturn:
    obs, rewards, terminated, truncated, extras = super().step(action)
    if hasattr(self, "rma_manager"):
      self.rma_manager.update()
    if hasattr(self, "constraint_manager"):
      cstr_prob = self.constraint_manager.compute()
      rewards = torch.clip(rewards * (1.0 - cstr_prob), min=0.0)
      terminated_float = cstr_prob.clone()
      terminated_float[terminated.bool()] = 1.0
      terminated = terminated_float
    if hasattr(self, "abstraction_manager"):
      self.abstraction_manager.compute(dt=self.step_dt)
    return obs, rewards, terminated, truncated, extras


ColosseumEnvCfg.class_type = ColosseumEnv
