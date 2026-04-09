"""RmaBasedEnv — environment that owns an RmaManager.

All encoder-specific state (depth buffers, sensor reads, etc.) lives inside
individual RmaTerm instances. The env just forwards lifecycle calls.

Configuration follows the mjlab pattern: the env config holds
``encoders: dict[str, RmaTermCfg]`` directly. Visualization callbacks are
declared in ``viz_callbacks`` and registered by ViewerCompatibleEnv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Callable, ClassVar

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.types import VecEnvStepReturn

from colosseum.envs.viewer_compatible_env import ViewerCompatibleEnv
from colosseum.managers.rma_manager import RmaManager, RmaTermCfg


@dataclass(kw_only=True)
class RmaBasedEnvCfg(ManagerBasedRlEnvCfg):
  class_type: ClassVar[type[RmaBasedEnv]]  # set after class definition
  encoders: dict[str, RmaTermCfg] = field(default_factory=dict)
  use_depth_camera: bool = False
  """Include depth camera sensor in the scene (Phase 2 only).

  Keep False during Phase 1 to avoid the rendering overhead of unused cameras.
  """
  # (name, factory) pairs: factory(env) → object with debug_vis(vis).
  # Registered into manager_visualizers by ViewerCompatibleEnv.
  viz_callbacks: Annotated[list[tuple[str, Callable]], tyro.conf.Suppress] = field(default_factory=list)


class RmaBasedEnv(ViewerCompatibleEnv):
  """Env that owns an RmaManager.

  The rma_manager is built in load_managers() after the observation manager
  is ready (RmaTerm.__init__ reads group_obs_dim to infer encoder input dims).
  """

  cfg: RmaBasedEnvCfg  # type: ignore[override]

  def load_managers(self) -> None:
    super().load_managers()
    self.rma_manager = RmaManager(cfg=self.cfg.encoders, env=self)

  def step(self, action: torch.Tensor) -> VecEnvStepReturn:
    result = super().step(action)
    self.rma_manager.update()
    return result

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    super()._reset_idx(env_ids)
    if env_ids is not None:
      self.rma_manager.reset(env_ids)


RmaBasedEnvCfg.class_type = RmaBasedEnv
