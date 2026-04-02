from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.manager_base import ManagerBase, ManagerTermBase

if TYPE_CHECKING:
  from colosseum.envs.abstraction_based_env import AbstractionBasedEnv


# ── Settings (change-detection mechanism) ──────────────────────────────────


@dataclass(frozen=True, eq=False)
class AbstractionSettings:
  """Immutable snapshot of the inputs that drive an abstraction rebuild.

  Subclass this per abstraction type and add exactly the fields whose
  change should trigger a rebuild (goal position, obstacle mask, etc.).
  Custom __eq__ handles torch.Tensor comparison correctly.
  """

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, AbstractionSettings):
      return False
    for field in self.__dataclass_fields__:
      a, b = getattr(self, field), getattr(other, field)
      if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
          return False
      elif a != b:
        return False
    return True


# ── Term ────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class AbstractionTermCfg:
  """Base config for an abstraction term.

  Subclass and implement build() to return the concrete AbstractionTerm.
  """

  debug_vis: bool = False

  @abstractmethod
  def build(self, env: AbstractionBasedEnv) -> AbstractionTerm:
    raise NotImplementedError


class AbstractionTerm(ManagerTermBase):
  """Stateful abstraction that caches signals between environment steps.

  Lifecycle (called by AbstractionManager):
    reset(env_ids)  ->  _update_settings -> _maybe_rebuild
    compute(dt)     ->  _update_signals

  Signals are NOT defined here. Each concrete subclass exposes whatever
  signals it computes (value, direction, footstep, reachability, ...) as
  its own methods. Consumers call those methods directly after obtaining
  the term via AbstractionManager.get_term().
  """

  def __init__(self, cfg: AbstractionTermCfg, env: AbstractionBasedEnv):
    self.cfg = cfg
    super().__init__(env)

  def reset(self, env_ids: torch.Tensor) -> None:
    assert isinstance(env_ids, torch.Tensor)
    self._update_settings(env_ids)
    self._maybe_rebuild(env_ids)

  def compute(self, dt: float) -> None:
    del dt
    env_ids = torch.arange(self.num_envs, device=self.device)
    self._update_signals(env_ids)

  def debug_vis(self, visualizer) -> None:
    if self.cfg.debug_vis:
      self._debug_vis_impl(visualizer)

  def _debug_vis_impl(self, visualizer) -> None:
    pass

  @abstractmethod
  def _update_settings(self, env_ids: torch.Tensor) -> None:
    """Read current env state and update internal settings snapshot."""
    raise NotImplementedError

  @abstractmethod
  def _maybe_rebuild(self, env_ids: torch.Tensor) -> None:
    """Rebuild the abstraction if settings changed since last reset."""
    raise NotImplementedError

  @abstractmethod
  def _update_signals(self, env_ids: torch.Tensor) -> None:
    """Recompute cached signal tensors from current env state."""
    raise NotImplementedError


# ── Manager ─────────────────────────────────────────────────────────────────


class AbstractionManager(ManagerBase):
  def __init__(self, cfg: dict[str, AbstractionTermCfg], env: AbstractionBasedEnv):
    self.cfg = deepcopy(cfg)
    super().__init__(env=env)

  @property
  def active_terms(self) -> list[str]:
    return self._term_names

  def _prepare_terms(self) -> None:
    self._term_names: list[str] = []
    self._terms: dict[str, AbstractionTerm] = {}
    for name, term_cfg in self.cfg.items():
      if term_cfg is None:
        continue
      self._term_names.append(name)
      self._terms[name] = term_cfg.build(env=self._env)

  def get_term(self, name: str) -> AbstractionTerm | None:
    return self._terms.get(name)

  def reset(self, env_ids: torch.Tensor) -> dict:
    for term in self._terms.values():
      term.reset(env_ids)
    return {}

  def compute(self, dt: float) -> None:
    for term in self._terms.values():
      term.compute(dt)

  def debug_vis(self, visualizer) -> None:
    for term in self._terms.values():
      term.debug_vis(visualizer)


class NullAbstractionManager(AbstractionManager):
  def _prepare_terms(self) -> None:
    self._term_names = []
    self._terms = {}

  def reset(self, env_ids: torch.Tensor) -> dict:
    return {}

  def compute(self, dt: float) -> None:
    pass
