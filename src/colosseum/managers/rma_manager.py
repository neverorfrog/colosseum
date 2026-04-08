"""RMA Manager — term-based privileged encoder lifecycle.

Pattern mirrors mjlab's CommandManager / AbstractionManager:
  RmaTermCfg   (abstract dataclass, build() → RmaTerm)
  RmaTerm      (abstract, lifecycle: encode / update / reset)
  RmaManager   (ManagerBase, holds a dict of RmaTerm instances)

Concrete terms live in rma_terms.py:
  PrivilegedRmaTerm  — GT obs group → MLP → latent  (Phase 1)
  DepthRmaTerm       — depth sensor → rolling buffer → CNN+LSTM → latent  (Phase 2)

RmaManager.encode(privileged_obs) iterates terms in declaration order and
concatenates latents, giving a stable actor input layout.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from mjlab.managers.manager_base import ManagerBase, ManagerTermBase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class RmaTermCfg(abc.ABC):
  """Base config for an RMA encoder term.

  Subclass and implement build() to return the concrete RmaTerm.
  """

  obs_group: str
  """Privileged observation group this term supervises (must exist in obs manager)."""

  latent_dim: int = 8
  """Dimensionality of the encoder output latent."""

  @abc.abstractmethod
  def build(self, env: ManagerBasedRlEnv) -> RmaTerm:
    raise NotImplementedError


class RmaTerm(ManagerTermBase):
  """Stateful encoder term managed by RmaManager.

  Lifecycle (called by RmaManager):
    update()          — called every env step (e.g. roll depth buffer)
    reset(env_ids)    — called on episode reset
    encode(priv_obs)  — return latent tensor (N, latent_dim)

  Terms that use a GT obs dict (PrivilegedRmaTerm) set needs_privileged_obs=True
  so RmaManager tells RmaPPO to store those groups in the rollout buffer.
  Terms with an internal buffer (DepthRmaTerm) set it to False.
  """

  def __init__(self, cfg: RmaTermCfg, env: ManagerBasedRlEnv) -> None:
    self.cfg = cfg
    super().__init__(env)

  @property
  def latent_dim(self) -> int:
    return self.cfg.latent_dim

  @property
  def needs_privileged_obs(self) -> bool:
    """True if this term's encode() needs a GT obs tensor from the buffer."""
    return True

  @property
  def encoder(self) -> nn.Module:
    """The underlying nn.Module (for parameter iteration and checkpointing)."""
    raise NotImplementedError

  @abc.abstractmethod
  def encode(self, privileged_obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Produce latent vector.

    Args:
      privileged_obs: Dict of GT privileged groups from the obs manager.
                      Terms with needs_privileged_obs=False may ignore this.

    Returns:
      (N, latent_dim) latent tensor.
    """
    raise NotImplementedError

  def update(self) -> None:
    """Called every env step. Override to update internal buffers."""

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    """Called on episode reset. Override to zero internal buffers."""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class RmaManager(ManagerBase):
  """Manages RMA encoder terms for RMA-style training.

  Takes a dict of RmaTermCfg directly (same pattern as ObservationManager,
  RewardManager, etc. in mjlab). Instantiated in load_managers() after the
  observation manager is ready (terms read group_obs_dim to infer input dims).
  """

  def __init__(self, cfg: dict[str, RmaTermCfg], env: ManagerBasedRlEnv) -> None:
    self.cfg = cfg
    self._terms: dict[str, RmaTerm] = {}
    super().__init__(env)

  # ------------------------------------------------------------------
  # ManagerBase interface
  # ------------------------------------------------------------------

  @property
  def active_terms(self) -> list[str]:
    return list(self._terms.keys())

  def _prepare_terms(self) -> None:
    for name, term_cfg in self.cfg.items():
      if term_cfg is None:
        continue
      self._terms[name] = term_cfg.build(self._env)

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  @property
  def total_latent_dim(self) -> int:
    """Sum of all term latent dims (actor input grows by this amount)."""
    return sum(t.latent_dim for t in self._terms.values())

  @property
  def privileged_group_names(self) -> list[str]:
    """Obs group names that need to be stored in the rollout buffer.

    Only includes groups from terms that actually need GT obs at encode time
    (PrivilegedRmaTerm). Depth terms manage their own buffers internally.
    """
    return [
      t.cfg.obs_group for t in self._terms.values() if t.needs_privileged_obs
    ]

  def encode(self, privileged_obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run all terms and return concatenated latents.

    Args:
      privileged_obs: Maps group name → (N, group_dim) tensor.

    Returns:
      z: (N, total_latent_dim) concatenated latents in declaration order.
    """
    return torch.cat(
      [term.encode(privileged_obs) for term in self._terms.values()], dim=-1
    )

  def update(self) -> None:
    """Called every env step — forwards to each term's update()."""
    for term in self._terms.values():
      term.update()

  def reset(self, env_ids: torch.Tensor) -> dict:
    for term in self._terms.values():
      term.reset(env_ids)
    return {}

  def parameters(self):
    """Yield encoder parameters (for optimizer construction)."""
    for term in self._terms.values():
      yield from term.encoder.parameters()

  def state_dict(self) -> dict[str, dict]:
    """Nested state dict keyed by term name."""
    return {name: term.encoder.state_dict() for name, term in self._terms.items()}

  def load_state_dict(self, state: dict[str, dict]) -> None:
    for name, enc_state in state.items():
      self._terms[name].encoder.load_state_dict(enc_state)

  def train(self, mode: bool = True) -> None:
    for term in self._terms.values():
      term.encoder.train(mode)

  def eval(self) -> None:
    self.train(False)
