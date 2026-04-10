"""RMA Manager — term-based privileged encoder lifecycle.

Pattern mirrors mjlab's CommandManager / AbstractionManager:
  RmaTermCfg   (abstract dataclass, build() → RmaTerm)
  RmaTerm      (abstract, lifecycle: encode_privileged / encode_adaptation / update / reset)
  RmaManager   (ManagerBase, holds a dict of RmaTerm instances)

Each RmaTerm pairs two encoders for the same latent slot:
  privileged_encoder  — GT obs → MLP → latent  (Phase 1, target)
  adaptation_encoder  — sensor data → net → latent  (Phase 2, trainable)

The manager iterates terms in declaration order, giving a stable actor input layout.
The manager never inspects encoder internals — it calls the term interface only.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from mjlab.managers.manager_base import ManagerBase, ManagerTermBase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class RmaTermCfg(abc.ABC):
  """Base config for an RMA encoder term.

  A term pairs two encoders for the same latent slot:
  - privileged_encoder: reads from ``privileged_obs_group`` (GT, always available).
  - adaptation_encoder: reads from ``adaptation_obs_group`` (sensor, Phase 2 only).

  Subclass and implement build() to return the concrete RmaTerm.
  """

  privileged_obs_group: str
  """Obs group fed to the privileged encoder (must exist in obs manager)."""

  adaptation_obs_group: str | None = None
  """Obs group fed to the adaptation encoder.
  Set to None if this term has no Phase 2 adaptation encoder."""

  latent_dim: int = 8
  """Shared output dimension for both encoders."""

  @abc.abstractmethod
  def build(self, env: ManagerBasedRlEnv) -> RmaTerm:
    raise NotImplementedError


class RmaTerm(ManagerTermBase):
  """Stateful encoder term managed by RmaManager.

  Each term owns a privileged encoder (always present) and optionally an
  adaptation encoder (present only if Phase 2 is configured).

  Interface the manager calls — terms must implement:
    encode_privileged(obs_dict)   — GT obs → latent  (Phase 1)
    encode_adaptation(obs_dict)   — sensor obs → latent  (Phase 2)

  Optional lifecycle hooks:
    update()          — called every env step (e.g. roll depth buffer)
    reset(env_ids)    — called on episode reset
  """

  def __init__(self, cfg: RmaTermCfg, env: ManagerBasedRlEnv) -> None:
    self.cfg = cfg
    super().__init__(env)

  # ------------------------------------------------------------------
  # Properties
  # ------------------------------------------------------------------

  @property
  def latent_dim(self) -> int:
    return self.cfg.latent_dim

  @property
  @abc.abstractmethod
  def privileged_encoder(self) -> nn.Module:
    """The privileged (GT-supervised) encoder nn.Module."""
    raise NotImplementedError

  @property
  def adaptation_encoder(self) -> nn.Module | None:
    """The adaptation (sensor-supervised) encoder nn.Module.

    Returns None if this term has no Phase 2 encoder. Override in subclasses
    that support Phase 2.
    """
    return None

  # ------------------------------------------------------------------
  # Encoding interface
  # ------------------------------------------------------------------

  @abc.abstractmethod
  def encode_privileged(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode GT privileged obs to latent.

    Args:
      obs_dict: Full obs dict (term reads its own ``privileged_obs_group`` key).

    Returns:
      (N, latent_dim) latent tensor.
    """
    raise NotImplementedError

  def encode_adaptation(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode sensor obs to latent (Phase 2).

    Args:
      obs_dict: Full obs dict (term reads its own ``adaptation_obs_group`` key).

    Returns:
      (N, latent_dim) latent tensor.

    Raises:
      NotImplementedError: If this term has no adaptation encoder.
    """
    raise NotImplementedError(
      f"{self.__class__.__name__} has no adaptation encoder. "
      "Override encode_adaptation() or set adaptation_obs_group in the cfg."
    )

  # ------------------------------------------------------------------
  # Parameter access
  # ------------------------------------------------------------------

  def privileged_parameters(self):
    """Yield privileged encoder parameters (optimised in Phase 1)."""
    yield from self.privileged_encoder.parameters()

  def adaptation_parameters(self):
    """Yield adaptation encoder parameters (optimised in Phase 2)."""
    if self.adaptation_encoder is not None:
      yield from self.adaptation_encoder.parameters()

  # ------------------------------------------------------------------
  # Adaptation mask
  # ------------------------------------------------------------------

  def get_adaptation_mask(self) -> torch.Tensor | None:
    """Return (N,) bool mask of envs with a valid adaptation signal.

    None means all envs are valid (default).  Override in subclasses where
    the adaptation sensor is only available for a subset of environments
    (e.g. ball within camera FOV) to let the manager apply a masked loss
    and fall back to the privileged encoder during collection.
    """
    return None

  def get_reset_event(self) -> torch.Tensor | None:
    """Return (N,) bool mask for reset-like events in the current step.

    Default is None (no per-step reset events provided by this term).
    """
    return None

  # ------------------------------------------------------------------
  # Adaptation obs snapshot
  # ------------------------------------------------------------------

  def get_current_adaptation_obs(self) -> dict[str, torch.Tensor]:
    """Return current adaptation obs keyed by adaptation_obs_group.

    Override in subclasses that maintain internal sensor buffers (e.g. depth CNN).
    Default returns empty dict if no adaptation_obs_group is configured.
    """
    if self.cfg.adaptation_obs_group is None:
      return {}
    raise NotImplementedError(
      f"{self.__class__.__name__} has adaptation_obs_group="
      f"'{self.cfg.adaptation_obs_group}' but does not implement "
      "get_current_adaptation_obs()."
    )

  def compute_loss(
    self,
    privileged_obs: dict[str, torch.Tensor],
    adaptation_obs: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor] | None:
    """Override for custom adaptation losses.

    Return:
      - dict of named scalar losses to be optimised/logged, or
      - None to use manager default latent regression for this term.
    """
    return None

  # ------------------------------------------------------------------
  # Lifecycle hooks
  # ------------------------------------------------------------------

  def update(self) -> None:
    """Called every env step. Override to update internal buffers."""

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    """Called on episode reset. Override to zero internal buffers."""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class RmaManager(ManagerBase):
  """Manages RMA encoder terms.

  Takes a dict of RmaTermCfg directly (same pattern as ObservationManager).
  Instantiated in load_managers() after the observation manager is ready
  (terms read group_obs_dim to infer encoder input dims).

  The manager never inspects encoder internals. It calls the term interface:
    term.encode_privileged(obs_dict)
    term.encode_adaptation(obs_dict)
    term.privileged_parameters()
    term.adaptation_parameters()
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
  # Dimension properties
  # ------------------------------------------------------------------

  @property
  def total_latent_dim(self) -> int:
    """Sum of all term latent dims (actor input grows by this amount)."""
    return sum(t.latent_dim for t in self._terms.values())

  @property
  def privileged_group_names(self) -> list[str]:
    """Privileged obs group names that RmaPPO must store in the rollout buffer."""
    return [t.cfg.privileged_obs_group for t in self._terms.values()]

  @property
  def adaptation_group_names(self) -> list[str]:
    """Adaptation obs group names (needed for Phase 2 rollout storage)."""
    return [
      t.cfg.adaptation_obs_group
      for t in self._terms.values()
      if t.cfg.adaptation_obs_group is not None
    ]

  # ------------------------------------------------------------------
  # Encoding
  # ------------------------------------------------------------------

  def encode(
    self,
    obs_dict: dict[str, torch.Tensor],
    *,
    phase: int = 1,
  ) -> torch.Tensor:
    """Run all terms and return concatenated latents.

    Args:
      obs_dict: Full obs dict (each term reads the key(s) it needs).
      phase:    1 → privileged encoders, 2 → adaptation encoders.

    Returns:
      z: (N, total_latent_dim) concatenated latents in declaration order.
    """
    encode_fn = "encode_privileged" if phase == 1 else "encode_adaptation"
    return torch.cat(
      [getattr(term, encode_fn)(obs_dict) for term in self._terms.values()],
      dim=-1,
    )

  def get_adaptation_obs(self) -> dict[str, torch.Tensor]:
    """Snapshot current adaptation obs from all terms.

    Returns a dict keyed by each term's ``adaptation_obs_group`` name,
    which is the same key that ``encode_adaptation`` and
    ``compute_adaptation_loss`` expect.  Call this immediately after
    ``env.step()`` so the buffers are aligned with the current physics state.
    """
    result: dict[str, torch.Tensor] = {}
    for term in self._terms.values():
      result.update(term.get_current_adaptation_obs())
    return result

  def get_adaptation_mask(self) -> torch.Tensor | None:
    """Combined adaptation mask across all terms (AND).

    Returns None when no term defines a mask (all envs valid for all terms).
    When multiple terms define masks the AND is taken: an env is valid only
    when all terms have a valid signal.
    """
    masks = [
      m for term in self._terms.values()
      if (m := term.get_adaptation_mask()) is not None
    ]
    if not masks:
      return None
    result = masks[0]
    for m in masks[1:]:
      result = result & m
    return result

  def get_reset_event(self) -> torch.Tensor | None:
    """Combined reset-like event mask across all terms (OR)."""
    events = [
      e for term in self._terms.values()
      if (e := term.get_reset_event()) is not None
    ]
    if not events:
      return None
    result = events[0]
    for e in events[1:]:
      result = result | e
    return result

  def encode_phase2_with_fallback(
    self,
    priv_obs: dict[str, torch.Tensor],
    adapt_obs: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    """Phase 2 encoding with per-env fallback to the privileged encoder.

    For envs where get_adaptation_mask() is True (valid sensor signal):
      → adaptation encoders (gradient-tracked).
    For envs where get_adaptation_mask() is False (signal unavailable):
      → privileged encoders detached (no gradient, policy stays well-conditioned).

    If priv_obs is empty (real-hardware deployment, no GT available) the
    fallback is skipped and the adaptation encoder output is returned as-is.

    Args:
      priv_obs:   GT privileged obs dict (for the frozen fallback path).
      adapt_obs:  Sensor adaptation obs dict (depth frames, etc.).

    Returns:
      z: (N, total_latent_dim) blended latent tensor.
    """
    z_adapt = self.encode(adapt_obs, phase=2)
    mask = self.get_adaptation_mask()
    if mask is None or not priv_obs:
      return z_adapt
    with torch.no_grad():
      z_priv = self.encode(priv_obs, phase=1)
    return torch.where(mask.unsqueeze(-1), z_adapt, z_priv)

  # ------------------------------------------------------------------
  # Phase 2 loss
  # ------------------------------------------------------------------

  def compute_adaptation_loss(
    self,
    privileged_obs: dict[str, torch.Tensor],
    adaptation_obs: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    """Compute adaptation losses for all terms.

    For each term:
      1) use term.compute_loss(...) when provided, else
      2) fallback to latent MSE regression against frozen privileged encoder.

    Args:
      privileged_obs:  Obs dict for privileged encoder (GT groups).
      adaptation_obs:  Obs dict for adaptation encoder (sensor groups).
      mask:            Optional (...,) bool tensor (temporally-aligned, snapshotted
                       at collection time).  When provided, loss is computed only
                       on mask=True samples.  None = all samples are valid.

    Returns:
      Dict of named scalar loss tensors.
    """
    def _masked_mse(
      pred: torch.Tensor,
      target: torch.Tensor,
      valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
      if valid_mask is None:
        return F.mse_loss(pred, target)

      valid_mask = valid_mask.to(dtype=torch.bool)
      diff = (pred - target).pow(2)

      if valid_mask.dim() == diff.dim():
        if valid_mask.any():
          return diff[valid_mask].mean()
        return diff.sum() * 0.0

      while diff.dim() > valid_mask.dim():
        diff = diff.mean(dim=-1)

      if valid_mask.any():
        return diff[valid_mask].mean()
      return diff.sum() * 0.0

    losses: dict[str, torch.Tensor] = {}

    for term_name, term in self._terms.items():
      custom = term.compute_loss(privileged_obs, adaptation_obs, mask=mask)
      if custom is not None:
        for key, value in custom.items():
          if key in losses:
            losses[f"{term_name}.{key}"] = value
          else:
            losses[key] = value
        continue

      with torch.no_grad():
        z_target = term.encode_privileged(privileged_obs)
      z_pred = term.encode_adaptation(adaptation_obs)
      fallback_key = "latent_mse" if "latent_mse" not in losses else f"{term_name}.latent_mse"
      losses[fallback_key] = _masked_mse(z_pred, z_target, mask)

    if losses:
      return losses

    # No adaptation terms available — return differentiable zero.
    for term_name, term in self._terms.items():
      if term.adaptation_encoder is None:
        continue
      p = next(iter(term.adaptation_encoder.parameters()), None)
      if p is not None:
        return {"latent_mse": p.sum() * 0.0}
    return {"latent_mse": torch.tensor(0.0, device=self._env.device)}

  # ------------------------------------------------------------------
  # Parameter access
  # ------------------------------------------------------------------

  def parameters(self):
    """Phase 1: yield all privileged encoder parameters."""
    yield from self.privileged_parameters()

  def privileged_parameters(self):
    """Yield privileged encoder parameters across all terms."""
    for term in self._terms.values():
      yield from term.privileged_parameters()

  def adaptation_parameters(self):
    """Yield adaptation encoder parameters across all terms."""
    for term in self._terms.values():
      yield from term.adaptation_parameters()

  # ------------------------------------------------------------------
  # Lifecycle
  # ------------------------------------------------------------------

  def update(self) -> None:
    """Called every env step — forwards to each term's update()."""
    for term in self._terms.values():
      term.update()

  def reset(self, env_ids: torch.Tensor) -> dict:
    for term in self._terms.values():
      term.reset(env_ids)
    return {}

  def train(self, mode: bool = True) -> None:
    for term in self._terms.values():
      term.privileged_encoder.train(mode)
      if term.adaptation_encoder is not None:
        term.adaptation_encoder.train(mode)

  def eval(self) -> None:
    self.train(False)

  # ------------------------------------------------------------------
  # Checkpointing
  # ------------------------------------------------------------------

  def state_dict(self) -> dict[str, dict]:
    """Nested state dict keyed by term name, then encoder type."""
    result = {}
    for name, term in self._terms.items():
      result[name] = {"privileged": term.privileged_encoder.state_dict()}
      if term.adaptation_encoder is not None:
        result[name]["adaptation"] = term.adaptation_encoder.state_dict()
    return result

  def load_state_dict(self, state: dict[str, dict]) -> None:
    for name, enc_states in state.items():
      if name not in self._terms:
        continue
      term = self._terms[name]
      # Support legacy format where enc_states is the privileged encoder's
      # state dict directly (no nested 'privileged'/'adaptation' keys).
      if "privileged" not in enc_states:
        term.privileged_encoder.load_state_dict(enc_states)
        continue
      term.privileged_encoder.load_state_dict(enc_states["privileged"])
      if "adaptation" in enc_states and term.adaptation_encoder is not None:
        term.adaptation_encoder.load_state_dict(enc_states["adaptation"])
