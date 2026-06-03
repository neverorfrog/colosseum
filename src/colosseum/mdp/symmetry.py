"""Layer 1: Universal symmetry utilities (robot-agnostic).

Provides:
- MirrorableObservationTermCfg: ObservationTermCfg subclass that carries a mirror_fn.
- Generic mirror functions for common observation types.
- build_symmetry_spec / mirror_obs: compile obs mirror layout and apply it.
- augment_batch: double batch via x-z plane mirroring (holosoma-style data aug).

Usage pattern
-------------
1. Declare mirror_fn on obs terms in observation_cfg.py:

    from colosseum.mdp.symmetry import MirrorableObservationTermCfg, mirror_ang_vel
    from colosseum.robots.t1_23dof.mdp.symmetry import mirror_joints

    "base_ang_vel": MirrorableObservationTermCfg(
        func=builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        mirror_fn=mirror_ang_vel,
    )

2. In the algorithm, build specs once from env's observation manager:

    from colosseum.mdp.symmetry import build_symmetry_spec, mirror_obs

    self._actor_sym_spec = build_symmetry_spec(env.observation_manager, "actor")
    self._critic_sym_spec = build_symmetry_spec(env.observation_manager, "critic")

3. Data augmentation during PPO update (holosoma style):

    actor_obs, critic_obs, actions = augment_batch(
        actor_obs, critic_obs, actions,
        self._actor_sym_spec, self._critic_sym_spec, self._action_mirror_fn,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from mjlab.managers.observation_manager import ObservationTermCfg


# ---------------------------------------------------------------------------
# Subclass: ObservationTermCfg + mirror_fn
# ---------------------------------------------------------------------------

@dataclass
class MirrorableObservationTermCfg(ObservationTermCfg):
  """ObservationTermCfg extended with a left-right mirror function.

  mirror_fn: pure function (Tensor) -> Tensor applied to one term's slice of
  the concatenated observation vector. None means the term is invariant.
  """
  mirror_fn: Callable[[torch.Tensor], torch.Tensor] | None = None


# ---------------------------------------------------------------------------
# Generic mirror functions
# ---------------------------------------------------------------------------

def mirror_ang_vel(x: torch.Tensor) -> torch.Tensor:
  """Mirror base angular velocity under y → -y reflection.

  Angular velocity is a pseudovector:
    ω' = det(diag(1,-1,1)) * diag(1,-1,1) · ω = [-ωx, ωy, -ωz]

  Args:
    x: (..., 3) tensor [ωx, ωy, ωz].
  Returns:
    (..., 3) mirrored tensor [-ωx, ωy, -ωz].
  """
  return x * x.new_tensor([-1.0, 1.0, -1.0])


def mirror_projected_gravity(x: torch.Tensor) -> torch.Tensor:
  """Mirror projected gravity vector under y → -y reflection.

  Args:
    x: (..., 3) tensor [gx, gy, gz].
  Returns:
    (..., 3) mirrored tensor [gx, -gy, gz].
  """
  return x * x.new_tensor([1.0, -1.0, 1.0])


def mirror_base_lin_vel(x: torch.Tensor) -> torch.Tensor:
  """Mirror base linear velocity in base frame under y → -y reflection.

  Args:
    x: (..., 3) tensor [vx, vy, vz].
  Returns:
    (..., 3) mirrored tensor [vx, -vy, vz].
  """
  return x * x.new_tensor([1.0, -1.0, 1.0])


def mirror_velocity_command(x: torch.Tensor) -> torch.Tensor:
  """Mirror twist velocity command under y → -y reflection.

  [vx, vy, vyaw] → [vx, -vy, -vyaw].

  Args:
    x: (..., 3) tensor [vx, vy, vyaw].
  Returns:
    (..., 3) mirrored tensor [vx, -vy, -vyaw].
  """
  return x * x.new_tensor([1.0, -1.0, -1.0])


def mirror_xy(x: torch.Tensor) -> torch.Tensor:
  """Mirror a 2D body-frame planar vector under y → -y reflection.

  [px, py] → [px, -py]. Use for body-frame XY positions/velocities
  (e.g. ball position, ball velocity).

  Args:
    x: (..., 2) tensor [px, py].
  Returns:
    (..., 2) mirrored tensor [px, -py].
  """
  return x * x.new_tensor([1.0, -1.0])


def mirror_gait_phase(x: torch.Tensor) -> torch.Tensor:
  """Mirror 4D gait phase clock under left-right swap.

  [cos_L, cos_R, sin_L, sin_R] → [cos_R, cos_L, sin_R, sin_L].
  Left and right feet simply exchange roles.
  """
  return x[..., [1, 0, 3, 2]]


# ---------------------------------------------------------------------------
# Compiled symmetry spec
# ---------------------------------------------------------------------------

@dataclass
class TermMirrorSpec:
  """Compiled slice spec for one observation term."""
  start: int
  end: int
  base_dim: int
  history_length: int
  mirror_fn: Callable[[torch.Tensor], torch.Tensor] | None


def build_symmetry_spec(obs_manager, group_name: str) -> list[TermMirrorSpec]:
  """Compile a symmetry spec from the observation manager for one group.

  Call once at algorithm init time; the returned spec is allocation-free
  to apply at training time via mirror_obs().

  Args:
    obs_manager: mjlab ObservationManager (env.observation_manager).
    group_name:  Observation group name, e.g. "actor" or "critic".

  Returns:
    List of TermMirrorSpec, one per term in declaration order.
  """
  term_names = obs_manager.active_terms[group_name]
  term_dims = obs_manager.group_obs_term_dim[group_name]

  specs: list[TermMirrorSpec] = []
  offset = 0

  for name, dims in zip(term_names, term_dims):
    total_size = math.prod(dims)
    cfg = obs_manager.get_term_cfg(group_name, name)
    mirror_fn = getattr(cfg, "mirror_fn", None)
    history = getattr(cfg, "history_length", 0) or 0
    base_dim = total_size // history if history > 0 else total_size

    specs.append(TermMirrorSpec(
      start=offset,
      end=offset + total_size,
      base_dim=base_dim,
      history_length=history,
      mirror_fn=mirror_fn,
    ))
    offset += total_size

  return specs


def mirror_obs(obs: torch.Tensor, spec: list[TermMirrorSpec]) -> torch.Tensor:
  """Apply left-right mirror to a concatenated observation tensor.

  Terms with mirror_fn=None are copied unchanged. Terms with history are
  reshaped to (..., H, D), mirrored per-timestep, then flattened back.

  Args:
    obs:  (..., obs_dim) tensor.
    spec: compiled spec from build_symmetry_spec().

  Returns:
    (..., obs_dim) mirrored tensor, same dtype and device.
  """
  chunks: list[torch.Tensor] = []

  for term in spec:
    chunk = obs[..., term.start:term.end]

    if term.mirror_fn is not None:
      if term.history_length > 0:
        leading = chunk.shape[:-1]
        chunk = chunk.view(*leading, term.history_length, term.base_dim)
        chunk = term.mirror_fn(chunk)
        chunk = chunk.view(*leading, -1)
      else:
        chunk = term.mirror_fn(chunk)

    chunks.append(chunk)

  return torch.cat(chunks, dim=-1)


# ---------------------------------------------------------------------------
# Batch augmentation (holosoma-style data doubling)
# ---------------------------------------------------------------------------

def augment_obs(obs: torch.Tensor, spec: list[TermMirrorSpec]) -> torch.Tensor:
  """Double the batch by appending mirrored observations.

  Args:
    obs: (B, obs_dim) observation tensor.
    spec: compiled spec from build_symmetry_spec().

  Returns:
    (2*B, obs_dim) tensor: [original, mirrored].
  """
  return torch.cat((obs, mirror_obs(obs, spec)), dim=0)


def augment_actions(
  actions: torch.Tensor,
  action_mirror_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
  """Double the action batch by appending mirrored actions.

  Args:
    actions: (B, action_dim) tensor.
    action_mirror_fn: function to mirror actions (e.g. mirror_joints).

  Returns:
    (2*B, action_dim) tensor: [original, mirrored].
  """
  return torch.cat((actions, action_mirror_fn(actions)), dim=0)
