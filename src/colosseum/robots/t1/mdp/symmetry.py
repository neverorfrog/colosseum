"""Layer 2: T1-specific symmetry (left-right reflection about the sagittal plane).

Resolves joint mirror mapping from name-based config (SYMMETRY_JOINT_NAMES and
FLIP_SIGN_JOINT_NAMES in t1/constants.py), matching holosoma's convention.

Provides mirror_joints for 23-DOF joint observations and mirror_actions for the
21-DOF action space (head joints excluded).
"""

from __future__ import annotations

import torch

from colosseum.robots.t1.constants import (
  FLIP_SIGN_JOINT_NAMES,
  JOINT_NAMES,
  SYMMETRY_JOINT_NAMES,
)

_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(JOINT_NAMES)}

# Joint permutation: index i maps to the mirrored joint's index.
JOINT_MIRROR_INDICES: list[int] = [
  _name_to_idx[SYMMETRY_JOINT_NAMES[name]] for name in JOINT_NAMES
]

# Sign mask: -1 for joints whose sign flips under mirroring, +1 otherwise.
_flip_set: set[str] = set(FLIP_SIGN_JOINT_NAMES)
JOINT_MIRROR_SIGNS: list[float] = [
  -1.0 if name in _flip_set else 1.0 for name in JOINT_NAMES
]

# Pre-built tensors so mirror_joints is allocation-free at runtime.
_MIRROR_IDX = torch.tensor(JOINT_MIRROR_INDICES, dtype=torch.long)
_MIRROR_SIGNS = torch.tensor(JOINT_MIRROR_SIGNS, dtype=torch.float32)


def mirror_joints(x: torch.Tensor) -> torch.Tensor:
  """Mirror a joint-indexed tensor under left-right reflection.

  Works for joint positions and velocities (23-DOF, JOINT_NAMES order).

  Args:
    x: (..., 23) tensor in JOINT_NAMES order.

  Returns:
    (..., 23) mirrored tensor.
  """
  signs = _MIRROR_SIGNS.to(device=x.device)
  idx = _MIRROR_IDX.to(device=x.device)
  return x[..., idx] * signs


# Action space excludes head joints (AAHead_yaw, Head_pitch) — 21 DOF.
_ACTION_NAMES = [
  n for n in JOINT_NAMES if n not in ("AAHead_yaw", "Head_pitch")
]
_action_name_to_idx = {n: i for i, n in enumerate(_ACTION_NAMES)}
_ACTION_MIRROR_INDICES = [
  _action_name_to_idx[SYMMETRY_JOINT_NAMES[n]] for n in _ACTION_NAMES
]
_ACTION_MIRROR_SIGNS = [
  -1.0 if n in _flip_set else 1.0 for n in _ACTION_NAMES
]
_MIRROR_IDX_ACTION = torch.tensor(_ACTION_MIRROR_INDICES, dtype=torch.long)
_MIRROR_SIGNS_ACTION = torch.tensor(_ACTION_MIRROR_SIGNS, dtype=torch.float32)


def mirror_actions(x: torch.Tensor) -> torch.Tensor:
  """Mirror a 21-DOF action tensor (JOINT_NAMES minus head joints).

  Args:
    x: (..., 21) action tensor in JOINT_NAMES order (head excluded).

  Returns:
    (..., 21) mirrored tensor.
  """
  signs = _MIRROR_SIGNS_ACTION.to(device=x.device)
  idx = _MIRROR_IDX_ACTION.to(device=x.device)
  return x[..., idx] * signs


# Waist + 12 legs (13 DOF) subset, in MuJoCo model index order (Waist comes
# before the legs). This is both the action space and the joint obs set for the
# t1-velocity-booster task (arms/head pinned, excluded). Waist is a midline
# yaw joint: it mirrors to itself with a sign flip (already in FLIP_SIGN_*).
_LEGS_WAIST_NAMES = [
  "Waist",
  "Left_Hip_Pitch",
  "Left_Hip_Roll",
  "Left_Hip_Yaw",
  "Left_Knee_Pitch",
  "Left_Ankle_Pitch",
  "Left_Ankle_Roll",
  "Right_Hip_Pitch",
  "Right_Hip_Roll",
  "Right_Hip_Yaw",
  "Right_Knee_Pitch",
  "Right_Ankle_Pitch",
  "Right_Ankle_Roll",
]
_lw_name_to_idx = {n: i for i, n in enumerate(_LEGS_WAIST_NAMES)}
_LW_MIRROR_INDICES = [
  _lw_name_to_idx[SYMMETRY_JOINT_NAMES[n]] for n in _LEGS_WAIST_NAMES
]
_LW_MIRROR_SIGNS = [-1.0 if n in _flip_set else 1.0 for n in _LEGS_WAIST_NAMES]
_MIRROR_IDX_LW = torch.tensor(_LW_MIRROR_INDICES, dtype=torch.long)
_MIRROR_SIGNS_LW = torch.tensor(_LW_MIRROR_SIGNS, dtype=torch.float32)


def mirror_legs_waist(x: torch.Tensor) -> torch.Tensor:
  """Mirror a (..., 13) Waist+legs tensor (positions, velocities or actions).

  The action space and the joint obs set are identical for the booster task, so
  this single function serves both ``mirror_joints`` and ``mirror_actions``.
  """
  signs = _MIRROR_SIGNS_LW.to(device=x.device)
  idx = _MIRROR_IDX_LW.to(device=x.device)
  return x[..., idx] * signs
