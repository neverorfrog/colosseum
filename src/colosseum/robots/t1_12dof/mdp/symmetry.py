"""Left-right symmetry for the 12-DOF model (sagittal-plane reflection).

The 12-DOF action space is the full joint set (no head/arm joints to exclude),
so ``mirror_joints`` and ``mirror_actions`` are the same 12-dim permutation.
"""

from __future__ import annotations

import torch

from colosseum.robots.t1_12dof.constants import (
  FLIP_SIGN_JOINT_NAMES,
  JOINT_NAMES,
  SYMMETRY_JOINT_NAMES,
)

_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(JOINT_NAMES)}

# index i -> mirrored joint's index.
JOINT_MIRROR_INDICES: list[int] = [
  _name_to_idx[SYMMETRY_JOINT_NAMES[name]] for name in JOINT_NAMES
]

_flip_set: set[str] = set(FLIP_SIGN_JOINT_NAMES)
JOINT_MIRROR_SIGNS: list[float] = [
  -1.0 if name in _flip_set else 1.0 for name in JOINT_NAMES
]

_MIRROR_IDX = torch.tensor(JOINT_MIRROR_INDICES, dtype=torch.long)
_MIRROR_SIGNS = torch.tensor(JOINT_MIRROR_SIGNS, dtype=torch.float32)


def mirror_joints(x: torch.Tensor) -> torch.Tensor:
  """Mirror a (..., 12) joint-indexed tensor (positions or velocities)."""
  signs = _MIRROR_SIGNS.to(device=x.device)
  idx = _MIRROR_IDX.to(device=x.device)
  return x[..., idx] * signs


# Action space == full joint set for the 12-DOF model.
mirror_actions = mirror_joints
