"""Curriculum functions for the soccer-maze Sokoban task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from colosseum.envs.abstraction_based_env import AbstractionBasedEnv
from colosseum.mdp.abstraction.maze.sokoban_grid_abstraction import SokobanGridAbstraction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def sokoban_cache_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  abstraction_name: str,
  clear_at_steps: list[int],
) -> torch.Tensor:
  """Clear the Sokoban plan cache at specified global-step thresholds.

  When training crosses a threshold for the first time the cache is invalidated
  so that plans are recomputed for whatever new configuration is active (e.g.
  after the obstacle mask is updated by a separate map-complexity curriculum).

  This function is stateless on the abstraction — it tracks which thresholds
  have already fired via a private attribute ``_sokoban_cleared_up_to`` on the
  env object.

  Args:
      env:              Environment instance (must be AbstractionBasedEnv).
      env_ids:          Environments being reset (unused — cache is global).
      abstraction_name: Key of the SokobanGridAbstraction in abstraction_manager.
      clear_at_steps:   Sorted list of global step counts at which to clear.
                        Example: [500_000, 2_000_000, 5_000_000].

  Returns:
      Scalar tensor: index of the last threshold that has been crossed
      (0 = none yet, 1 = first, …).  Used as a curriculum metric.
  """
  del env_ids  # Cache is global, not per-environment.
  assert isinstance(env, AbstractionBasedEnv)

  step = env.common_step_counter
  last_cleared: int = getattr(env, "_sokoban_cleared_up_to", -1)

  for threshold in sorted(clear_at_steps):
    if step >= threshold > last_cleared:
      abstraction = env.abstraction_manager.get_term(abstraction_name)
      assert isinstance(abstraction, SokobanGridAbstraction)
      abstraction.clear_cache()
      env._sokoban_cleared_up_to = threshold
      last_cleared = threshold

  stage = sum(1 for t in clear_at_steps if step >= t)
  return torch.tensor(stage, dtype=torch.float32)
