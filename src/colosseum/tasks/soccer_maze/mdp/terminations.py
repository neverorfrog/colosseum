"""Termination functions for the soccer-maze task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from colosseum.envs.abstraction_based_env import AbstractionBasedEnv
from colosseum.mdp.abstraction.maze.sokoban_grid_abstraction import SokobanGridAbstraction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def sokoban_plan_deviated(env: ManagerBasedRlEnv, abstraction_name: str) -> torch.Tensor:
  """Terminate when the Sokoban plan deviates (robot or ball went to wrong cell).

  Reads ``deviation_detected`` from the named SokobanGridAbstraction.  The flag
  is set by ``_update_signals`` and cleared on reset.

  Args:
      env:              Environment instance.
      abstraction_name: Key under which the SokobanGridAbstraction is registered
                        in the abstraction manager (e.g. ``"sokoban"``).

  Returns:
      [N] bool tensor — True for envs whose current plan has deviated.
  """
  assert isinstance(env, AbstractionBasedEnv)
  abstraction = env.abstraction_manager.get_term(abstraction_name)
  assert isinstance(abstraction, SokobanGridAbstraction)
  return abstraction.deviation_detected
