"""Ball-gated gait-shaping rewards for the residual dribbling tasks.

Thin wrappers around the generic gait rewards that switch off within a radius
of the ball: far from it the robot must respect the gait clock (clean,
phase-disciplined approach steps), inside the gate footwork is unconstrained so
the policy is free to pick step length, frequency, and timing for the strike.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from colosseum.mdp.rewards import feet_phase, feet_swing

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _ball_far_mask(env: ManagerBasedRlEnv, gate_distance: float) -> torch.Tensor:
  robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  return ((ball_xy - robot_xy).norm(dim=-1) > gate_distance).float()


def feet_phase_ball_gated(
  env: ManagerBasedRlEnv, gate_distance: float = 0.5, **kwargs
) -> torch.Tensor:
  return feet_phase(env, **kwargs) * _ball_far_mask(env, gate_distance)


def feet_swing_ball_gated(
  env: ManagerBasedRlEnv, gate_distance: float = 0.5, **kwargs
) -> torch.Tensor:
  return feet_swing(env, **kwargs) * _ball_far_mask(env, gate_distance)
