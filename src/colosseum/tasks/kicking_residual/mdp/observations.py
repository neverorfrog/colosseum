"""kicking-residual observation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def ball_angle_cos_sin(
  env: ManagerBasedRlEnv,
  command_name: str = "ball_angle",
) -> torch.Tensor:
  """(cos θ, sin θ) of the ball_angle command, in the robot body frame.

  ``command_name`` (BallAngleCommand) already stores the goal kick direction
  expressed in the robot's current body frame, so no further rotation is
  needed here. Shape (N, 2).
  """
  angle = env.command_manager.get_command(command_name)[:, 0]
  return torch.stack([angle.cos(), angle.sin()], dim=-1)
