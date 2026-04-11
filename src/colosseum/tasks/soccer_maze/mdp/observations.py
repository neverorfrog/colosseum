"""Soccer-maze specific observation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ball_vel_command_body(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ball velocity command rotated into the robot body frame. Shape (N, 2).

  The command from AbstractionVelocityCommand is in world frame [vx_w, vy_w].
  The ball_pos observation is in body frame.  Giving both in the same frame
  lets the actor learn the relative geometry without needing heading knowledge.
  """
  command = env.command_manager.get_command(command_name)[:, :2]  # [N, 2] world
  robot = env.scene["robot"]
  quat_w = robot.data.root_link_quat_w  # [N, 4]
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  cmd_3d = torch.cat(
    [command, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1
  )  # [N, 3]
  return quat_apply(quat_conj, cmd_3d)[:, :2]  # [N, 2]


def ball_vel_xy_body(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball XY velocity in robot body frame. Shape (N, 2).

  Consistent frame with ball_pos (also body frame) so the actor can reason
  about ball state without needing heading knowledge.
  """
  ball_vel_w = env.scene["ball"].data.root_link_lin_vel_w[:, :3]  # [N, 3]
  robot = env.scene["robot"]
  quat_w = robot.data.root_link_quat_w  # [N, 4]
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  return quat_apply(quat_conj, ball_vel_w)[:, :2]  # [N, 2]
