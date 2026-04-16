"""Soccer-maze specific reward functions.

Body-frame variants of ball velocity rewards.  The ball velocity command
from AbstractionVelocityCommand is world-frame, but the actor observes
ball position, ball velocity, and command all in robot body frame.  These
rewards do the same comparison in body frame so everything is consistent.

Mathematically the L2 and angular rewards are rotation-invariant — the
values are identical to their world-frame counterparts — but computing
them in body frame avoids mixing frames when the command is rotated for
the actor observation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _ball_vel_body(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball XY velocity in robot body frame. Shape (N, 2)."""
  ball_vel_w = env.scene["ball"].data.root_link_lin_vel_w[:, :3]
  quat_w = env.scene["robot"].data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  return quat_apply(quat_conj, ball_vel_w)[:, :2]


def _cmd_body(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ball velocity command rotated into robot body frame. Shape (N, 2)."""
  cmd_w = env.command_manager.get_command(command_name)[:, :2]
  quat_w = env.scene["robot"].data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  cmd_3d = torch.cat(
    [cmd_w, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1
  )
  return quat_apply(quat_conj, cmd_3d)[:, :2]


def ball_vel_tracking_body(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 1.0,
) -> torch.Tensor:
  """exp(-sharpness * |v_ball_b - v_cmd_b|²). Body-frame variant."""
  ball_vel_b = _ball_vel_body(env)
  cmd_b = _cmd_body(env, command_name)
  error_sq = ((ball_vel_b - cmd_b) ** 2).sum(dim=-1)
  return torch.exp(-sharpness * error_sq)


def ball_vel_angle_body(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Direction match 1 - (ψ_ball - ψ_cmd)²/π². Body-frame variant."""
  ball_vel_b = _ball_vel_body(env)
  cmd_b = _cmd_body(env, command_name)
  psi_ball = torch.atan2(ball_vel_b[:, 1], ball_vel_b[:, 0])
  psi_cmd = torch.atan2(cmd_b[:, 1], cmd_b[:, 0])
  angle_err = (psi_ball - psi_cmd + math.pi) % (2 * math.pi) - math.pi
  return 1.0 - (angle_err**2) / (math.pi**2)


def robot_ball_yaw_body(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """Ball ahead of robot along command direction (body frame).

  e1 = 1 - dot(d_robot->ball_b, cmd_dir_b)  (ball in command direction)
  e2 = 1 - d_robot->ball_b[0] / |d|         (ball in front, X-forward)
  reward = exp(-2 * (e1 + e2))
  """
  robot = env.scene["robot"]
  ball_pos_w = env.scene["ball"].data.root_link_pos_w[:, :3]
  robot_pos_w = robot.data.root_link_pos_w[:, :3]

  # Robot-to-ball direction in body frame
  relative_w = ball_pos_w - robot_pos_w
  quat_w = robot.data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  d_ball_b = quat_apply(quat_conj, relative_w)[:, :2]
  d_norm = d_ball_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  d_ball_b_unit = d_ball_b / d_norm

  # Command direction in body frame
  cmd_b = _cmd_body(env, command_name)
  cmd_norm = cmd_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  unit_cmd = cmd_b / cmd_norm

  e1 = 1.0 - (d_ball_b_unit * unit_cmd).sum(dim=-1)
  e2 = 1.0 - d_ball_b_unit[:, 0]  # dot with [1, 0] (X-forward)

  reward = torch.exp(-2.0 * (e1 + e2))
  return reward * (cmd_b.norm(dim=-1) > min_speed).float()
