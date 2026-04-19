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

from colosseum.tasks.soccer_maze.mdp.sokoban_command import SokobanCommand

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
  min_speed: float = 0.05,
) -> torch.Tensor:
  """Direction match 1 - (ψ_ball - ψ_cmd)²/π². Body-frame variant.

  Masked to zero when ball command speed is below min_speed (e.g. during MOVE
  actions where the ball command is [0, 0] and direction is undefined).
  """
  ball_vel_b = _ball_vel_body(env)
  cmd_b = _cmd_body(env, command_name)
  psi_ball = torch.atan2(ball_vel_b[:, 1], ball_vel_b[:, 0])
  psi_cmd = torch.atan2(cmd_b[:, 1], cmd_b[:, 0])
  angle_err = (psi_ball - psi_cmd + math.pi) % (2 * math.pi) - math.pi
  active = (cmd_b.norm(dim=-1) > min_speed).float()
  return (1.0 - (angle_err**2) / (math.pi**2)) * active


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


def robot_heading_alignment(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """cos(heading_error) during MOVE actions; 0 during PUSH.

  Robot velocity command is in body frame with X=forward (body_forward_axis=(1,0,0)).
  cos(heading_error) = robot_lin_vel_x / |robot_lin_vel|.
  Breaks the sideways-walking local optimum that linear-velocity-only rewards miss.
  """
  sokoban: SokobanCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  vel = sokoban.robot_lin_vel  # [N, 2] body frame, X=forward
  vel_norm = vel.norm(dim=-1).clamp(min=1e-6)
  cos_err = vel[:, 0] / vel_norm
  move_mask = (~sokoban.is_push).float()
  return cos_err * move_mask


def robot_lin_vel_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Track robot body-frame linear velocity during MOVE actions only.

  Returns 0 during PUSH so it does not conflict with approach velocity rewards.
  """
  sokoban: SokobanCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  actual = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
  error = ((sokoban.robot_lin_vel - actual) ** 2).sum(dim=-1)
  move_mask = (~sokoban.is_push).float()
  return torch.exp(-error / std**2) * move_mask


def robot_ang_vel_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Track robot body-frame yaw rate during MOVE actions.

  Same formula as mjlab track_angular_velocity but reads from
  SokobanCommand.robot_omega_z (zero during PUSH).
  """
  sokoban: SokobanCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  actual = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
  error = (sokoban.robot_omega_z - actual) ** 2
  return torch.exp(-error / std**2)


def robot_ball_approach_vel_push(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """robot_ball_approach_vel gated to PUSH actions only.

  During MOVE the robot may need to reposition freely; giving approach-vel
  reward then discourages walking away from the ball to set up a push angle.
  """
  sokoban: SokobanCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  robot = env.scene["robot"]
  ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
  robot_pos = robot.data.root_link_pos_w[:, :2]
  d = ball_pos - robot_pos
  d_unit = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  approach_vel = (robot.data.root_link_lin_vel_w[:, :2] * d_unit).sum(dim=-1)
  cmd_speed = sokoban.ball_vel.norm(dim=-1)  # non-zero only during PUSH
  deficit = (cmd_speed - approach_vel).clamp(min=0.0)
  return torch.exp(-(deficit**2)) * sokoban.is_push.float()


def robot_ball_distance_push(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 0.5,
) -> torch.Tensor:
  """robot_ball_distance gated to PUSH actions only.

  During MOVE the robot must freely reposition; penalizing distance then
  prevents it from walking behind the ball to set up the correct push angle.
  """
  sokoban: SokobanCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  robot_pos = env.scene["robot"].data.root_link_pos_w[:, :2]
  ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
  dist_sq = ((ball_pos - robot_pos) ** 2).sum(dim=-1)
  return torch.exp(-sharpness * dist_sq) * sokoban.is_push.float()
