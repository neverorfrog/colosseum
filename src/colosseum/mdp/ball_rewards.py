"""Generic ball reward functions (robot-agnostic, no task-specific gating).

These functions are shared across all ball-dribbling tasks.  Task-specific
variants (e.g. Sokoban-gated versions for soccer-maze) live in each task's
own mdp/rewards.py and may import helpers from here.

Reference: Ji et al., "DribbleBot" (ICRA 2023), TABLE III.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Camera FOV helper
# ---------------------------------------------------------------------------


def camera_fov_mask(
  env: ManagerBasedRlEnv,
  pos_w: torch.Tensor,
  camera_name: str,
  camera_fovy: float,
  camera_aspect_ratio: float,
  depth_clip: float,
) -> torch.Tensor:
  """Return (N,) bool: True where pos_w (N, 2) world-XY falls inside camera frustum."""
  N = env.num_envs
  device = env.device
  try:
    cam_id = env.sim.mj_model.camera(camera_name).id
  except Exception:
    return torch.ones(N, dtype=torch.bool, device=device)
  cam_pos = env.sim.data.cam_xpos[:, cam_id, :]
  cam_mat = env.sim.data.cam_xmat[:, cam_id, :].reshape(-1, 3, 3)
  pos_3d = torch.cat([pos_w, torch.zeros(N, 1, device=device)], dim=-1)
  p_rel = pos_3d - cam_pos
  p_cam = torch.bmm(cam_mat.transpose(1, 2), p_rel.unsqueeze(-1)).squeeze(-1)
  in_front = p_cam[:, 2] < 0
  depth = (-p_cam[:, 2]).clamp_min(1e-6)
  tan_half_v = math.tan(math.radians(camera_fovy / 2))
  tan_half_h = tan_half_v * camera_aspect_ratio
  nx = p_cam[:, 0] / (depth * tan_half_h)
  ny = p_cam[:, 1] / (depth * tan_half_v)
  return in_front & (nx.abs() <= 1.0) & (ny.abs() <= 1.0) & (depth < depth_clip)


# ---------------------------------------------------------------------------
# Body-frame helpers
# ---------------------------------------------------------------------------


def ball_vel_body(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball XY velocity in robot body frame. Shape (N, 2)."""
  ball_vel_w = env.scene["ball"].data.root_link_lin_vel_w[:, :3]
  quat_w = env.scene["robot"].data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  return quat_apply(quat_conj, ball_vel_w)[:, :2]


def cmd_body(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ball velocity command rotated into robot body frame. Shape (N, 2).

  Reads the command from the standard command manager (world frame).
  """
  cmd_w = env.command_manager.get_command(command_name)[:, :2]
  quat_w = env.scene["robot"].data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  cmd_3d = torch.cat([cmd_w, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1)
  return quat_apply(quat_conj, cmd_3d)[:, :2]


# ---------------------------------------------------------------------------
# Ball velocity rewards (body-frame variants, DribbleBot TABLE III)
# ---------------------------------------------------------------------------


def ball_vel_tracking_body(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 1.0,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """exp(-sharpness * |v_ball_b - v_cmd_b|²). Zero when ball is stationary."""
  ball_vel_b = ball_vel_body(env)
  error_sq = ((ball_vel_b - cmd_body(env, command_name)) ** 2).sum(dim=-1)
  reward = torch.exp(-sharpness * error_sq)
  return reward * (ball_vel_b.norm(dim=-1) > min_speed).float()


def ball_vel_angle_body(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """Direction match 1 - (ψ_ball - ψ_cmd)²/π². Zero when ball is stationary."""
  ball_vel_b = ball_vel_body(env)
  cmd_b = cmd_body(env, command_name)
  psi_ball = torch.atan2(ball_vel_b[:, 1], ball_vel_b[:, 0])
  psi_cmd = torch.atan2(cmd_b[:, 1], cmd_b[:, 0])
  angle_err = (psi_ball - psi_cmd + math.pi) % (2 * math.pi) - math.pi
  reward = 1.0 - (angle_err**2) / (math.pi**2)
  return reward * (ball_vel_b.norm(dim=-1) > min_speed).float()


def ball_vel_norm(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 1.0,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """Speed matching: exp(-sharpness * (|v^cmd| - |v^b|)²). Zero when ball is stationary."""
  ball_vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  target_vel = env.command_manager.get_command(command_name)[:, :2]
  speed_err = (target_vel.norm(dim=-1) - ball_vel.norm(dim=-1)) ** 2
  reward = torch.exp(-sharpness * speed_err)
  return reward * (ball_vel.norm(dim=-1) > min_speed).float()


# ---------------------------------------------------------------------------
# Ball velocity rewards (world-frame originals)
# ---------------------------------------------------------------------------


def ball_vel_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 1.0,
) -> torch.Tensor:
  """Full XY velocity vector tracking: exp(-sharpness * |v^b - v^cmd|²)."""
  ball_vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  target_vel = env.command_manager.get_command(command_name)[:, :2]
  error_sq = ((ball_vel - target_vel) ** 2).sum(dim=-1)
  return torch.exp(-sharpness * error_sq)


def ball_vel_angle(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """Direction match: 1 - (ψ_b - ψ_cmd)²/π²."""
  ball_vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  target_vel = env.command_manager.get_command(command_name)[:, :2]
  psi_b = torch.atan2(ball_vel[:, 1], ball_vel[:, 0])
  psi_cmd = torch.atan2(target_vel[:, 1], target_vel[:, 0])
  angle_err = (psi_b - psi_cmd + math.pi) % (2 * math.pi) - math.pi
  reward = 1.0 - (angle_err**2) / (math.pi**2)
  return reward * (ball_vel.norm(dim=-1) > min_speed).float()


# ---------------------------------------------------------------------------
# Robot–ball spatial relationship
# ---------------------------------------------------------------------------


def robot_ball_yaw_body(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """Ball ahead of robot along command direction AND robot facing that way (body frame).

  e1 = 1 - dot(d_robot→ball_b, cmd_dir_b)
  e2 = 1 - d_robot→ball_b[0] / |d|  (ball in front, X-forward)
  reward = exp(-2 * (e1 + e2))
  """
  robot = env.scene["robot"]
  ball_pos_w = env.scene["ball"].data.root_link_pos_w[:, :3]
  robot_pos_w = robot.data.root_link_pos_w[:, :3]

  relative_w = ball_pos_w - robot_pos_w
  quat_w = robot.data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  d_ball_b = quat_apply(quat_conj, relative_w)[:, :2]
  d_norm = d_ball_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  d_ball_b_unit = d_ball_b / d_norm

  cmd_b = cmd_body(env, command_name)
  unit_cmd = cmd_b / cmd_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)

  e1 = 1.0 - (d_ball_b_unit * unit_cmd).sum(dim=-1)
  e2 = 1.0 - d_ball_b_unit[:, 0]

  reward = torch.exp(-2.0 * (e1 + e2))
  return reward * (cmd_b.norm(dim=-1) > min_speed).float()


def robot_ball_distance(
  env: ManagerBasedRlEnv,
  close_distance: float = 0.3,
  behind_close_penalty: float = 2.0,
  far_sharpness: float = 3.0,
  between_feet_forward_distance: float = 0.1,
  between_feet_penalty: float = 2.0,
) -> torch.Tensor:
  """Front/back-aware robot-ball proximity reward.

  Returns 1.0 when ball is close and in front, decays exponentially when far,
  and gives a low constant reward when ball is behind or between the feet.
  """
  robot = env.scene["robot"]
  ball_pos_w = env.scene["ball"].data.root_link_pos_w[:, :3]
  robot_pos_w = robot.data.root_link_pos_w[:, :3]

  relative_w = ball_pos_w - robot_pos_w
  quat_w = robot.data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  ball_b = quat_apply(quat_conj, relative_w)[:, :2]
  dist = ball_b.norm(dim=-1)

  between_feet = (dist <= close_distance) & (
    ball_b[:, 0].abs() < between_feet_forward_distance
  )
  front_close = (dist <= close_distance) & (ball_b[:, 0] >= 0.0) & ~between_feet
  behind_close = (dist <= close_distance) & (ball_b[:, 0] < 0.0) & ~between_feet

  far_excess = (dist - close_distance).clamp(min=0.0)
  far_reward = torch.exp(-far_sharpness * far_excess.pow(2))
  behind_close_reward = torch.full_like(dist, math.exp(-behind_close_penalty))
  between_feet_reward = torch.full_like(dist, math.exp(-between_feet_penalty))

  return torch.where(
    between_feet,
    between_feet_reward,
    torch.where(
      front_close,
      torch.ones_like(dist),
      torch.where(behind_close, behind_close_reward, far_reward),
    ),
  )


def robot_ball_approach_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Robot base moving toward ball fast enough (penalizes only speed deficit).

  deficit = max(0, cmd_speed - proj(robot_vel, d_robot→ball))
  reward  = exp(-deficit²)
  """
  robot = env.scene["robot"]
  ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
  robot_pos = robot.data.root_link_pos_w[:, :2]

  d_robot_ball = ball_pos - robot_pos
  d_robot_ball = d_robot_ball / d_robot_ball.norm(dim=-1, keepdim=True).clamp(min=1e-6)

  approach_vel = (robot.data.root_link_lin_vel_w[:, :2] * d_robot_ball).sum(dim=-1)
  cmd_speed = env.command_manager.get_command(command_name)[:, :2].norm(dim=-1)
  deficit = (cmd_speed - approach_vel).clamp(min=0.0)
  return torch.exp(-(deficit**2))
