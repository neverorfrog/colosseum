"""Ball dribbling/kicking reward functions.

Primary task reference: Ji et al., "DribbleBot" (ICRA 2023).
Ball velocity rewards split into three terms (TABLE III):
  - ball_vel_tracking: full vector error exp(-δ|v^b - v^cmd|²)
  - ball_vel_norm:     speed matching  exp(-δ(|v^cmd| - |v^b|)²)
  - ball_vel_angle:    direction match 1 - (ψ_b - ψ_cmd)²/π²
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# ------------------------------------------------------------------
# Primary task rewards (DribbleBot TABLE III)
# ------------------------------------------------------------------


def ball_vel_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 1.0,
) -> torch.Tensor:
  """Full XY velocity vector tracking: exp(-sharpness * |v^b - v^cmd|²).

  Low weight (0.5) — hardest to achieve, but penalises both speed and
  direction error simultaneously.
  """
  ball_vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  target_vel = env.command_manager.get_command(command_name)[:, :2]  # type: ignore
  error_sq = ((ball_vel - target_vel) ** 2).sum(dim=-1)
  return torch.exp(-sharpness * error_sq)


def ball_vel_norm(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 1.0,
) -> torch.Tensor:
  """Speed matching: exp(-sharpness * (|v^cmd| - |v^b|)²).

  Rewards matching commanded speed regardless of direction.
  Prevents exploitation of direction-only rewards by kicking too hard.
  """
  ball_vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  target_vel = env.command_manager.get_command(command_name)[:, :2]  # type: ignore
  speed_err = (target_vel.norm(dim=-1) - ball_vel.norm(dim=-1)) ** 2
  return torch.exp(-sharpness * speed_err)


def ball_vel_angle(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Direction match: 1 - (ψ_b - ψ_cmd)²/π².

  Ranges from 1.0 (perfect alignment) to 0.0 (opposite direction).
  Gives partial credit for near-correct directions.
  """
  ball_vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  target_vel = env.command_manager.get_command(command_name)[:, :2]  # type: ignore
  psi_b = torch.atan2(ball_vel[:, 1], ball_vel[:, 0])
  psi_cmd = torch.atan2(target_vel[:, 1], target_vel[:, 0])
  angle_err = (psi_b - psi_cmd + math.pi) % (2 * math.pi) - math.pi
  return 1.0 - (angle_err**2) / (math.pi**2)


# ------------------------------------------------------------------
# Phase-schedule feet rewards (DribbleBot TABLE III)
# ------------------------------------------------------------------


def swing_phase_schedule(
  env: ManagerBasedRlEnv,
  phase_command_name: str,
  sensor_name: str,
  sharpness: float = 0.1,
) -> torch.Tensor:
  """During swing phase, penalize foot-ground contact force.

  reward = sum_feet( [1 - κ] * exp(-sharpness * |f_foot|²) )
  κ = (1 + cos(φ)) / 2  →  0 in full swing, 1 in full stance.
  """
  phase = env.command_manager.get_command(
    phase_command_name
  )  # (N, 4): [cL, cR, sL, sR]
  kappa = (1.0 + phase[:, :2]) / 2.0  # (N, 2)
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None
  force_sq = (contact_sensor.data.force**2).sum(dim=-1)  # (N, 2)
  return ((1.0 - kappa) * torch.exp(-sharpness * force_sq)).sum(dim=-1)


def stance_phase_schedule(
  env: ManagerBasedRlEnv,
  phase_command_name: str,
  asset_cfg: SceneEntityCfg,
  sharpness: float = 0.1,
) -> torch.Tensor:
  """During stance phase, penalize foot XY sliding velocity.

  reward = sum_feet( κ * exp(-sharpness * |v_foot_xy|²) )
  """
  phase = env.command_manager.get_command(phase_command_name)  # (N, 4)
  kappa = (1.0 + phase[:, :2]) / 2.0  # (N, 2)
  asset: Entity = env.scene[asset_cfg.name]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # (N, 2, 2)
  vel_sq = (foot_vel_xy**2).sum(dim=-1)  # (N, 2)
  return (kappa * torch.exp(-sharpness * vel_sq)).sum(dim=-1)


# ------------------------------------------------------------------
# Pose deviation
# ------------------------------------------------------------------


def pose_deviation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  std: float,
) -> torch.Tensor:
  """Penalize joint deviation from default pose: exp(-mean(error²/std²)).

  Matches variable_posture's formula with a single scalar std.
  Register separate terms for arms and legs with different std and weight.
  Smaller std = tighter constraint.
  """
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.exp(-torch.mean(torch.square(q - q_default) / (std**2), dim=1))


# ------------------------------------------------------------------
# Robot–ball spatial relationship
# ------------------------------------------------------------------


def robot_ball_distance(
  env: ManagerBasedRlEnv,
  sharpness: float = 2.0,
) -> torch.Tensor:
  """exp(-sharpness * ||robot_xy - ball_xy||²). Dense within ~0.7m."""
  robot_pos = env.scene["robot"].data.root_link_pos_w[:, :2]
  ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
  dist_sq = ((ball_pos - robot_pos) ** 2).sum(dim=-1)
  return torch.exp(-sharpness * dist_sq)


def robot_ball_yaw(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """Ball ahead of robot along command direction AND robot body faces that direction.

  e1 = 1 - dot(d_robot→ball, d̂_cmd)   (ball not ahead)
  e2 = 1 - dot(d_robot→ball, body_fwd) (robot not facing cmd)
  reward = exp(-2 * (e1 + e2))
  """
  robot = env.scene["robot"]
  ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
  robot_pos = robot.data.root_link_pos_w[:, :2]

  target_vel = env.command_manager.get_command(command_name)[:, :2]  # type: ignore
  unit_cmd = target_vel / target_vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)

  d_robot_ball = ball_pos - robot_pos
  d_robot_ball = d_robot_ball / d_robot_ball.norm(dim=-1, keepdim=True).clamp(min=1e-6)

  e1 = 1.0 - (d_robot_ball * unit_cmd).sum(dim=-1)

  q = robot.data.root_link_quat_w
  yaw = torch.atan2(
    2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
    1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
  )
  body_fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
  e2 = 1.0 - (d_robot_ball * body_fwd).sum(dim=-1)

  reward = torch.exp(-2.0 * (e1 + e2))
  return reward * (target_vel.norm(dim=-1) > min_speed).float()


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
  cmd_speed = env.command_manager.get_command(command_name)[:, :2].norm(dim=-1)  # type: ignore
  deficit = (cmd_speed - approach_vel).clamp(min=0.0)
  return torch.exp(-(deficit**2))
