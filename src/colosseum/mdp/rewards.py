"""Layer 1: Universal reward functions (robot-agnostic).

These training wrapper functions work across all robots and tasks.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-xy_squared / std**2)


def pose_deviation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  std: float,
) -> torch.Tensor:
  """Penalize joint deviation from default pose: exp(-mean(error²/std²)).

  Register separate terms for arms and legs with different std and weight.
  Smaller std = tighter constraint.
  """
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.exp(-torch.mean(torch.square(q - q_default) / (std**2), dim=1))


def feet_distance_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_dist: float = 0.2,
) -> torch.Tensor:
  """Penalize when feet are closer than min_dist (XY plane).

  penalty = clip(min_dist - ||p_left_xy - p_right_xy||, 0, min_dist)
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :3]  # (N, 2, 3)
  base_pos_w = asset.data.root_link_pos_w[:, :3].unsqueeze(1)  # (N, 1, 3)
  quat_w = asset.data.root_link_quat_w  # (N, 4)
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)

  left_b = quat_apply(quat_conj, foot_pos_w[:, 0] - base_pos_w[:, 0])  # (N, 3)
  right_b = quat_apply(quat_conj, foot_pos_w[:, 1] - base_pos_w[:, 0])  # (N, 3)
  dist = (left_b[:, 1] - right_b[:, 1]).abs()  # Y axis
  return (min_dist - dist).clamp(min=0.0, max=min_dist)


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
  phase = env.command_manager.get_command(phase_command_name)  # (N, 4)
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
