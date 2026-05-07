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

  Returns a value in [0, 1]: 0 when feet are at least min_dist apart,
  1 when fully overlapping. Normalized so the weight directly sets the
  maximum penalty regardless of min_dist.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :3]  # (N, 2, 3)
  base_pos_w = asset.data.root_link_pos_w[:, :3].unsqueeze(1)  # (N, 1, 3)
  quat_w = asset.data.root_link_quat_w  # (N, 4)
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)

  left_b = quat_apply(quat_conj, foot_pos_w[:, 0] - base_pos_w[:, 0])  # (N, 3)
  right_b = quat_apply(quat_conj, foot_pos_w[:, 1] - base_pos_w[:, 0])  # (N, 3)
  dist = (left_b[:, 1] - right_b[:, 1]).abs()  # Y axis
  return (min_dist - dist).clamp(min=0.0) / min_dist


def foot_orientation_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize feet tilted away from flat in world frame.

  Projects gravity into each foot's local frame. A flat foot gives XY components
  of zero; any tilt (from hip roll, knee valgus, ankle — any joint in the chain)
  makes them nonzero. Captures what ankle-angle-based pose rewards miss.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_quats = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # (N, K, 4)
  gravity_w = asset.data.gravity_vec_w  # (3,)
  penalty = torch.zeros(foot_quats.shape[0], device=foot_quats.device)
  for i in range(foot_quats.shape[1]):
    g_local = quat_apply_inverse(foot_quats[:, i], gravity_w)  # (N, 3)
    penalty = penalty + g_local[:, :2].square().sum(dim=-1).sqrt()
  return penalty


def orientation_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat base orientation: sum(projected_gravity_xy²).

  Unbounded quadratic — keeps a strong gradient even at large tilt angles,
  unlike the exp-shaped upright reward which saturates near zero when the
  robot is badly tilted.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    gravity_w = asset.data.gravity_vec_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def _bezier_foot_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
  """Cubic Bézier foot height profile keyed to gait phase φ ∈ [-π, π].

  x ∈ [0, 0.5]: foot rises from 0 → swing_height (stance → swing).
  x ∈ [0.5, 1]: foot falls from swing_height → 0 (swing → stance).
  At φ=π (standing snap value): x=1.0, height=0 (feet on ground).
  """
  def _cubic_bezier(y0: torch.Tensor, y1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    bezier = t**3 + 3.0 * t**2 * (1.0 - t)
    return y0 + (y1 - y0) * bezier

  x = (phi + math.pi) / (2.0 * math.pi)
  h = torch.full_like(phi, swing_height)
  z = torch.zeros_like(phi)
  rising = _cubic_bezier(z, h, 2.0 * x)
  falling = _cubic_bezier(h, z, 2.0 * x - 1.0)
  return torch.where(x <= 0.5, rising, falling)


def feet_phase(
  env: ManagerBasedRlEnv,
  phase_command_name: str,
  height_sensor_name: str,
  swing_height: float = 0.09,
  tracking_sigma: float = 0.008,
) -> torch.Tensor:
  """Reward foot height tracking against a cubic Bézier gait profile.

  Reads raw phase angles from the GaitPhaseCommand term (not the cos/sin
  observation) and compares actual terrain-relative foot clearance to the
  Bézier target. Handles standing naturally: when the phase is snapped to π
  the target height is 0, so both feet are driven to the ground with no
  separate gating needed.
  """
  gait_term = env.command_manager.get_term(phase_command_name)
  phi = gait_term.phase  # (N, 2), raw angles in [-π, π]
  height_sensor = env.scene[height_sensor_name]
  foot_heights = height_sensor.data.heights  # (N, 2)
  expected = _bezier_foot_height(phi, swing_height)
  error = torch.square(foot_heights - expected).sum(dim=-1)
  return torch.exp(-error / tracking_sigma)


def swing_phase_schedule(
  env: ManagerBasedRlEnv,
  phase_command_name: str,
  sensor_name: str,
  sharpness: float = 0.1,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """During swing phase, penalize foot-ground contact force.

  reward = sum_feet( [1 - κ] * exp(-sharpness * |f_foot|²) )
  κ = (1 + cos(φ)) / 2  →  0 in full swing, 1 in full stance.
  Zeroed when ||cmd_xy|| < command_threshold (standing envs).
  """
  phase = env.command_manager.get_command(phase_command_name)  # (N, 4)
  kappa = (1.0 + phase[:, :2]) / 2.0  # (N, 2)
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None
  force_sq = (contact_sensor.data.force**2).sum(dim=-1)  # (N, 2)
  reward = ((1.0 - kappa) * torch.exp(-sharpness * force_sq)).sum(dim=-1)
  if command_name is not None:
    cmd = env.command_manager.get_command(command_name)
    moving = (torch.norm(cmd[:, :2], dim=-1) > command_threshold).float()
    reward = reward * moving
  return reward


def stance_phase_schedule(
  env: ManagerBasedRlEnv,
  phase_command_name: str,
  asset_cfg: SceneEntityCfg,
  sharpness: float = 0.1,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """During stance phase, penalize foot XY sliding velocity.

  reward = sum_feet( κ * exp(-sharpness * |v_foot_xy|²) )
  Zeroed when ||cmd_xy|| < command_threshold (standing envs).
  """
  phase = env.command_manager.get_command(phase_command_name)  # (N, 4)
  kappa = (1.0 + phase[:, :2]) / 2.0  # (N, 2)
  asset: Entity = env.scene[asset_cfg.name]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # (N, 2, 2)
  vel_sq = (foot_vel_xy**2).sum(dim=-1)  # (N, 2)
  reward = (kappa * torch.exp(-sharpness * vel_sq)).sum(dim=-1)
  if command_name is not None:
    cmd = env.command_manager.get_command(command_name)
    moving = (torch.norm(cmd[:, :2], dim=-1) > command_threshold).float()
    reward = reward * moving
  return reward
