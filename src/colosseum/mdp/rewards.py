"""Layer 1: Universal reward functions (robot-agnostic).

These training wrapper functions work across all robots and tasks.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_apply_inverse,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_height_penalty(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base height deviation from a target (use negative weight).

  Quadratic: (base_height - target_height)². Unbounded — keeps gradient even
  at large deviations. Height is measured above the env origin (terrain floor).
  """
  asset: Entity = env.scene[asset_cfg.name]
  base_height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return torch.square(base_height - target_height)


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


class pose_deviation_penalty:
  """Penalize joint deviation from default pose with per-joint weights.

  Supports three velocity regimes (standing/walking/running) with separate
  weight dicts, mirroring variable_posture's API but using a weighted sum
  instead of mean-exp so each joint's signal is never diluted by others.

  Weighted sum: sum(w_i * (q_i - q_default_i)²). Use negative term weight.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    def _resolve(key: str) -> torch.Tensor:
      _, _, weights = resolve_matching_names_values(
        data=cfg.params[key],
        list_of_strings=joint_names,
      )
      return torch.tensor(weights, device=env.device, dtype=torch.float32)

    self.weights_standing = _resolve("weights_standing")
    self.weights_walking = (
      _resolve("weights_walking")
      if "weights_walking" in cfg.params
      else self.weights_standing
    )
    self.weights_running = (
      _resolve("weights_running")
      if "weights_running" in cfg.params
      else self.weights_walking
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    weights_standing: dict[str, float],
    weights_walking: dict[str, float] | None = None,
    weights_running: dict[str, float] | None = None,
    walking_threshold: float = 0.05,
    running_threshold: float = 1.0,
    command_name: str | None = None,
  ) -> torch.Tensor:
    del weights_standing, weights_walking, weights_running
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error_sq = torch.square(q - q_default)  # (N, J)

    if command_name is not None:
      cmd = env.command_manager.get_command(command_name)
      speed = torch.norm(cmd[:, :2], dim=-1)  # (N,)
      standing = (speed <= walking_threshold).float()
      running = (speed > running_threshold).float()
      walking = 1.0 - standing - running
      weights = (
        self.weights_standing * standing.unsqueeze(1)
        + self.weights_walking * walking.unsqueeze(1)
        + self.weights_running * running.unsqueeze(1)
      )  # (N, J)
    else:
      weights = self.weights_standing.unsqueeze(0)

    return torch.sum(error_sq * weights, dim=1)


def dof_vel_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize sum of squared joint velocities (use negative weight)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)


def dof_acc_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize sum of squared joint accelerations (use negative weight)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=-1)


def feet_distance_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_dist: float = 0.2,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize when feet are closer than min_dist (XY plane).

  Returns a value in [0, 1]: 0 when feet are at least min_dist apart,
  1 when fully overlapping. Normalized so the weight directly sets the
  maximum penalty regardless of min_dist.

  If command_name is given, the penalty is gated on linear velocity command
  magnitude — zero when standing still so the policy never tries to reposition
  grounded feet while stopped.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :3]  # (N, 2, 3)
  base_pos_w = asset.data.root_link_pos_w[:, :3].unsqueeze(1)  # (N, 1, 3)
  quat_w = asset.data.root_link_quat_w  # (N, 4)
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)

  left_b = quat_apply(quat_conj, foot_pos_w[:, 0] - base_pos_w[:, 0])  # (N, 3)
  right_b = quat_apply(quat_conj, foot_pos_w[:, 1] - base_pos_w[:, 0])  # (N, 3)
  dist = (left_b[:, 1] - right_b[:, 1]).abs()  # Y axis
  penalty = (min_dist - dist).clamp(min=0.0) / min_dist

  if command_name is not None:
    cmd = env.command_manager.get_command(command_name)
    walking = (torch.norm(cmd[:, :2], dim=-1) > command_threshold).float()
    penalty = penalty * walking

  return penalty


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


def feet_yaw_diff_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize the yaw difference between the two feet (use negative weight).

  Keeps the feet parallel (no splay / toe-in / toe-out relative to each other).
  The signed yaw difference is wrapped to (-pi, pi] so a half-turn apart is the
  maximum penalty, then squared. Port of `_reward_feet_yaw_diff` in t1.py.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_quats = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # (N, 2, 4)
  _, _, yaw_l = euler_xyz_from_quat(foot_quats[:, 0])
  _, _, yaw_r = euler_xyz_from_quat(foot_quats[:, 1])
  diff = (yaw_l - yaw_r + math.pi) % (2 * math.pi) - math.pi
  return diff.square()


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

  def _cubic_bezier(
    y0: torch.Tensor, y1: torch.Tensor, t: torch.Tensor
  ) -> torch.Tensor:
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
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Reward foot height tracking against a cubic Bézier gait profile.

  Reads raw phase angles from the GaitPhaseCommand term (not the cos/sin
  observation) and compares actual terrain-relative foot clearance to the
  Bézier target. When command_name is set, zeroes the reward for standing
  envs (‖cmd_xy‖ ≤ command_threshold) so static_stance can own that regime.
  """
  gait_term = env.command_manager.get_term(phase_command_name)
  phi = gait_term.phase  # (N, 2), raw angles in [-π, π]
  height_sensor = env.scene[height_sensor_name]
  foot_heights = height_sensor.data.heights  # (N, 2)
  expected = _bezier_foot_height(phi, swing_height)
  error = torch.square(foot_heights - expected).sum(dim=-1)
  reward = torch.exp(-error / tracking_sigma)
  if command_name is not None:
    cmd = env.command_manager.get_command(command_name)
    moving = (torch.norm(cmd[:, :2], dim=-1) > command_threshold).float()
    reward = reward * moving
  return reward


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
  reward = (kappa * torch.exp(-sharpness * force_sq)).sum(dim=-1)
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
  reward = ((1.0 - kappa) * torch.exp(-sharpness * vel_sq)).sum(dim=-1)
  if command_name is not None:
    cmd = env.command_manager.get_command(command_name)
    moving = (torch.norm(cmd[:, :2], dim=-1) > command_threshold).float()
    reward = reward * moving
  return reward


def feet_slip(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  sensor_name: str,
  contact_threshold: float = 0.5,
) -> torch.Tensor:
  """Penalize foot XY sliding while the foot is in contact (use negative weight).

  Gated on contact (‖force‖ > contact_threshold), not on command, so it discourages
  dragging a planted foot in both walking and standing regimes while leaving the
  policy free to lift a foot to take a recovery step. Replaces the command-gated
  static_stance, whose all-velocity penalty froze recovery stepping at standstill.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None
  contact = contact_sensor.data.force.norm(dim=-1) > contact_threshold  # (N, 2)
  asset: Entity = env.scene[asset_cfg.name]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # (N, 2, 2)
  vel_sq = (foot_vel_xy**2).sum(dim=-1)  # (N, 2)
  return (vel_sq * contact.float()).sum(dim=-1)


def static_stance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize foot XY sliding when velocity command ≈ 0 (use negative weight).

  Penalizes all foot velocity regardless of contact state. The outer `standing`
  gate already zeroes this during walking, so the contact gate is not needed —
  and would create a perverse incentive to lift feet while standing.
  """
  cmd = env.command_manager.get_command(command_name)
  standing = (torch.norm(cmd[:, :2], dim=-1) <= command_threshold).float()
  asset: Entity = env.scene[asset_cfg.name]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # (N, 2, 2)
  vel_sq = (foot_vel_xy**2).sum(dim=-1).mean(dim=-1)  # (N,) mean over feet
  return vel_sq * standing


class feet_no_slip:
  """Penalize a planted foot's drift from where it first touched down (neg weight).

  Velocity-based slip penalties (both this repo's L2 ``feet_slip`` and booster_gym's
  L2 ``_reward_feet_slip``) have a gradient that vanishes quadratically as foot
  velocity -> 0, so they are blind to a very slow standing creep (~1e-5 m/s) that
  still drives stick-slip vibration on rubber soles. This term anchors the foot's
  xy position at first contact and penalizes the *accumulated* L1 displacement from
  that anchor while the foot stays in contact:

    - L1 -> non-vanishing gradient: "perfectly still" is consistently preferred
      over "creeping", down to arbitrarily small velocities.
    - anchored displacement -> escalating pressure: sustained creep grows the
      offset every step, so the penalty climbs the longer the foot drifts.

  Contact-gated (not command-gated): a lifted foot is masked out and re-anchors on
  the next touchdown, so recovery stepping at standstill stays free (unlike the old
  command-gated ``static_stance``). The anchor re-initializes on episode reset, so
  the first post-reset step sees zero displacement (no stale-anchor spike).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._env = env
    self.anchor_xy: torch.Tensor | None = None
    self.was_contact: torch.Tensor | None = None

  def _ensure(self, n_feet: int, device) -> None:
    if self.anchor_xy is None:
      n = self._env.num_envs
      self.anchor_xy = torch.zeros(n, n_feet, 2, device=device)
      self.was_contact = torch.zeros(n, n_feet, dtype=torch.bool, device=device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    sensor_name: str,
    contact_threshold: float = 0.5,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.force is not None
    contact = contact_sensor.data.force.norm(dim=-1) > contact_threshold  # (N, F)
    asset: Entity = env.scene[asset_cfg.name]
    foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # (N, F, 2)
    self._ensure(foot_xy.shape[1], foot_xy.device)
    assert self.anchor_xy is not None and self.was_contact is not None
    # Re-anchor feet that just made contact (first contact, or first step post-reset).
    new_contact = contact & ~self.was_contact  # (N, F)
    self.anchor_xy[new_contact] = foot_xy[new_contact]
    disp = (foot_xy - self.anchor_xy).abs().sum(dim=-1)  # (N, F) L1 drift
    self.was_contact.copy_(contact)
    return (disp * contact.float()).sum(dim=-1)  # (N,)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self.was_contact is None:
      return
    if env_ids is None:
      env_ids = slice(None)
    self.was_contact[env_ids] = False


class arm_swing_penalty:
  """Penalize arm motion that is not anti-phase with hip motion.

  For each side: (shoulder_pitch_offset + hip_pitch_offset)²
  Scaled by forward speed command so the penalty is stronger at higher speeds
  and zero at standstill.

  This encourages natural arm swing: when the hip pitches forward, the
  shoulder pitches backward (opposite direction), like human walking.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    jids = cfg.params["asset_cfg"].joint_ids
    if isinstance(jids, slice):
      jids = list(range(jids.start or 0, jids.stop or len(asset.joint_names)))
    joint_names = [asset.joint_names[i] for i in jids]
    name_to_local = {n: i for i, n in enumerate(joint_names)}

    _, lsp = asset.find_joints("Left_Shoulder_Pitch", joint_names)
    _, lhp = asset.find_joints("Left_Hip_Pitch", joint_names)
    _, rsp = asset.find_joints("Right_Shoulder_Pitch", joint_names)
    _, rhp = asset.find_joints("Right_Hip_Pitch", joint_names)

    self.ls_idx = name_to_local[lsp[0]]
    self.lh_idx = name_to_local[lhp[0]]
    self.rs_idx = name_to_local[rsp[0]]
    self.rh_idx = name_to_local[rhp[0]]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.05,
  ) -> torch.Tensor:
    del command_threshold
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    ls_off = q[:, self.ls_idx] - q_default[:, self.ls_idx]
    lh_off = q[:, self.lh_idx] - q_default[:, self.lh_idx]
    rs_off = q[:, self.rs_idx] - q_default[:, self.rs_idx]
    rh_off = q[:, self.rh_idx] - q_default[:, self.rh_idx]

    left_error = torch.square(ls_off + lh_off)
    right_error = torch.square(rs_off + rh_off)

    cmd = env.command_manager.get_command(command_name)
    speed = torch.abs(cmd[:, 0])

    return (left_error + right_error) * speed


def arm_phase(
  env: ManagerBasedRlEnv,
  phase_command_name: str,
  asset_cfg: SceneEntityCfg,
  swing_amplitude: float = 0.25,
  max_speed: float = 1.5,
  tracking_sigma: float = 0.05,
  command_name: str | None = None,
) -> torch.Tensor:
  """Reward shoulder pitch tracking a cosine arm-swing profile.

  Contralateral coupling: left arm tracks right foot phase, right arm tracks
  left foot phase.
      target = default_shoulder_pitch - amplitude(speed) * cos(phi_contralateral)

  The oscillation is centered on the default pose (-cos sweeps [-1, +1]), so the
  shoulder swings q_default ± amplitude — q_default is the swing center, never an
  edge. A forward walking posture belongs in HOME_QPOS, not here.

  amplitude and tracking_sigma both scale with speed up to max_speed. The reward
  itself is scaled by the same speed ramp (speed_scale), so it fades continuously
  to zero at standstill instead of switching off at a threshold — no discontinuity
  for the policy to jerk against on stop.

  asset_cfg must resolve [Left_Shoulder_Pitch, Right_Shoulder_Pitch] in
  that order, matching gait phase order (left=0, right=1).
  """
  gait_term = env.command_manager.get_term(phase_command_name)
  phi = gait_term.phase  # (N, 2): col0=left foot, col1=right foot

  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (N, 2)
  q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]  # (N, 2)

  phi_contra = phi[:, [1, 0]]  # contralateral coupling

  if command_name is not None:
    cmd = env.command_manager.get_command(command_name)
    speed = torch.norm(cmd[:, :2], dim=-1)  # (N,)
    speed_scale = torch.clamp(speed / max_speed, 0.0, 1.0)  # (N,)
    effective_amplitude = (swing_amplitude * speed_scale).unsqueeze(-1)  # (N, 1)
    effective_sigma = tracking_sigma / (
      1.0 + speed_scale
    )  # (N,), tighter at high speed
  else:
    effective_amplitude = swing_amplitude
    effective_sigma = tracking_sigma

  target = q_default - effective_amplitude * torch.cos(phi_contra)  # (N, 2)
  error = torch.sum(torch.square(q - target), dim=-1)  # (N,)
  reward = torch.exp(-error / effective_sigma)  # (N,)

  if command_name is not None:
    reward = reward * speed_scale
  return reward


def track_linear_velocity_filtered(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
) -> torch.Tensor:
  """Track commanded base linear velocity using the EMA-filtered velocity.

  Mirrors t1.py: reads ``filtered_lin_vel`` off the curriculum command term so
  that marching-in-place (which filters to ~0) earns no tracking reward, forcing
  sustained directed locomotion. Same Gaussian kernel as mjlab's raw version.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  filtered = env.command_manager.get_term(command_name).filtered_lin_vel
  xy_error = torch.sum(torch.square(command[:, :2] - filtered[:, :2]), dim=1)
  z_error = torch.square(filtered[:, 2])
  return torch.exp(-(xy_error + z_error) / std**2)


def track_angular_velocity_filtered(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
) -> torch.Tensor:
  """Track commanded base yaw rate using the EMA-filtered angular velocity.

  Filtered counterpart of mjlab's ``track_angular_velocity`` (see
  ``track_linear_velocity_filtered``).
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  filtered = env.command_manager.get_term(command_name).filtered_ang_vel
  z_error = torch.square(command[:, 2] - filtered[:, 2])
  xy_error = torch.sum(torch.square(filtered[:, :2]), dim=1)
  return torch.exp(-(z_error + xy_error) / std**2)
