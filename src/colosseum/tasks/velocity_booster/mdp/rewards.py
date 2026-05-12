"""BoosterGym-style foot and arm reward functions.

These replace the cubic Bezier foot-height prescription with contact-outcome
rewards.  The policy discovers its own stepping style instead of being forced
onto a template trajectory.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def feet_swing(
  env: ManagerBasedRlEnv,
  phase_command_name: str,
  sensor_name: str,
  swing_half_width: float = 0.2 * math.pi,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Reward per foot that is airborne during its swing window (booster_gym style).

  Each foot has its own phase angle φ ∈ [-π, π] where φ≈0 is peak swing.
  The swing window is |φ| < swing_half_width.  A foot earns +1 if it is inside
  its swing window AND NOT in ground contact.  Maximum 2 per env.

  Args:
      phase_command_name: Name of the GaitPhaseCommand term.
      sensor_name: Name of the foot contact sensor.
      swing_half_width: Half-width of the swing window in radians (default
          0.2π ≈ 0.628 rad, matching booster_gym's swing_period=0.2).
      command_name: If set, zeroes reward when ‖cmd_xy‖ ≤ command_threshold.
      command_threshold: Velocity norm below which the env is "standing".

  Returns:
      Tensor [num_envs] in [0, 1, 2].
  """
  phase = env.command_manager.get_term(phase_command_name).phase  # (N, 2)
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None
  in_contact = contact_sensor.data.force[:, :, 2].abs() > 0.01  # (N, 2)

  in_swing = phase.abs() < swing_half_width  # (N, 2)

  reward = (in_swing & ~in_contact).float().sum(dim=-1)  # (N,)

  if command_name is not None:
    cmd = env.command_manager.get_command(command_name)
    moving = (torch.norm(cmd[:, :2], dim=-1) > command_threshold).float()
    reward = reward * moving
  return reward


def arm_swing(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
) -> torch.Tensor:
  """Penalize shoulder pitch + hip pitch moving in the same direction.

  Booster_gym's _reward_arm_swing: enforces anti-phase arm-leg coupling.
  For each side, (shoulder_pitch_offset + hip_pitch_offset)² is penalized,
  scaled by the commanded forward velocity magnitude.

  Args:
      asset_cfg: Must resolve exactly 4 joints in order:
          Left_Shoulder_Pitch, Left_Hip_Pitch,
          Right_Shoulder_Pitch, Right_Hip_Pitch.
      command_name: Twist command name for forward velocity scaling.

  Returns:
      Tensor [num_envs] — combined left+right anti-phase error.
  """
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (N, 4)
  q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]  # (N, 4)
  offset = q - q_default  # (N, 4)

  # Indices: 0 = L_Shoulder, 1 = L_Hip, 2 = R_Shoulder, 3 = R_Hip
  left_error = (offset[:, 0] + offset[:, 1]).square()
  right_error = (offset[:, 2] + offset[:, 3]).square()

  cmd = env.command_manager.get_command(command_name)
  cmd_scale = cmd[:, 0].abs()

  return (left_error + right_error) * cmd_scale


def feet_slip(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize horizontal foot velocity while the foot is in ground contact.

  Booster_gym's _reward_feet_slip: ‖v_foot_xy‖² when contact, summed over
  both feet.  Zeroed for the first simulation step to avoid reset artefacts.

  Args:
      asset_cfg: Must resolve site_ids pointing to the two foot sites.
      sensor_name: Name of the foot contact sensor.

  Returns:
      Tensor [num_envs] — summed XY velocity magnitude during contact.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # (N, 2, 2)
  vel_sq = (foot_vel_xy**2).sum(dim=-1)  # (N, 2)

  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None
  in_contact = contact_sensor.data.force[:, :, 2].abs() > 0.01  # (N, 2)

  slip = (vel_sq * in_contact.float()).sum(dim=-1)  # (N,)

  first_step = env.episode_length_buf <= 1
  slip = slip * (~first_step).float()
  return slip
