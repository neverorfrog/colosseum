"""CaT constraint functions for mjlab environments.

Each function returns a tensor of shape [num_envs] or [num_envs, num_dims]
where positive values indicate violation magnitude and values <= 0 are safe.
These are passed directly to ConstraintTermCfg.func.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def joint_torque(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Violation when |applied_torque| exceeds limit.

  Returns: [num_envs, num_joints]
  """
  data = env.scene[asset_cfg.name].data
  return torch.abs(data.qfrc_actuator[:, asset_cfg.joint_ids]) - limit


def joint_velocity(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Violation when |joint_vel| exceeds limit.

  Returns: [num_envs, num_joints]
  """
  data = env.scene[asset_cfg.name].data
  return torch.abs(data.joint_vel[:, asset_cfg.joint_ids]) - limit


def joint_acceleration(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Violation when |joint_acc| exceeds limit.

  Returns: [num_envs, num_joints]
  """
  data = env.scene[asset_cfg.name].data
  return torch.abs(data.joint_acc[:, asset_cfg.joint_ids]) - limit


def joint_position(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Violation when |joint_pos| exceeds limit (absolute position).

  Returns: [num_envs, num_joints]
  """
  data = env.scene[asset_cfg.name].data
  return torch.abs(data.joint_pos[:, asset_cfg.joint_ids]) - limit


def joint_range(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Violation when |joint_pos - default_joint_pos| exceeds limit (relative range).

  Returns: [num_envs, num_joints]
  """
  data = env.scene[asset_cfg.name].data
  return (
    torch.abs(data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids])
    - limit
  )


def base_orientation(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Violation when the horizontal component of projected gravity exceeds limit.

  Returns: [num_envs]
  """
  data = env.scene[asset_cfg.name].data
  return torch.norm(data.projected_gravity_b[:, :2], dim=1) - limit


def upsidedown(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Violation when the robot is upside-down (gravity_b z-component > limit).

  Use limit=0.0 for immediate hard termination on any inversion.

  Returns: [num_envs]
  """
  data = env.scene[asset_cfg.name].data
  return (data.projected_gravity_b[:, 2] - limit).clamp(min=0.0)


def min_base_height(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Violation when the base height drops below limit.

  Returns: [num_envs]
  """
  data = env.scene[asset_cfg.name].data
  return limit - data.root_link_pos_w[:, 2]


def contact(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Violation when any of the specified bodies experience contact force > 1 N.

  asset_cfg should point to a contact sensor, not the robot asset.

  Returns: [num_envs]
  """
  sensor = env.scene[asset_cfg.name]
  net_forces = sensor.data.net_forces_w_history  # [num_envs, history, num_bodies, 3]
  max_force = torch.max(
    torch.norm(net_forces[:, :, asset_cfg.body_ids], dim=-1), dim=1
  ).values  # [num_envs, num_bodies]
  return torch.any(max_force > 1.0, dim=1).float()


def foot_contact_force(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Violation when peak foot contact force exceeds limit.

  asset_cfg should point to a contact sensor.

  Returns: [num_envs, num_feet]
  """
  sensor = env.scene[asset_cfg.name]
  net_forces = sensor.data.net_forces_w_history
  peak = torch.max(
    torch.norm(net_forces[:, :, asset_cfg.body_ids], dim=-1), dim=1
  ).values  # [num_envs, num_feet]
  return peak - limit


def air_time(
  env: ManagerBasedRlEnv,
  limit: float,
  velocity_deadzone: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Violation when a foot touches down after being airborne longer than limit.

  Only active when the velocity command magnitude exceeds velocity_deadzone.
  asset_cfg should point to a contact sensor.

  Returns: [num_envs, num_feet]
  """
  sensor = env.scene[asset_cfg.name]
  touchdown = sensor.compute_first_contact(env.step_dt)[:, asset_cfg.body_ids]
  last_air_time = sensor.data.last_air_time[:, asset_cfg.body_ids]
  cmd_norm = torch.norm(
    env.command_manager.get_command("base_velocity")[:, :3], dim=1
  )
  moving = (cmd_norm > velocity_deadzone).float().unsqueeze(1)
  return (limit - last_air_time) * touchdown.float() * moving


def sensor_any_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Violation when any contact force in the sensor exceeds force_threshold.

  Works with any ContactSensor regardless of reduce mode or history length.

  Returns: [num_envs] — 1.0 if violated, 0.0 otherwise.
  """
  sensor = env.scene[sensor_name]
  force = sensor.data.force
  if force is None:
    return torch.zeros(env.num_envs, device=env.device)
  # If last dim is 3, it is a force vector — compute magnitude first.
  if force.shape[-1] == 3:
    force = torch.norm(force, dim=-1)
  # Flatten all slot/history dims and take the max per env.
  max_force = force.reshape(env.num_envs, -1).max(dim=-1).values
  return (max_force > force_threshold).float()


def sensor_peak_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  limit: float,
) -> torch.Tensor:
  """Violation magnitude when peak contact force in sensor exceeds limit.

  Returns [num_envs, num_slots] so CaT can normalize and take max per env.
  """
  sensor = env.scene[sensor_name]
  force = sensor.data.force
  if force is None:
    return torch.zeros(env.num_envs, 1, device=env.device)
  if force.shape[-1] == 3:
    force = torch.norm(force, dim=-1)
  return force.reshape(env.num_envs, -1) - limit


def pose_deviation_cstr(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  base_limit: float,
  command_name: str,
) -> torch.Tensor:
  """Violation when mean squared joint deviation from default exceeds an adaptive limit.

  The limit relaxes linearly with command speed so the robot can deviate
  more when actively executing a task (e.g. dribbling at high speed) than
  when standing still.

    adaptive_limit = base_limit * (1 + 2 * speed)

  At speed = 0 the limit equals base_limit; at speed = 1 it triples.

  Returns: [num_envs]
  """
  asset = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  deviation = torch.mean(torch.square(q - q_default), dim=1)  # [N]

  cmd = env.command_manager.get_command(command_name)
  speed = cmd[:, :2].norm(dim=-1).clamp(0.0, 1.0)  # [N]
  adaptive_limit = base_limit * (1.0 + 2.0 * speed)

  return deviation - adaptive_limit
