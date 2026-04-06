from __future__ import annotations

import torch
from loguru import logger
from mjlab.entity import Entity, EntityData
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor

from colosseum.envs.abstraction_based_env import AbstractionBasedEnv
from colosseum.tasks.maze.mdp.grid_abstraction import GridAbstraction
from colosseum.tasks.maze.mdp.observations import (
  agent_pos_local,
  agent_to_goal_vector,
  agent_vel,
  agent_z_vel,
)


def goal_reward(
  env: ManagerBasedRlEnv,
  threshold: float = 0.2,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("root_site",)),
) -> torch.Tensor:
  """Sparse reward: 1.0 when agent is within threshold of goal."""
  distance = torch.norm(agent_to_goal_vector(env, asset_cfg), dim=1)
  return (distance < threshold).float()


def goal_distance_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("root_site",)),
) -> torch.Tensor:
  """Dense cost proportional to distance to goal. Returns exp(-distance)."""
  distance = torch.norm(agent_to_goal_vector(env, asset_cfg), dim=1)
  return torch.exp(-distance)


def wall_collisions(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalty: 1.0 for each env colliding with a wall."""
  sensor: ContactSensor = env.scene[sensor_name]
  return sensor.data.found.any(dim=-1).float()


def contact_force_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalty proportional to total contact force magnitude."""
  sensor: ContactSensor = env.scene[sensor_name]
  return sensor.data.force.norm(dim=-1).sum(dim=-1)


def z_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("root_site",)),
) -> torch.Tensor:
  """Penalty for vertical movement. Returns |vz|."""
  return agent_z_vel(env, asset_cfg).squeeze(-1).abs()


def is_healthy(
  env: ManagerBasedRlEnv,
  healthy_z_range: tuple[float, float] = (0.2, 1.0),
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """True if robot z-position is within healthy_z_range."""
  asset: Entity = env.scene[asset_cfg.name]
  z_pos = asset.data.root_link_pos_w[:, 2]
  return (z_pos >= healthy_z_range[0]) & (z_pos <= healthy_z_range[1])


def track_velocity_direction(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 4.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("root_site",)),
  use_body_frame: bool = False,
) -> torch.Tensor:
  """Direction-tracking reward: tanh(k · (v_act · v_cmd_unit) / |v_cmd|).

  +1 when velocity matches command direction, -1 when opposite, 0 when stationary.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  commanded_vel = command[:, :2]

  asset: Entity = env.scene[asset_cfg.name]
  if use_body_frame:
    actual_vel = asset.data.root_link_lin_vel_b[:, :2]
  else:
    actual_vel = agent_vel(env, asset_cfg)[:, :2]

  cmd_speed = commanded_vel.norm(dim=-1).clamp(min=1e-6)
  cmd_unit = commanded_vel / cmd_speed.unsqueeze(-1)
  projection = (actual_vel * cmd_unit).sum(dim=-1) / cmd_speed
  return torch.tanh(sharpness * projection)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward for tracking commanded angular velocity around the vertical axis."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  commanded_ang_vel = command[:, 2]
  asset: Entity = env.scene[asset_cfg.name]
  actual_ang_vel = asset.data.root_link_ang_vel_w[:, 2]
  return torch.exp(-torch.square(commanded_ang_vel - actual_ang_vel))