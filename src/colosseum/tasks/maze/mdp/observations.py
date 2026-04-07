from __future__ import annotations

import torch
from mjlab.entity import Entity, EntityData
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg


def agent_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names="root_site"),
) -> torch.Tensor:
  """Agent position in world coordinates. Returns [num_envs, 2]."""
  asset: Entity = env.scene[asset_cfg.name]
  site_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids]
  if site_pos_w.ndim == 3:
    site_pos_w = site_pos_w[:, 0]
  return site_pos_w[:, :2]


def agent_pos_local(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names="root_site"),
) -> torch.Tensor:
  """Agent position in environment-local (maze-centered) coordinates. Returns [num_envs, 2]."""
  return agent_pos(env, asset_cfg) - env.scene.env_origins[:, :2]


def agent_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names="root_site"),
) -> torch.Tensor:
  """Agent XY velocity in world frame. Returns [num_envs, 2]."""
  asset: Entity = env.scene[asset_cfg.name]
  data: EntityData = asset.data
  site_vel_w = data.site_vel_w[:, asset_cfg.site_ids, :2]
  if site_vel_w.ndim == 3:
    site_vel_w = site_vel_w[:, 0]
  return site_vel_w


def agent_vel_body(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Agent linear velocity in body frame (XY). Returns [num_envs, 2]."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b[:, :2]


def agent_z_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names="root_site"),
) -> torch.Tensor:
  """Agent vertical velocity. Returns [num_envs, 1]."""
  asset: Entity = env.scene[asset_cfg.name]
  site_vel_w = asset.data.site_vel_w[:, asset_cfg.site_ids, 2]
  if site_vel_w.ndim == 2:
    site_vel_w = site_vel_w[:, 0]
  return site_vel_w.unsqueeze(-1)


def goal_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Goal position in world coordinates from command manager. Returns [num_envs, 2]."""
  pos: torch.Tensor = env.command_manager.get_command("goal")
  assert pos is not None, "Command 'goal' not found in command manager"
  return pos


def goal_pos_local(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Goal position in environment-local (maze-centered) coordinates. Returns [num_envs, 2]."""
  return goal_pos(env) - env.scene.env_origins[:, :2]


def agent_to_goal_vector(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Vector from agent to goal (goal - agent) in world frame. Returns [num_envs, 2]."""
  return goal_pos(env) - agent_pos(env, asset_cfg)
