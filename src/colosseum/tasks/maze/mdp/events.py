"""Event functions for maze navigation tasks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.tasks.maze.terrain import MazeTerrainEntity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def reset_wall_positions(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
) -> None:
  """Set mocap wall positions for the given environments.

  Must be registered as both a startup event (to initialize all envs) and a
  reset event (to restore positions after reset_data reverts mocap_pos to the
  model defaults stored in body_pos).

  Wall blocks are K mocap bodies. For each world, wall position k is:
      mocap_pos[world, k] = env_origins[world] + wall_local_centers[k]

  Args:
      env: Environment instance.
      env_ids: Indices of environments to update. None means all environments.
  """
  terrain = env.scene.terrain
  if not isinstance(terrain, MazeTerrainEntity):
    return

  local_centers = terrain.wall_local_centers  # (K, 3)
  if local_centers.shape[0] == 0:
    return

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  # Broadcast to (N_reset, K, 3)
  wall_world_pos = (
    terrain.env_origins[env_ids].unsqueeze(1) + local_centers.unsqueeze(0)
  )

  # mocap_pos: TorchArray of shape (num_envs, K, 3)
  env.sim.data.mocap_pos[env_ids] = wall_world_pos


def reset_to_valid_maze_position(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  z_offset: float = 0.05,
  yaw_range: tuple[float, float] = (-math.pi, math.pi),
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Reset agent to a random valid maze position.

  Samples from valid reset cells defined in the maze map ('r'/'R' cells).
  Sets position in (x, y) from sampled cell and z from z_offset parameter.

  Args:
      env: Environment instance
      env_ids: Environment indices to reset (None = all environments)
      z_offset: Height above ground to place the agent (meters)
      yaw_range: Range (min, max) in radians for random yaw sampling
      asset_cfg: Scene entity configuration for the agent
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  if not isinstance(terrain, MazeTerrainEntity):
    raise ValueError(
      f"Expected MazeTerrainEntity, got {type(terrain)}. "
      "Make sure you're using a maze task."
    )

  valid_reset_local = terrain.valid_reset_positions_local  # (num_valid_resets, 2)
  num_valid_positions = valid_reset_local.shape[0]

  if num_valid_positions == 0:
    raise ValueError(
      "No valid reset positions found in maze. "
      "Make sure your maze map has cells marked with 'r' or 'R'."
    )

  num_resets = len(env_ids)
  sampled_indices = torch.randint(
    0, num_valid_positions, size=(num_resets,), device=env.device
  )
  sampled_positions_local = valid_reset_local[sampled_indices]  # (num_resets, 2)

  world_xy = sampled_positions_local + env.scene.env_origins[env_ids, :2]

  world_positions = torch.zeros((num_resets, 3), device=env.device)
  world_positions[:, :2] = world_xy
  world_positions[:, 2] = env.scene.env_origins[env_ids, 2] + z_offset

  if not asset.is_fixed_base:
    default_root_state = asset.data.default_root_state[env_ids].clone()
    default_root_state[:, 0:3] = world_positions

    # Random yaw so the robot trains with all heading errors, matching the
    # diversity provided by UniformVelocityCommandCfg in the velocity task.
    yaw = torch.empty(num_resets, device=env.device).uniform_(*yaw_range)
    half_yaw = yaw / 2
    quat = torch.zeros(num_resets, 4, device=env.device)
    quat[:, 0] = torch.cos(half_yaw)  # w
    quat[:, 3] = torch.sin(half_yaw)  # z
    default_root_state[:, 3:7] = quat

    default_root_state[:, 7:13] = 0.0
    asset.write_root_state_to_sim(default_root_state, env_ids=env_ids)
  else:
    raise ValueError(
      f"Cannot reset position for entity '{asset_cfg.name}'. "
      "The entity must be floating-base."
    )
