"""Event functions for soccer-maze task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.tasks.maze.terrain import MazeTerrainEntity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def reset_robot_and_ball_sokoban(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_z_offset: float = 0.665,
  ball_z: float = 0.2,
  yaw_range: tuple[float, float] = (-math.pi, math.pi),
  jitter_fraction: float = 0.3,
) -> None:
  """Reset robot and ball to independent, jittered positions within valid maze cells.

  Sokoban-aware variant of ``reset_robot_and_ball``.  Each entity is placed at
  the centre of an independently sampled valid maze cell, plus a small continuous
  jitter so the RL agent sees diverse continuous states that all share the same
  abstract (cell-level) configuration.

  The jitter keeps both entities strictly within their sampled cell:
      delta ∈ [-jitter_fraction × cell_size/2, +jitter_fraction × cell_size/2]

  Robot and ball are guaranteed to be in different cells (ball index is shifted
  by one if the same cell is drawn).

  Args:
      env:             Environment instance.
      env_ids:         Environments to reset (None = all).
      robot_cfg:       Scene entity config for the robot.
      ball_cfg:        Scene entity config for the ball.
      robot_z_offset:  Root height above ground for the robot (metres).
      ball_z:          Ball height above ground (metres, = ball radius).
      yaw_range:       (min, max) yaw in radians for random robot orientation.
      jitter_fraction: Half-width of the within-cell jitter as a fraction of
                       cell_size/2.  0.3 → jitter in ±0.15·cell_size.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  robot: Entity = env.scene[robot_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]

  terrain = env.scene.terrain
  if not isinstance(terrain, MazeTerrainEntity):
    raise ValueError(f"Expected MazeTerrainEntity, got {type(terrain)}")

  # All free (non-wall) cells — superset of 'r' cells, sufficient for robot and ball.
  valid_local = terrain.valid_free_positions_local  # (num_valid, 2) cell centres
  num_valid = valid_local.shape[0]
  if num_valid < 2:
    raise ValueError(
      "Maze has fewer than 2 free cells — cannot place robot and ball in distinct cells."
    )

  n = len(env_ids)
  half_jitter = jitter_fraction * (terrain.maze.cell_size / 2)

  # ── Robot cell ──────────────────────────────────────────────────────────────
  robot_idx = torch.randint(0, num_valid, (n,), device=env.device)
  robot_local = valid_local[robot_idx]  # (n, 2)

  robot_jitter = torch.empty((n, 2), device=env.device).uniform_(-half_jitter, half_jitter)
  robot_local = robot_local + robot_jitter

  # ── Ball cell (different from robot cell) ───────────────────────────────────
  ball_idx = torch.randint(0, num_valid, (n,), device=env.device)
  same = ball_idx == robot_idx
  ball_idx[same] = (robot_idx[same] + 1) % num_valid
  ball_local = valid_local[ball_idx]  # (n, 2)

  ball_jitter = torch.empty((n, 2), device=env.device).uniform_(-half_jitter, half_jitter)
  ball_local = ball_local + ball_jitter

  # ── World positions ─────────────────────────────────────────────────────────
  origins = env.scene.env_origins[env_ids]
  robot_world_xy = robot_local + origins[:, :2]
  ball_world_xy  = ball_local  + origins[:, :2]

  # ── Robot state ─────────────────────────────────────────────────────────────
  robot_pos = torch.zeros((n, 3), device=env.device)
  robot_pos[:, :2] = robot_world_xy
  robot_pos[:, 2]  = origins[:, 2] + robot_z_offset

  yaw = torch.empty(n, device=env.device).uniform_(*yaw_range)
  half_yaw = yaw / 2
  robot_quat = torch.zeros((n, 4), device=env.device)
  robot_quat[:, 0] = torch.cos(half_yaw)  # w
  robot_quat[:, 3] = torch.sin(half_yaw)  # z

  robot_root_state = robot.data.default_root_state[env_ids].clone()
  robot_root_state[:, :3]  = robot_pos
  robot_root_state[:, 3:7] = robot_quat
  robot_root_state[:, 7:]  = 0.0
  robot.write_root_state_to_sim(robot_root_state, env_ids=env_ids)

  # ── Ball state ───────────────────────────────────────────────────────────────
  ball_pos = torch.zeros((n, 3), device=env.device)
  ball_pos[:, :2] = ball_world_xy
  ball_pos[:, 2]  = ball_z

  ball_quat = torch.zeros((n, 4), device=env.device)
  ball_quat[:, 0] = 1.0  # identity

  ball_root_state = torch.cat(
    [ball_pos, ball_quat, torch.zeros((n, 6), device=env.device)], dim=-1
  )
  ball.write_root_state_to_sim(ball_root_state, env_ids=env_ids)


def reset_robot_and_ball(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_z_offset: float = 0.665,
  ball_x_offset: float = 0.4,
  ball_z: float = 0.2,
  yaw_range: tuple[float, float] = (0.0, 0.0),
) -> None:
  """Reset both robot and ball together from the same sampled maze position.

  Samples a valid maze reset cell, places the robot there, then places the
  ball ``ball_x_offset`` metres ahead in world-X from the same position.
  Both writes use the freshly computed world position so there is no data
  staleness between separate events.

  Args:
      env: Environment instance.
      env_ids: Indices of environments to reset (None = all).
      robot_cfg: Scene entity config for the robot.
      ball_cfg: Scene entity config for the ball.
      robot_z_offset: Root height above ground for the robot (metres).
      ball_x_offset: Forward offset (world X) of the ball from the robot (metres).
      ball_z: Ball height above ground (should equal ball radius).
      yaw_range: (min, max) yaw in radians for the robot spawn orientation.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  robot: Entity = env.scene[robot_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]

  terrain = env.scene.terrain
  if not isinstance(terrain, MazeTerrainEntity):
    raise ValueError(f"Expected MazeTerrainEntity, got {type(terrain)}")

  valid_reset_local = terrain.valid_reset_positions_local  # (num_valid, 2)
  num_valid = valid_reset_local.shape[0]
  if num_valid == 0:
    raise ValueError("No valid reset positions found in maze (need 'r'/'R' cells).")

  n = len(env_ids)
  indices = torch.randint(0, num_valid, (n,), device=env.device)
  pos_local = valid_reset_local[indices]  # (n, 2)
  world_xy = pos_local + env.scene.env_origins[env_ids, :2]

  # --- Robot state ---
  robot_pos = torch.zeros((n, 3), device=env.device)
  robot_pos[:, :2] = world_xy
  robot_pos[:, 2] = env.scene.env_origins[env_ids, 2] + robot_z_offset

  yaw = torch.empty(n, device=env.device).uniform_(*yaw_range)
  half_yaw = yaw / 2
  robot_quat = torch.zeros((n, 4), device=env.device)
  robot_quat[:, 0] = torch.cos(half_yaw)  # w
  robot_quat[:, 3] = torch.sin(half_yaw)  # z

  robot_root_state = robot.data.default_root_state[env_ids].clone()
  robot_root_state[:, :3] = robot_pos
  robot_root_state[:, 3:7] = robot_quat
  robot_root_state[:, 7:] = 0.0
  robot.write_root_state_to_sim(robot_root_state, env_ids=env_ids)

  # --- Ball state (same world_xy + forward offset, independent of cached data) ---
  ball_pos = torch.zeros((n, 3), device=env.device)
  ball_pos[:, :2] = world_xy
  ball_pos[:, 0] += ball_x_offset
  ball_pos[:, 2] = ball_z

  ball_quat = torch.zeros((n, 4), device=env.device)
  ball_quat[:, 0] = 1.0  # identity

  ball_root_state = torch.cat(
    [ball_pos, ball_quat, torch.zeros((n, 6), device=env.device)], dim=-1
  )  # (n, 13)
  ball.write_root_state_to_sim(ball_root_state, env_ids=env_ids)
