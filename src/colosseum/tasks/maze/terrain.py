from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import torch
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
from mjlab.utils import spec_config as spec_cfg

from colosseum.tasks.maze.maze import Maze, MazeCfg


class MazeTerrainEntity(TerrainEntity):
  """Terrain importer that builds maze walls from a Maze instance.

  Maze logic (coordinate conversions, valid positions) lives in Maze (CPU-only,
  init-time only). Valid positions are converted to GPU tensors once and cached
  for use by MDP components during training.

  Wall merging: adjacent wall cells are merged into maximal rectangles to
  minimize the number of MuJoCo bodies (critical for 512+ parallel envs).
  """

  def __init__(self, cfg: MazeTerrainEntityCfg, device: str) -> None:
    self.maze = Maze(cfg.maze_cfg)

    self.valid_reset_positions_local = torch.tensor(
      self.maze.valid_reset_positions_local, dtype=torch.float32, device=device
    )
    self.valid_goal_positions_local = torch.tensor(
      self.maze.valid_goal_positions_local, dtype=torch.float32, device=device
    )

    super().__init__(cfg, device)
    self._create_mazes()

  def import_ground_plane(self, name: str) -> None:
    texture = spec_cfg.TextureCfg(
      name="grid",
      type="2d",
      builtin="checker",
      rgb1=(0.6, 0.6, 0.6),
      rgb2=(0.8, 0.8, 0.8),
      mark="edge",
      markrgb=(0.4, 0.4, 0.4),
      width=100,
      height=100,
    )
    material = spec_cfg.MaterialCfg(
      name="grid",
      texture="grid",
      texrepeat=(10, 10),
      texuniform=True,
      reflectance=0.0,
    )
    texture.edit_spec(self._spec)
    material.edit_spec(self._spec)

    self._spec.worldbody.add_body(name=name).add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      pos=(0.05, 0.05, 0),
      size=(0, 0, 1),
      rgba=(0.99, 0.99, 0.99, 1),
      material="grid",
      conaffinity=1,
      contype=1,
    )

  def _create_mazes(self) -> None:
    """Create maze walls for each parallel environment.

    Merges adjacent wall cells into rectangular blocks to reduce the number
    of MuJoCo bodies (one body per block instead of one per cell).
    """
    assert self.env_origins is not None
    walls_parent = self._spec.worldbody.add_body(name="walls", pos=(0, 0, 0))
    wall_blocks = self._find_wall_blocks()

    half = (self.maze.cell_size / 2) * self.maze.wall_size_factor
    for env_idx in range(self.cfg.num_envs):
      env_origin = self.env_origins[env_idx]

      env_walls_body = walls_parent.add_body(
        name=f"{env_idx}_walls",
        pos=(env_origin[0].item(), env_origin[1].item(), 0),
      )

      for block_idx, (i_range, j_range) in enumerate(wall_blocks):
        i_start, i_end = i_range
        j_start, j_end = j_range

        start_x_l, start_y_l = self.maze.grid_to_local(i_start, j_start)
        end_x_l, _ = self.maze.grid_to_local(i_start, j_end)
        _, end_y_l = self.maze.grid_to_local(i_end, j_start)

        center_x_l = (start_x_l + end_x_l) / 2
        center_y_l = (start_y_l + end_y_l) / 2

        env_walls_body.add_geom(
          name=f"{env_idx}_wall_block_{block_idx}_geom",
          type=mujoco.mjtGeom.mjGEOM_BOX,
          pos=(center_x_l, center_y_l, self.maze.wall_height / 2),
          size=(
            half * (j_end - j_start + 1),
            half * (i_end - i_start + 1),
            self.maze.wall_height / 2,
          ),
          rgba=(0.8, 0.8, 0.8, 1.0),
          conaffinity=1,
          contype=1,
          group=1,
        )

  def _find_wall_blocks(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    visited = [[False] * self.maze.num_cols for _ in range(self.maze.num_rows)]
    blocks = []
    for i in range(self.maze.num_rows):
      for j in range(self.maze.num_cols):
        if self.maze.is_wall(i, j) and not visited[i][j]:
          i_end, j_end = self._grow_wall_rectangle(i, j, visited)
          blocks.append(((i, i_end), (j, j_end)))
    return blocks

  def _grow_wall_rectangle(
    self, i_start: int, j_start: int, visited: list[list[bool]]
  ) -> tuple[int, int]:
    j_end = j_start
    while (
      j_end < self.maze.num_cols
      and self.maze.is_wall(i_start, j_end)
      and not visited[i_start][j_end]
    ):
      j_end += 1
    j_end -= 1

    i_end = i_start
    while True:
      i_end += 1
      if i_end >= self.maze.num_rows:
        i_end -= 1
        break
      if not all(
        self.maze.is_wall(i_end, j) and not visited[i_end][j]
        for j in range(j_start, j_end + 1)
      ):
        i_end -= 1
        break

    for i in range(i_start, i_end + 1):
      for j in range(j_start, j_end + 1):
        visited[i][j] = True

    return i_end, j_end


@dataclass
class MazeTerrainEntityCfg(TerrainEntityCfg):
  """Configuration for MazeTerrainEntity."""

  maze_cfg: MazeCfg = field(default_factory=MazeCfg)
  class_type: type = MazeTerrainEntity
