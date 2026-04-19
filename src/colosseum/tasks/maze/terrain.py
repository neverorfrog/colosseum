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

    self.valid_free_positions_local = torch.tensor(
      self.maze.valid_free_positions_local, dtype=torch.float32, device=device
    )
    self.valid_reset_positions_local = torch.tensor(
      self.maze.valid_reset_positions_local, dtype=torch.float32, device=device
    )
    self.valid_ball_positions_local = torch.tensor(
      self.maze.valid_ball_positions_local, dtype=torch.float32, device=device
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
    """Create maze walls as K mocap bodies shared across all environments.

    Instead of N×K static bodies (N envs × K wall blocks), creates K mocap
    bodies. Wall positions are driven per-world via data.mocap_pos, so each
    world sees its own walls at the correct location while the model only
    contains K geoms. This reduces broadphase cost from O((N·K) log(N·K))
    to O(K log K) per world.

    Call reset_wall_positions (startup + reset events) to populate mocap_pos.
    """
    assert self.env_origins is not None
    wall_blocks = self._find_wall_blocks()

    if not wall_blocks:
      self.wall_local_centers = torch.zeros(0, 3, dtype=torch.float32, device=self._device)
      return

    half = (self.maze.cell_size / 2) * self.maze.wall_size_factor
    local_centers: list[list[float]] = []

    # Default body position uses env 0's origin so the initial visual is correct.
    env0_x = self.env_origins[0, 0].item()
    env0_y = self.env_origins[0, 1].item()

    for block_idx, (i_range, j_range) in enumerate(wall_blocks):
      i_start, i_end = i_range
      j_start, j_end = j_range

      start_x_l, start_y_l = self.maze.grid_to_local(i_start, j_start)
      end_x_l, _ = self.maze.grid_to_local(i_start, j_end)
      _, end_y_l = self.maze.grid_to_local(i_end, j_start)

      cx = (start_x_l + end_x_l) / 2
      cy = (start_y_l + end_y_l) / 2
      cz = self.maze.wall_height / 2

      local_centers.append([cx, cy, cz])

      # One mocap body per block: position is overridden per-world at runtime.
      body = self._spec.worldbody.add_body(
        name=f"wall_block_{block_idx}",
        pos=(env0_x + cx, env0_y + cy, cz),
      )
      body.mocap = True

      body.add_geom(
        name=f"wall_block_{block_idx}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0, 0, 0),
        size=(
          half * (j_end - j_start + 1),
          half * (i_end - i_start + 1),
          cz,
        ),
        rgba=(0.8, 0.8, 0.8, 1.0),
        conaffinity=1,
        contype=1,
        group=1,
      )

    self.wall_local_centers = torch.tensor(
      local_centers, dtype=torch.float32, device=self._device
    )  # (K, 3)

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
