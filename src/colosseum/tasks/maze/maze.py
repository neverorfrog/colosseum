from __future__ import annotations

from dataclasses import dataclass, field

import torch

from colosseum.utils.grid_frame import GridFrame


class Maze:
  """Maze structure with coordinate conversion utilities.

  Handles:
  - Coordinate conversions between grid and environment-local frames
  - Extraction of valid reset/goal positions (in environment-local coordinates)
  - Wall/cell queries

  Coordinate frames:
  - Grid: (i, j) - discrete indices into maze_map array
  - Environment-Local: (x_local, y_local) - continuous, maze-centered frame

  Both the terrain importer and the grid abstraction use environment-local
  coordinates, which are offset by env_origins at runtime to get world coords.
  """

  def __init__(self, cfg: MazeCfg) -> None:
    self.cfg = cfg
    self.maze_map = cfg.maze_map
    self.cell_size = cfg.cell_size
    self.wall_height = cfg.wall_height
    self.wall_size_factor = cfg.wall_size_factor

    self.grid_frame = GridFrame(
      num_rows=len(self.maze_map),
      num_cols=len(self.maze_map[0]) if self.maze_map else 0,
      cell_size=self.cell_size,
      center_x=0.0,
      center_y=0.0,
    )

    self.valid_free_positions_local: list[tuple[float, float]] = []  # all non-wall cells
    self.valid_reset_positions_local: list[tuple[float, float]] = []
    self.valid_ball_positions_local: list[tuple[float, float]] = []
    self.valid_goal_positions_local: list[tuple[float, float]] = []
    self._extract_valid_positions()

  @property
  def num_rows(self) -> int:
    return self.grid_frame.num_rows

  @property
  def num_cols(self) -> int:
    return self.grid_frame.num_cols

  def grid_to_local(self, i: int, j: int) -> tuple[float, float]:
    return self.grid_frame.grid_to_local(i, j)

  def local_to_grid(self, x_local: float, y_local: float) -> tuple[int, int]:
    return self.grid_frame.local_to_grid(x_local, y_local)

  def is_wall(self, i: int, j: int) -> bool:
    if not (0 <= i < self.num_rows and 0 <= j < self.num_cols):
      return True
    cell = self.maze_map[i][j]
    return cell in (1, "1", "W", "w")

  def is_valid_cell(self, i: int, j: int) -> bool:
    return not self.is_wall(i, j)

  def build_obstacle_mask(self, resolution_factor: int = 1) -> torch.Tensor:
    """Build a boolean obstacle mask from the maze map.

    Args:
        resolution_factor: k >= 1. Each maze cell becomes k×k abstraction cells,
                           producing a finer grid (same as passing a k× upsampled
                           GridFrame to GridAbstractionTermCfg).

    Returns:
        [num_rows*k, num_cols*k] bool CPU tensor, True = wall/blocked cell.
    """
    base_mask = torch.zeros((self.num_rows, self.num_cols), dtype=torch.bool)
    for i in range(self.num_rows):
      for j in range(self.num_cols):
        if self.is_wall(i, j):
          base_mask[i, j] = True

    if resolution_factor == 1:
      return base_mask

    return base_mask.repeat_interleave(resolution_factor, dim=0).repeat_interleave(
      resolution_factor, dim=1
    )

  def build_upsampled_grid_frame(self, resolution_factor: int) -> GridFrame:
    """Build a finer GridFrame where each maze cell is split into k×k cells.

    Use together with build_obstacle_mask(resolution_factor) to pass both
    to GridAbstractionTermCfg.

    Args:
        resolution_factor: k >= 1. k=2 → each maze cell becomes 2×2 abstraction cells.
    """
    if resolution_factor < 1:
      raise ValueError(f"resolution_factor must be >= 1, got {resolution_factor}")
    return GridFrame(
      num_rows=self.num_rows * resolution_factor,
      num_cols=self.num_cols * resolution_factor,
      cell_size=self.cell_size / resolution_factor,
      center_x=self.grid_frame.center_x,
      center_y=self.grid_frame.center_y,
    )

  def _extract_valid_positions(self) -> None:
    for i, row in enumerate(self.maze_map):
      for j, cell in enumerate(row):
        x_local, y_local = self.grid_to_local(i, j)
        if not self.is_wall(i, j):
          self.valid_free_positions_local.append((x_local, y_local))
        if cell in ("r", "R"):
          self.valid_reset_positions_local.append((x_local, y_local))
        elif cell in ("b", "B"):
          self.valid_ball_positions_local.append((x_local, y_local))
        elif cell in ("g", "G"):
          self.valid_goal_positions_local.append((x_local, y_local))


@dataclass(frozen=True)
class MazeCfg:
  """Configuration for a Maze.

  Attributes:
      maze_map: 2D list defining maze structure.
          - 1, '1', 'W', 'w': Wall
          - 0, '0': Empty cell
          - 'r', 'R': Valid robot reset position
          - 'b', 'B': Valid ball reset position
          - 'g', 'G': Valid goal position
      cell_size: Size of each cell in meters.
      wall_height: Height of walls in meters.
      wall_size_factor: Fraction of cell size occupied by wall geometry.
                        1.0 = walls fill full cell, 0.8 = 20% safety margin.
  """

  maze_map: list[list[str | int]] = field(default_factory=list)
  cell_size: float = 0.1
  wall_height: float = 0.2
  wall_size_factor: float = 0.8
