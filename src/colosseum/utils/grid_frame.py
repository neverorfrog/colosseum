from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import torch


@dataclass
class GridFrame:
  """Grid coordinate frame with conversion methods.

  Handles mapping between:
  - Grid indices (i, j): discrete, i=row (0 at top), j=column (0 at left)
  - Local coordinates (x, y): continuous, centered at (center_x, center_y)

  Convention: y increases upward in local frame, i increases downward in grid.

  Coordinate System:
      Local Frame:          Grid Frame:
      y ▲                   (0,0)───► j
        │                     │
        │                     ▼
        └───► x               i

  The grid center is at (center_x, center_y) in local coordinates.
  By default, center is at origin (0, 0).
  """

  num_rows: int
  num_cols: int
  cell_size: float
  center_x: float = 0.0
  center_y: float = 0.0

  @property
  def x_map_center(self) -> float:
    return self.num_cols / 2 * self.cell_size

  @property
  def y_map_center(self) -> float:
    return self.num_rows / 2 * self.cell_size

  def grid_to_local(self, i: int, j: int) -> tuple[float, float]:
    """Convert grid indices to local coordinates (cell center)."""
    x_local = (j + 0.5) * self.cell_size - self.x_map_center + self.center_x
    y_local = self.y_map_center - (i + 0.5) * self.cell_size + self.center_y
    return x_local, y_local

  def local_to_grid(self, x_local: float, y_local: float) -> tuple[int, int]:
    """Convert local coordinates to grid indices (clamped to bounds)."""
    x_rel = x_local - self.center_x
    y_rel = y_local - self.center_y
    j = int(floor((x_rel + self.x_map_center) / self.cell_size))
    i = int(floor((self.y_map_center - y_rel) / self.cell_size))
    i = max(0, min(i, self.num_rows - 1))
    j = max(0, min(j, self.num_cols - 1))
    return i, j

  def grid_to_local_batch(self, indices: torch.Tensor, center: bool = True) -> torch.Tensor:
    """Convert grid indices to local coordinates (batched).

    Args:
        indices: [batch, 2] grid indices (i, j)
        center: If True, return cell center; if False, return corner

    Returns:
        positions: [batch, 2] local positions (x, y)
    """
    offset = 0.5 if center else 0.0
    i = indices[:, 0].float()
    j = indices[:, 1].float()
    x_local = (j + offset) * self.cell_size - self.x_map_center + self.center_x
    y_local = self.y_map_center - (i + offset) * self.cell_size + self.center_y
    return torch.stack([x_local, y_local], dim=1)

  def local_to_grid_batch(self, positions: torch.Tensor) -> torch.Tensor:
    """Convert local coordinates to grid indices (batched).

    Args:
        positions: [batch, 2] local positions (x, y)

    Returns:
        indices: [batch, 2] grid indices (i, j), clamped to valid range
    """
    x_rel = positions[:, 0] - self.center_x
    y_rel = positions[:, 1] - self.center_y
    j = torch.floor((x_rel + self.x_map_center) / self.cell_size).long()
    i = torch.floor((self.y_map_center - y_rel) / self.cell_size).long()
    i = torch.clamp(i, 0, self.num_rows - 1)
    j = torch.clamp(j, 0, self.num_cols - 1)
    return torch.stack([i, j], dim=1)

  def grid_direction_to_local(self, di: float, dj: float) -> tuple[float, float]:
    """Convert a grid-space direction (row/col deltas) to local-frame unit vector.

    Grid space: i increases downward, j increases rightward.
    Local space: x increases rightward, y increases upward.
    Conversion: dx = dj, dy = -di

    Args:
        di: Row delta in grid space (positive = downward in grid).
        dj: Column delta in grid space (positive = rightward in grid).

    Returns:
        (dx, dy) unit vector in local frame. Returns (0, 0) for zero input.
    """
    dx = float(dj)
    dy = float(-di)
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-6:
      return 0.0, 0.0
    return dx / norm, dy / norm

  @classmethod
  def from_maze(cls, maze) -> GridFrame:
    return cls(
      num_rows=maze.num_rows,
      num_cols=maze.num_cols,
      cell_size=maze.cell_size,
      center_x=0.0,
      center_y=0.0,
    )

  @classmethod
  def from_bounds(
    cls,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    cell_size: float,
  ) -> GridFrame:
    """Create GridFrame from workspace bounds."""
    width = x_max - x_min
    height = y_max - y_min
    num_cols = int(ceil(width / cell_size))
    num_rows = int(ceil(height / cell_size))
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return cls(
      num_rows=num_rows,
      num_cols=num_cols,
      cell_size=cell_size,
      center_x=center_x,
      center_y=center_y,
    )
