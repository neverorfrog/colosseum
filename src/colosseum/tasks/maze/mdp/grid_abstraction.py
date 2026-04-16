from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import torch
from loguru import logger
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.managers.abstraction_manager import (
  AbstractionSettings,
  AbstractionTerm,
  AbstractionTermCfg,
)
from colosseum.tasks.maze.mdp.grid_frame import GridFrame
from colosseum.tasks.maze.mdp.observations import agent_pos_local

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer

  from colosseum.envs.abstraction_based_env import AbstractionBasedEnv


# ── Settings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class GridAbstractionSettings(AbstractionSettings):
  """Settings that trigger a rebuild when the goal changes.

  Attributes:
      goal: Goal positions per environment [num_envs, 2] in local coordinates.
  """

  goal: torch.Tensor


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class GridAbstractionTermCfg(AbstractionTermCfg):
  """Configuration for the grid abstraction term.

  Attributes:
      grid_frame: Grid geometry (required).
      obstacle_mask: [rows, cols] bool tensor where True = blocked cell.
                     If None, assumes open space (no obstacles).
      direction_method: "harmonic" (Laplace) or "gradient" (discrete ∇V*).
  """

  grid_frame: GridFrame
  obstacle_mask: Optional[torch.Tensor] = None
  direction_method: Literal["harmonic", "gradient"] = "harmonic"
  wall_center_weight: float = 2.0
  debug_vis: bool = True

  def build(self, env: AbstractionBasedEnv) -> GridAbstraction:
    return GridAbstraction(cfg=self, env=env)


# ── Term ──────────────────────────────────────────────────────────────────────


class GridAbstraction(AbstractionTerm):
  """Grid-based abstraction term.

  Uses parallel Dijkstra to compute a cost-to-goal map, then derives a
  direction field from it. Rebuilds whenever the goal changes.

  Query methods: direction(), get_next_best_cell_indices(), distance_to_next_best_cell()
  """

  cfg: GridAbstractionTermCfg

  OBSTACLE_COST: float = 1e6
  GOAL_COST: float = 0.0
  FREE_COST: float = -1.0
  CARDINAL_COST: float = 1.0

  def __init__(self, cfg: GridAbstractionTermCfg, env: AbstractionBasedEnv):
    super().__init__(cfg, env)

    self.grid_frame = cfg.grid_frame
    self.obstacle_mask = (
      cfg.obstacle_mask.to(device=self.device, dtype=torch.bool)
      if cfg.obstacle_mask is not None
      else torch.zeros(
        (cfg.grid_frame.num_rows, cfg.grid_frame.num_cols),
        dtype=torch.bool,
        device=self.device,
      )
    )

    self.cost_map: Optional[torch.Tensor] = None
    self.direction_map: Optional[torch.Tensor] = None

    self.previous_settings: Optional[GridAbstractionSettings] = None
    self.settings: Optional[GridAbstractionSettings] = None

    self.neighbor_offsets = torch.tensor(
      [[-1, 0], [1, 0], [0, -1], [0, 1]], device=self.device, dtype=torch.long
    )
    self.movement_costs = torch.tensor(
      [self.CARDINAL_COST] * 4, device=self.device, dtype=torch.float32
    )

    self.wall_distance_map = self._compute_wall_distance_map()

  # ── Lifecycle ───────────────────────────────────────────────────────────────

  def _update_settings(self, env_ids: torch.Tensor) -> None:
    del env_ids
    if self.settings is not None:
      self.previous_settings = self.settings
    try:
      goal_world = self._env.command_manager.get_command("goal")
      assert isinstance(goal_world, torch.Tensor)
      goal_local = goal_world[:, :2] - self._env.scene.env_origins[:, :2]
      if self.settings is None or not torch.allclose(goal_local, self.settings.goal):
        self.settings = GridAbstractionSettings(goal=goal_local)
    except Exception as e:
      logger.warning(f"Could not update goals from command manager: {e}")

  def _maybe_rebuild(self, env_ids: torch.Tensor) -> None:
    needs_build = self.cost_map is None or self.direction_map is None
    if needs_build or self.previous_settings != self.settings:
      self._build_abstraction()

  def _build_abstraction(self) -> None:
    assert self.settings is not None
    if self.settings.goal.shape[0] != self.num_envs:
      raise ValueError(
        f"Goal shape mismatch: expected [{self.num_envs}, 2], got {self.settings.goal.shape}"
      )
    self.cost_map = self._compute_costs_parallel()
    self.direction_map = self._compute_direction_map(self.cost_map)

  def _update_signals(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Direction map is precomputed during _build_abstraction

  # ── Core Computation ────────────────────────────────────────────────────────

  def _compute_costs_parallel(self) -> torch.Tensor:
    assert self.settings is not None
    rows = self.grid_frame.num_rows
    cols = self.grid_frame.num_cols

    cost_map = torch.full(
      (self.num_envs, rows, cols),
      self.FREE_COST,
      dtype=torch.float32,
      device=self.device,
    )
    cost_map[:, self.obstacle_mask] = self.OBSTACLE_COST

    goal_indices = self._local_to_grid(self.settings.goal)
    for env_idx in range(self.num_envs):
      i, j = int(goal_indices[env_idx, 0].item()), int(goal_indices[env_idx, 1].item())
      if 0 <= i < rows and 0 <= j < cols:
        cost_map[env_idx, i, j] = self.GOAL_COST

    return self._dijkstra_parallel(cost_map)

  def _compute_wall_distance_map(self) -> torch.Tensor:
    """Multi-source BFS from all wall cells to compute nearest-wall distance.

    Runs on CPU once at build time (obstacle mask is env-independent).

    Returns:
        [rows, cols] float tensor on self.device.
        Obstacle cells have distance 0; free cells have hop-count to nearest wall.
    """
    rows, cols = self.grid_frame.num_rows, self.grid_frame.num_cols
    dist = torch.full((rows, cols), float("inf"))
    obstacle_cpu = self.obstacle_mask.cpu()
    dist[obstacle_cpu] = 0.0

    queue: deque[tuple[int, int]] = deque()
    for idx in obstacle_cpu.nonzero():
      queue.append((idx[0].item(), idx[1].item()))

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
      ci, cj = queue.popleft()
      cd = dist[ci, cj].item()
      for di, dj in offsets:
        ni, nj = ci + di, cj + dj
        if 0 <= ni < rows and 0 <= nj < cols and torch.isinf(dist[ni, nj]):
          dist[ni, nj] = cd + 1
          queue.append((ni, nj))

    return dist.to(device=self.device)

  def _dijkstra_parallel(self, cost_map: torch.Tensor) -> torch.Tensor:
    rows = self.grid_frame.num_rows
    cols = self.grid_frame.num_cols

    cost_map = torch.where(
      cost_map == self.FREE_COST,
      torch.tensor(float("inf"), device=self.device),
      cost_map,
    )

    visited = torch.zeros(
      (self.num_envs, rows, cols), dtype=torch.bool, device=self.device
    )
    is_obstacle = cost_map == self.OBSTACLE_COST
    env_range = torch.arange(self.num_envs, device=self.device)

    for _ in range(rows * cols):
      temp_costs = torch.where(visited, float("inf"), cost_map)
      flat_idx = torch.argmin(temp_costs.view(self.num_envs, -1), dim=1)
      current_i = flat_idx // cols
      current_j = flat_idx % cols
      current_costs = cost_map[env_range, current_i, current_j]

      if torch.all(torch.isinf(current_costs)):
        break

      visited[env_range, current_i, current_j] = True

      for dir_idx, (di, dj) in enumerate(self.neighbor_offsets):
        base_cost = self.movement_costs[dir_idx].item()
        ni = current_i + di
        nj = current_j + dj
        in_bounds = (ni >= 0) & (ni < rows) & (nj >= 0) & (nj < cols)

        for env_idx in range(self.num_envs):
          if not in_bounds[env_idx] or torch.isinf(current_costs[env_idx]):
            continue
          ni_val, nj_val = int(ni[env_idx].item()), int(nj[env_idx].item())
          if is_obstacle[env_idx, ni_val, nj_val] or visited[env_idx, ni_val, nj_val]:
            continue
          wall_dist = self.wall_distance_map[ni_val, nj_val].item()
          wall_penalty = self.cfg.wall_center_weight / (wall_dist + 1.0)
          move_cost = base_cost + wall_penalty
          new_cost = current_costs[env_idx] + move_cost
          if new_cost < cost_map[env_idx, ni_val, nj_val]:
            cost_map[env_idx, ni_val, nj_val] = new_cost

    return torch.where(torch.isinf(cost_map), self.OBSTACLE_COST, cost_map)

  def _compute_direction_map(self, cost_map: torch.Tensor) -> torch.Tensor:
    if self.cfg.direction_method == "gradient":
      return self._compute_direction_map_gradient(cost_map)
    elif self.cfg.direction_method == "harmonic":
      return self._compute_direction_map_harmonic(cost_map)
    else:
      raise ValueError(f"Unknown direction_method: {self.cfg.direction_method}")

  def _compute_direction_map_gradient(self, cost_map: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(
      cost_map, (1, 1, 1, 1), mode="constant", value=float(self.OBSTACLE_COST)
    )
    neighbor_costs = torch.stack(
      [
        padded[:, :-2, 1:-1],
        padded[:, 2:, 1:-1],
        padded[:, 1:-1, :-2],
        padded[:, 1:-1, 2:],
      ],
      dim=1,
    )
    best_dir_idx = torch.argmin(neighbor_costs, dim=1)

    di_offsets = torch.tensor([-1.0, 1.0, 0.0, 0.0], device=self.device)
    dj_offsets = torch.tensor([0.0, 0.0, -1.0, 1.0], device=self.device)
    direction_map = torch.stack(
      [di_offsets[best_dir_idx], dj_offsets[best_dir_idx]], dim=-1
    )

    direction_map[:, self.obstacle_mask] = 0.0
    direction_map[cost_map < 1e-6] = 0.0
    return direction_map

  def _compute_direction_map_harmonic(self, cost_map: torch.Tensor) -> torch.Tensor:
    phi = self._solve_harmonic_potential()

    phi_padded = torch.nn.functional.pad(phi, (1, 1, 1, 1), mode="replicate")
    grad_i = (phi_padded[:, 2:, 1:-1] - phi_padded[:, :-2, 1:-1]) / 2
    grad_j = (phi_padded[:, 1:-1, 2:] - phi_padded[:, 1:-1, :-2]) / 2

    direction_map = torch.stack([-grad_i, -grad_j], dim=-1)
    norms = torch.linalg.norm(direction_map, dim=-1, keepdim=True).clamp(min=1e-6)
    direction_map = direction_map / norms
    direction_map[:, self.obstacle_mask] = 0.0
    return direction_map

  def _solve_harmonic_potential(self) -> torch.Tensor:
    assert self.settings is not None
    rows, cols = self.grid_frame.num_rows, self.grid_frame.num_cols
    goal_indices = self._local_to_grid(self.settings.goal)

    goal_mask = torch.zeros(
      (self.num_envs, rows, cols), dtype=torch.bool, device=self.device
    )
    for env_idx in range(self.num_envs):
      gi = int(goal_indices[env_idx, 0].item())
      gj = int(goal_indices[env_idx, 1].item())
      if 0 <= gi < rows and 0 <= gj < cols:
        goal_mask[env_idx, gi, gj] = True

    phi = torch.full((self.num_envs, rows, cols), 0.5, device=self.device)
    phi[:, self.obstacle_mask] = 1.0
    phi[goal_mask] = 0.0

    updatable = (~self.obstacle_mask).unsqueeze(0) & ~goal_mask
    max_iter = max(rows, cols) ** 2
    tol = 1e-5

    for iteration in range(max_iter):
      phi_old = phi.clone()
      phi_padded = torch.nn.functional.pad(phi, (1, 1, 1, 1), mode="replicate")
      neighbor_avg = (
        phi_padded[:, :-2, 1:-1]
        + phi_padded[:, 2:, 1:-1]
        + phi_padded[:, 1:-1, :-2]
        + phi_padded[:, 1:-1, 2:]
      ) / 4.0
      phi = torch.where(updatable, neighbor_avg, phi)
      diff = torch.abs(phi - phi_old).max().item()
      if diff < tol:
        logger.info(
          f"Harmonic potential converged in {iteration + 1} iterations (diff={diff:.2e})"
        )
        break

    return phi

  # ── Coordinate Helpers ──────────────────────────────────────────────────────

  def _local_to_grid(self, positions: torch.Tensor) -> torch.Tensor:
    return self.grid_frame.local_to_grid_batch(positions)

  def _grid_to_local(self, indices: torch.Tensor, center: bool = True) -> torch.Tensor:
    return self.grid_frame.grid_to_local_batch(indices, center=center)

  def _ensure_built(self) -> None:
    env_ids = torch.arange(self.num_envs, device=self.device)
    self._update_settings(env_ids)
    self._maybe_rebuild(env_ids)

  # ── Query Interface ─────────────────────────────────────────────────────────

  def direction(self, positions: torch.Tensor) -> torch.Tensor:
    """Get direction at local positions.

    Grid space: i increases downward, j increases rightward.
    Local space: x increases rightward, y increases upward.
    Conversion: dx = dj, dy = -di

    Args:
        positions: [num_envs, 2] local (x, y)

    Returns:
        [num_envs, 2] unit vectors in local space (dx, dy)
    """
    if self.direction_map is None:
      self._ensure_built()
    assert self.direction_map is not None
    env_range = torch.arange(self.num_envs, device=self.device)
    indices = self._local_to_grid(positions)
    dirs_grid = self.direction_map[env_range, indices[:, 0], indices[:, 1]]
    return torch.stack([dirs_grid[:, 1], -dirs_grid[:, 0]], dim=-1)

  def get_next_best_cell_indices(self, positions: torch.Tensor) -> torch.Tensor:
    """Get grid indices of the next best cell (where direction field points).

    Args:
        positions: [num_envs, 2] local (x, y)

    Returns:
        [num_envs, 2] grid indices (i, j)
    """
    if self.direction_map is None:
      self._ensure_built()
    assert self.direction_map is not None
    env_range = torch.arange(self.num_envs, device=self.device)
    current_indices = self._local_to_grid(positions)
    directions_grid = self.direction_map[
      env_range, current_indices[:, 0], current_indices[:, 1]
    ]
    next_indices = current_indices + torch.round(directions_grid).long()
    next_indices[:, 0] = next_indices[:, 0].clamp(0, self.grid_frame.num_rows - 1)
    next_indices[:, 1] = next_indices[:, 1].clamp(0, self.grid_frame.num_cols - 1)
    return next_indices

  def distance_to_next_best_cell(self, positions: torch.Tensor) -> torch.Tensor:
    """Normalized distance from current position to next best cell center.

    Args:
        positions: [num_envs, 2] local (x, y)

    Returns:
        [num_envs] distances normalized by cell_size
    """
    if self.direction_map is None:
      self._ensure_built()
    next_indices = self.get_next_best_cell_indices(positions)
    next_centers = self._grid_to_local(next_indices, center=True)
    return torch.norm(positions - next_centers, dim=-1) / self.grid_frame.cell_size

  # ── Visualization ───────────────────────────────────────────────────────────

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    if self.cost_map is None or self.direction_map is None:
      return

    env_idx = 0
    cell_size = self.grid_frame.cell_size
    rows = self.grid_frame.num_rows
    cols = self.grid_frame.num_cols

    # Normalize cost → value: lower cost = closer to goal = green
    cost = self.cost_map[env_idx]
    free_mask = cost < self.OBSTACLE_COST / 2
    max_cost = cost[free_mask].max().item() if free_mask.any() else 1.0

    # Robot position and next-best-cell for highlights
    robot_pos_local = None
    next_best_cells = None
    try:
      local_pos = agent_pos_local(
        self._env, SceneEntityCfg("robot", site_names=("root_site",))
      )
      robot_pos_local = local_pos
      next_best_cells = self.get_next_best_cell_indices(local_pos)
    except Exception:
      pass

    step = max(1, int(np.sqrt(rows * cols / 100)))

    goal_cell_idx = None
    if self.settings is not None:
      goal_cell_idx = self._local_to_grid(self.settings.goal[env_idx : env_idx + 1])[0]

    for i in range(0, rows, step):
      for j in range(0, cols, step):
        is_goal_cell = (
          goal_cell_idx is not None
          and i == goal_cell_idx[0].item()
          and j == goal_cell_idx[1].item()
        )
        if is_goal_cell:
          continue

        indices = torch.tensor([[i, j]], device=self.device, dtype=torch.long)
        pos = self._grid_to_local(indices, center=True)[0].cpu().numpy()
        center = np.array([pos[0], pos[1], 0.01])
        normal = np.array([0.0, 0.0, 1.0])

        cost_val = self.cost_map[env_idx, i, j].item()
        is_obstacle = cost_val >= self.OBSTACLE_COST / 2

        if is_obstacle:
          visualizer.add_rectangle(
            center=center,
            width=cell_size * 0.9,
            height=cell_size * 0.9,
            normal=normal,
            color=(0.3, 0.3, 0.3, 0.8),
          )
          direction_grid = self.direction_map[env_idx, i, j]
          dx = direction_grid[1].item()
          dy = -direction_grid[0].item()
          if abs(dx) + abs(dy) > 1e-3:
            visualizer.add_arrow(
              start=np.array([pos[0], pos[1], 0.05]),
              end=np.array(
                [pos[0] + dx * cell_size * 0.4, pos[1] + dy * cell_size * 0.4, 0.05]
              ),
              color=(1.0, 0.0, 1.0, 1.0),
            )
        else:
          value = (
            float(np.clip(1.0 - cost_val / max_cost, 0.0, 1.0)) if max_cost > 0 else 0.0
          )
          visualizer.add_rectangle(
            center=center,
            width=cell_size * 0.9,
            height=cell_size * 0.9,
            normal=normal,
            color=(1.0 - value, value, 0.0, 0.6),
          )
          direction_grid = self.direction_map[env_idx, i, j]
          dx = direction_grid[1].item()
          dy = -direction_grid[0].item()
          if abs(dx) + abs(dy) > 1e-3:
            visualizer.add_arrow(
              start=np.array([pos[0], pos[1], 0.05]),
              end=np.array(
                [pos[0] + dx * cell_size * 0.5, pos[1] + dy * cell_size * 0.5, 0.05]
              ),
              width=0.1,
              color=(0.0, 0.5, 1.0, 0.9),
            )

    # Goal as blue cylinder
    if self.settings is not None:
      goal_pos = self.settings.goal[env_idx].cpu().numpy()
      visualizer.add_cylinder(
        start=np.array([goal_pos[0], goal_pos[1], 0.04]),
        end=np.array([goal_pos[0], goal_pos[1], 0.06]),
        radius=cell_size * 0.5,
        color=(0.0, 0.3, 1.0, 0.9),
      )
