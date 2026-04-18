from __future__ import annotations

import heapq
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
from colosseum.utils.grid_frame import GridFrame
from colosseum.tasks.maze.mdp.observations import agent_pos_local

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer

  from colosseum.envs.abstraction_based_env import AbstractionBasedEnv


# ── Settings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class GridAbstractionSettings(AbstractionSettings):
  """One unique abstract configuration: a (goal_cell, map) pair.

  Hash and equality are based on ``goal_cell`` and the *identity* of the ``map``
  tensor.  The term always passes the same ``_map_snapshot`` object to settings
  created within one map lifetime, so object-identity equality is a correct
  proxy for content equality without computing an expensive tensor hash.

  When the environment map changes, the term creates a new snapshot object;
  all subsequently created settings objects get a different ``id(map)`` and
  therefore hash to different registry slots — no stale entries are reused.

  Attributes:
      goal_cell: Discrete (row, col) goal index in the grid.
      map:       [rows, cols] bool tensor — True = wall.  Always the shared
                 snapshot object, never a fresh copy.
  """

  goal_cell: tuple[int, int]
  map: torch.Tensor  # [rows, cols]

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, GridAbstractionSettings):
      return NotImplemented
    return self.goal_cell == other.goal_cell and self.map is other.map

  def __hash__(self) -> int:
    return hash((self.goal_cell, id(self.map)))


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class GridAbstractionTermCfg(AbstractionTermCfg):
  """Configuration for the grid abstraction term.

  Attributes:
      grid_frame:         Grid geometry (cell size, origin, dimensions).
      map:                [rows, cols] bool tensor — True = wall / blocked.
                          Grid dimensions are implicit in map.shape.
                          If None, assumes open space.
      direction_method:   "harmonic" (Laplace) or "gradient" (discrete ∇V*).
      wall_center_weight: Weight that biases paths toward cell centres.
      debug_vis:          Enable per-step debug visualisation.
  """

  grid_frame: GridFrame
  map: Optional[torch.Tensor] = None
  direction_method: Literal["harmonic", "gradient"] = "harmonic"
  wall_center_weight: float = 2.0
  debug_vis: bool = True

  def build(self, env: AbstractionBasedEnv) -> GridAbstraction:
    return GridAbstraction(cfg=self, env=env)


# ── Term ──────────────────────────────────────────────────────────────────────


class GridAbstraction(AbstractionTerm):
  """Grid-based abstraction that supervises one Dijkstra problem per unique setting.

  The term acts as a registry supervisor.  A ``GridAbstractionSettings`` object
  represents one unique (goal_cell, map) pair.  Its identity-based hash makes it
  a valid dict key: the same ``_map_snapshot`` object is reused across all
  settings created within one map lifetime, so two settings objects with the
  same goal_cell and the same snapshot hash to the same slot.

  Registries on the term:
    ``_cost_map_registry``: GridAbstractionSettings → [rows, cols] Dijkstra cost map.
    ``_dir_map_registry``:  GridAbstractionSettings → [rows, cols, 2] direction map.

  On each reset, ``_maybe_rebuild`` detects which settings are new (not yet in the
  registry), runs Dijkstra once per new setting (O(V log V), CPU Python), computes
  direction maps for new settings in a batch, and assembles the per-env
  ``cost_map [N, rows, cols]`` and ``direction_map [N, rows, cols, 2]`` tensors.

  Query methods: direction(), get_next_best_cell_indices(), distance_to_next_best_cell()
  """

  cfg: GridAbstractionTermCfg

  OBSTACLE_COST: float = 1e6
  GOAL_COST: float = 0.0
  CARDINAL_COST: float = 1.0

  def __init__(self, cfg: GridAbstractionTermCfg, env: AbstractionBasedEnv):
    super().__init__(cfg, env)

    self.grid_frame = cfg.grid_frame
    self.map = (
      cfg.map.to(device=self.device, dtype=torch.bool)
      if cfg.map is not None
      else torch.zeros(
        (cfg.grid_frame.num_rows, cfg.grid_frame.num_cols),
        dtype=torch.bool,
        device=self.device,
      )
    )

    # ── Registries (keyed by GridAbstractionSettings) ─────────────────────────
    self._cost_map_registry: dict[GridAbstractionSettings, torch.Tensor] = {}
    self._dir_map_registry: dict[GridAbstractionSettings, torch.Tensor] = {}

    # Stable map snapshot shared by all settings within one map lifetime.
    # Replaced (new object) when self.map changes, invalidating old registry keys.
    self._map_snapshot: Optional[torch.Tensor] = None

    # ── Per-env current state ─────────────────────────────────────────────────
    self._env_goal_cells: Optional[torch.Tensor] = None   # [N, 2] discrete
    self._env_goals_local: Optional[torch.Tensor] = None  # [N, 2] continuous

    # ── Assembled per-env outputs ─────────────────────────────────────────────
    self.cost_map: Optional[torch.Tensor] = None       # [N, rows, cols]
    self.direction_map: Optional[torch.Tensor] = None  # [N, rows, cols, 2]

    # First registered setting — kept as a "is-initialised" sentinel for
    # subclasses that relied on the old ``self.settings is not None`` check.
    # Prefer checking ``self._env_goal_cells is not None`` in new code.
    self.settings: Optional[GridAbstractionSettings] = None

    self.wall_distance_map = self._compute_wall_distance_map()

  # ── Lifecycle ───────────────────────────────────────────────────────────────

  def _update_settings(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Goals are read from all envs (goal command is global).
    try:
      goal_world = self._env.command_manager.get_command("goal")
      assert isinstance(goal_world, torch.Tensor)
      goal_local = goal_world[:, :2] - self._env.scene.env_origins[:, :2]

      self._env_goals_local = goal_local
      self._env_goal_cells = self._local_to_grid(goal_local)  # [N, 2]

      # Replace snapshot (new object) when the environment map changes.
      # Old registry keys referenced the previous snapshot id — they are now
      # unreachable and will be garbage-collected after the clear below.
      if self._map_snapshot is None or not torch.equal(self.map, self._map_snapshot):
        self._map_snapshot = self.map.clone()
        self._cost_map_registry.clear()
        self._dir_map_registry.clear()
        self.settings = None
        logger.info("GridAbstraction: map changed — registries cleared")

      # Update compat sentinel.
      if self.settings is None:
        key = (int(self._env_goal_cells[0, 0].item()), int(self._env_goal_cells[0, 1].item()))
        self.settings = GridAbstractionSettings(goal_cell=key, map=self._map_snapshot)

    except Exception as e:
      logger.warning(f"Could not update goals from command manager: {e}")

  def _maybe_rebuild(self, env_ids: torch.Tensor) -> None:
    if self._env_goal_cells is None or self._map_snapshot is None:
      return

    unique_cells = torch.unique(self._env_goal_cells, dim=0)  # [K', 2]

    # Build settings objects for all active unique goal cells.
    # Same (goal_cell, map_snapshot) always yields the same hash → same registry slot.
    active_settings = [
      GridAbstractionSettings(
        goal_cell=(int(unique_cells[k, 0].item()), int(unique_cells[k, 1].item())),
        map=self._map_snapshot,
      )
      for k in range(unique_cells.shape[0])
    ]

    # Solve Dijkstra for any setting not yet in the registry.
    new_cost = [s for s in active_settings if s not in self._cost_map_registry]
    if new_cost:
      logger.info(f"GridAbstraction: running Dijkstra for {len(new_cost)} new setting(s)")
      for s in new_cost:
        self._cost_map_registry[s] = self._dijkstra_single(*s.goal_cell)

    # Compute direction maps for settings that are missing one.
    new_dir = [s for s in active_settings if s not in self._dir_map_registry]
    if new_dir:
      cost_batch = torch.stack(
        [self._cost_map_registry[s] for s in new_dir]
      )  # [M, rows, cols]
      goal_batch = torch.tensor(
        [s.goal_cell for s in new_dir], device=self.device, dtype=torch.long
      )  # [M, 2]
      dir_batch = self._compute_direction_map(cost_batch, goal_batch)
      if dir_batch is not None:
        for i, s in enumerate(new_dir):
          self._dir_map_registry[s] = dir_batch[i]  # [rows, cols, 2]

    if new_cost or new_dir or self.cost_map is None:
      self._assemble_outputs(active_settings)

  def _assemble_outputs(
    self, active_settings: list[GridAbstractionSettings]
  ) -> None:
    """Build [N, rows, cols] tensors by broadcasting from per-setting registries."""
    if self._env_goal_cells is None or self._map_snapshot is None:
      return

    # Map each unique goal cell to a slot index in active_settings.
    cell_to_slot: dict[tuple[int, int], int] = {s.goal_cell: i for i, s in enumerate(active_settings)}

    rows_t = self._env_goal_cells[:, 0]
    cols_t = self._env_goal_cells[:, 1]

    env_slot = torch.tensor(
      [
        cell_to_slot[(int(rows_t[n].item()), int(cols_t[n].item()))]
        for n in range(self.num_envs)
      ],
      device=self.device,
      dtype=torch.long,
    )  # [N]

    unique_cost_stack = torch.stack(
      [self._cost_map_registry[s] for s in active_settings]
    )  # [K, rows, cols]
    self.cost_map = unique_cost_stack[env_slot]  # [N, rows, cols]

    if all(s in self._dir_map_registry for s in active_settings):
      unique_dir_stack = torch.stack(
        [self._dir_map_registry[s] for s in active_settings]
      )  # [K, rows, cols, 2]
      self.direction_map = unique_dir_stack[env_slot]  # [N, rows, cols, 2]

  def _update_signals(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Cost/direction maps are precomputed in _maybe_rebuild.

  # ── Core Computation ────────────────────────────────────────────────────────

  def _dijkstra_single(self, goal_i: int, goal_j: int) -> torch.Tensor:
    """Run Dijkstra from one goal cell (O(V log V), CPU Python, no GPU syncs).

    Args:
        goal_i: Row index of the goal cell.
        goal_j: Column index of the goal cell.

    Returns:
        [rows, cols] float32 cost tensor on ``self.device``.
        Obstacle cells and unreachable free cells receive OBSTACLE_COST.
    """
    rows, cols = self.grid_frame.num_rows, self.grid_frame.num_cols
    obs = self.map.cpu().tolist()                   # list[list[bool]]
    wall_d = self.wall_distance_map.cpu().tolist()  # list[list[float]]

    INF = float("inf")
    dist = [[INF] * cols for _ in range(rows)]

    if not (0 <= goal_i < rows and 0 <= goal_j < cols) or obs[goal_i][goal_j]:
      return torch.full(
        (rows, cols), self.OBSTACLE_COST, dtype=torch.float32, device=self.device
      )

    dist[goal_i][goal_j] = self.GOAL_COST
    heap: list[tuple[float, int, int]] = [(self.GOAL_COST, goal_i, goal_j)]

    while heap:
      cost, ci, cj = heapq.heappop(heap)
      if cost > dist[ci][cj]:
        continue  # Stale heap entry.
      for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, nj = ci + di, cj + dj
        if not (0 <= ni < rows and 0 <= nj < cols) or obs[ni][nj]:
          continue
        wall_penalty = self.cfg.wall_center_weight / (wall_d[ni][nj] + 1.0)
        new_cost = cost + self.CARDINAL_COST + wall_penalty
        if new_cost < dist[ni][nj]:
          dist[ni][nj] = new_cost
          heapq.heappush(heap, (new_cost, ni, nj))

    result = torch.tensor(dist, dtype=torch.float32)
    result[self.map.cpu()] = self.OBSTACLE_COST
    result[result == INF] = self.OBSTACLE_COST
    return result.to(device=self.device)

  def _compute_wall_distance_map(self) -> torch.Tensor:
    """Multi-source BFS from all wall cells — computed once at init.

    Returns:
        [rows, cols] float tensor on self.device.
        Wall cells have distance 0; free cells have hop-count to nearest wall.
    """
    rows, cols = self.grid_frame.num_rows, self.grid_frame.num_cols
    dist = torch.full((rows, cols), float("inf"))
    map_cpu = self.map.cpu()
    dist[map_cpu] = 0.0

    queue: deque[tuple[int, int]] = deque()
    for idx in map_cpu.nonzero():
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

  # ── Direction Map ───────────────────────────────────────────────────────────

  def _compute_direction_map(
    self,
    cost_maps: torch.Tensor,         # [K, rows, cols]
    unique_goal_cells: torch.Tensor,  # [K, 2]
  ) -> Optional[torch.Tensor]:       # [K, rows, cols, 2] or None
    """Compute direction field for K unique cost maps.

    Override point: return None to opt out (e.g. SokobanGridAbstraction).
    The [K, rows, cols, 2] result is split and stored per-setting in
    ``_dir_map_registry``, then broadcast to [N, rows, cols, 2] in
    ``_assemble_outputs``.
    """
    if self.cfg.direction_method == "gradient":
      return self._compute_direction_map_gradient(cost_maps)
    elif self.cfg.direction_method == "harmonic":
      return self._compute_direction_map_harmonic(cost_maps, unique_goal_cells)
    else:
      raise ValueError(f"Unknown direction_method: {self.cfg.direction_method}")

  def _compute_direction_map_gradient(
    self,
    cost_map: torch.Tensor,  # [K, rows, cols]
  ) -> torch.Tensor:         # [K, rows, cols, 2]
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
    )  # [K, rows, cols, 2]

    direction_map[:, self.map] = 0.0
    direction_map[cost_map < 1e-6] = 0.0
    return direction_map

  def _compute_direction_map_harmonic(
    self,
    cost_map: torch.Tensor,          # [K, rows, cols] — unused, kept for signature symmetry
    unique_goal_cells: torch.Tensor, # [K, 2]
  ) -> torch.Tensor:                 # [K, rows, cols, 2]
    phi = self._solve_harmonic_potential(unique_goal_cells)  # [K, rows, cols]

    phi_padded = torch.nn.functional.pad(phi, (1, 1, 1, 1), mode="replicate")
    grad_i = (phi_padded[:, 2:, 1:-1] - phi_padded[:, :-2, 1:-1]) / 2
    grad_j = (phi_padded[:, 1:-1, 2:] - phi_padded[:, 1:-1, :-2]) / 2

    direction_map = torch.stack([-grad_i, -grad_j], dim=-1)  # [K, rows, cols, 2]
    norms = torch.linalg.norm(direction_map, dim=-1, keepdim=True).clamp(min=1e-6)
    direction_map = direction_map / norms
    direction_map[:, self.map] = 0.0
    return direction_map

  def _solve_harmonic_potential(
    self,
    unique_goal_cells: torch.Tensor,  # [K, 2]
  ) -> torch.Tensor:                  # [K, rows, cols]
    K = unique_goal_cells.shape[0]
    rows, cols = self.grid_frame.num_rows, self.grid_frame.num_cols

    goal_mask = torch.zeros((K, rows, cols), dtype=torch.bool, device=self.device)
    for k in range(K):
      gi = int(unique_goal_cells[k, 0].item())
      gj = int(unique_goal_cells[k, 1].item())
      if 0 <= gi < rows and 0 <= gj < cols:
        goal_mask[k, gi, gj] = True

    phi = torch.full((K, rows, cols), 0.5, device=self.device)
    phi[:, self.map] = 1.0
    phi[goal_mask] = 0.0

    updatable = (~self.map).unsqueeze(0) & ~goal_mask  # [K, rows, cols]
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

  def _require_direction_map(self) -> torch.Tensor:
    """Return the direction map, triggering the first build if needed.

    Raises RuntimeError if this abstraction opts out of direction-field navigation.
    """
    if self._env_goal_cells is None:
      self._ensure_built()
    if self.direction_map is None:
      raise RuntimeError(
        f"{type(self).__name__} does not compute a direction map. "
        "Use plan-based query methods instead of direction() / get_next_best_cell_indices()."
      )
    return self.direction_map

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
    direction_map = self._require_direction_map()
    env_range = torch.arange(self.num_envs, device=self.device)
    indices = self._local_to_grid(positions)
    dirs_grid = direction_map[env_range, indices[:, 0], indices[:, 1]]
    return torch.stack([dirs_grid[:, 1], -dirs_grid[:, 0]], dim=-1)

  def get_next_best_cell_indices(self, positions: torch.Tensor) -> torch.Tensor:
    """Get grid indices of the next best cell (where direction field points).

    Args:
        positions: [num_envs, 2] local (x, y)

    Returns:
        [num_envs, 2] grid indices (i, j)
    """
    direction_map = self._require_direction_map()
    env_range = torch.arange(self.num_envs, device=self.device)
    current_indices = self._local_to_grid(positions)
    directions_grid = direction_map[env_range, current_indices[:, 0], current_indices[:, 1]]
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
    self._require_direction_map()
    next_indices = self.get_next_best_cell_indices(positions)
    next_centers = self._grid_to_local(next_indices, center=True)
    return torch.norm(positions - next_centers, dim=-1) / self.grid_frame.cell_size

  # ── Visualization ───────────────────────────────────────────────────────────

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    if self.cost_map is None or self._env_goal_cells is None:
      return
    show_directions = self.direction_map is not None

    env_idx = 0
    cell_size = self.grid_frame.cell_size
    rows = self.grid_frame.num_rows
    cols = self.grid_frame.num_cols

    cost = self.cost_map[env_idx]
    free_mask = cost < self.OBSTACLE_COST / 2
    max_cost = cost[free_mask].max().item() if free_mask.any() else 1.0

    try:
      local_pos = agent_pos_local(
        self._env, SceneEntityCfg("robot", site_names=("root_site",))
      )
      self.get_next_best_cell_indices(local_pos)
    except Exception:
      pass

    step = max(1, int(np.sqrt(rows * cols / 100)))
    goal_cell = self._env_goal_cells[env_idx]

    for i in range(0, rows, step):
      for j in range(0, cols, step):
        if i == int(goal_cell[0].item()) and j == int(goal_cell[1].item()):
          continue

        indices = torch.tensor([[i, j]], device=self.device, dtype=torch.long)
        pos = self._grid_to_local(indices, center=True)[0].cpu().numpy()
        center = np.array([pos[0], pos[1], 0.01])
        normal = np.array([0.0, 0.0, 1.0])

        cost_val = self.cost_map[env_idx, i, j].item()
        is_obstacle = cost_val >= self.OBSTACLE_COST / 2

        if is_obstacle:
          visualizer.add_rectangle(
            center=center, width=cell_size * 0.9, height=cell_size * 0.9,
            normal=normal, color=(0.3, 0.3, 0.3, 0.8),
          )
          if show_directions:
            direction_grid = self.direction_map[env_idx, i, j]
            dx, dy = direction_grid[1].item(), -direction_grid[0].item()
            if abs(dx) + abs(dy) > 1e-3:
              visualizer.add_arrow(
                start=np.array([pos[0], pos[1], 0.05]),
                end=np.array(
                  [pos[0] + dx * cell_size * 0.4, pos[1] + dy * cell_size * 0.4, 0.05]
                ),
                color=(1.0, 0.0, 1.0, 1.0),
              )
        else:
          value = float(np.clip(1.0 - cost_val / max_cost, 0.0, 1.0)) if max_cost > 0 else 0.0
          visualizer.add_rectangle(
            center=center, width=cell_size * 0.9, height=cell_size * 0.9,
            normal=normal, color=(1.0 - value, value, 0.0, 0.6),
          )
          if show_directions:
            direction_grid = self.direction_map[env_idx, i, j]
            dx, dy = direction_grid[1].item(), -direction_grid[0].item()
            if abs(dx) + abs(dy) > 1e-3:
              visualizer.add_arrow(
                start=np.array([pos[0], pos[1], 0.05]),
                end=np.array(
                  [pos[0] + dx * cell_size * 0.5, pos[1] + dy * cell_size * 0.5, 0.05]
                ),
                width=0.1, color=(0.0, 0.5, 1.0, 0.9),
              )

    if self._env_goals_local is not None:
      goal_pos = self._env_goals_local[env_idx].cpu().numpy()
      visualizer.add_cylinder(
        start=np.array([goal_pos[0], goal_pos[1], 0.04]),
        end=np.array([goal_pos[0], goal_pos[1], 0.06]),
        radius=cell_size * 0.5,
        color=(0.0, 0.3, 1.0, 0.9),
      )
