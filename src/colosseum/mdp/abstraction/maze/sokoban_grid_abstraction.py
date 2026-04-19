"""Sokoban-style planning abstraction for the soccer-maze task.

Inherits Dijkstra cost map from GridAbstraction but opts out of the direction
field entirely (returns None from _compute_direction_map).  Instead it maintains
a per-env discrete plan (MOVE robot / PUSH ball sequence) looked up from a shared
cache keyed on the abstract configuration (robot_cell, ball_cell, goal_cell).

Lifecycle:
  reset(env_ids)     → cost map rebuilt if goal or obstacles changed, then plans
                       initialized for env_ids at their current (post-reset) positions.
  compute(dt)        → _update_signals(all_envs): advance plan step when the
                       current action's goal is reached; detect deviations and replan.

Design notes
------------
Hysteresis
  Raw cell indices flip every time a continuous position crosses a grid line,
  causing rapid oscillation near boundaries.  ``_confirmed_*_cells`` only
  register a transition when the position has penetrated
  ``hysteresis_fraction × cell_size`` past the boundary in the new cell's
  direction.  Transitions of more than one cell (teleport on reset) bypass
  hysteresis and are always accepted.

2-MDP / robot-ball co-occupancy
  Standard Sokoban forbids agent and box sharing a cell.  In soccer the robot
  must physically enter the ball's grid cell to kick it, so co-occupancy is a
  valid abstract state.  We handle it by tracking ``_robot_prev_cells``: when
  the robot's confirmed cell equals the ball's confirmed cell, the abstraction
  knows where the robot came from (approach direction) and continues executing
  the active PUSH action without replanning.  The PUSH completion check is
  unchanged — it simply waits for the ball to leave ``_push_start_ball_cell``.

Deviation detection and online replanning
  MOVE: robot left ``_move_start_robot_cell`` but ended up in a cell that is not
        ``expected_next_robot_cell`` → deviation → replan from current positions.
  PUSH: ball left ``_push_start_ball_cell`` but ended up in a cell that is not
        ``expected_next_ball_cell`` → deviation → replan from current positions.
  Replanning uses the shared cache: if the new (robot, ball, goal) config was
  seen before the plan is loaded instantly; otherwise the solver is called with
  a configurable timeout.  If the solver fails (no plan or timeout) the
  ``deviation_detected`` flag stays set and the termination term ends the episode.

Vectorization
  ``_update_signals`` is fully tensor-parallel (no Python loop over envs in the
  hot path).  ``_load_step_targets`` and plan solving are called only for
  deviating/advancing envs (rare), so their Python loops are negligible.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import torch
from loguru import logger

from colosseum.mdp.abstraction.maze.grid_abstraction import (
  GridAbstraction,
  GridAbstractionTermCfg,
)
from colosseum.mdp.abstraction.maze.sokoban_solver import (
  ENCODING,
  create_diagonal_sokoban_problem,
  create_free_diagonal_sokoban_problem,
  create_sokoban_problem,
  maze_to_string,
  plan_to_simple_steps,
  solve_sokoban_problem,
)

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer

  from colosseum.envs.abstraction_based_env import AbstractionBasedEnv


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class SokobanAction:
  """One step in a Sokoban plan.

  Attributes:
      action_type:       "MOVE" (robot navigates to a cell) or
                         "PUSH" (robot pushes the ball one cell).
      direction_grid:    (di, dj) direction in grid space (row_delta, col_delta).
                         Always a unit vector for both MOVE and PUSH since the
                         UP solver emits single-cell steps.
      robot_target_cell: (row, col) where the robot should be after the action.
                         For PUSH: robot ends up in the ball's former cell.
      ball_target_cell:  (row, col) where the ball lands.  Only set for PUSH.
  """

  action_type: Literal["MOVE", "PUSH"]
  direction_grid: tuple[int, int]
  robot_target_cell: tuple[int, int]
  ball_target_cell: Optional[tuple[int, int]]


SokobanPlan = list[SokobanAction]

# 6-int key: (robot_row, robot_col, ball_row, ball_col, goal_row, goal_col)
_PlanKey = tuple[int, int, int, int, int, int]

# Maps direction label suffix → (di, dj) = (row_delta, col_delta).
# Derived from maze-branch (dx,dy)=(col_delta,row_delta) by swapping: di=dy, dj=dx.
_DIRECTION_TO_IJ: dict[str, tuple[int, int]] = {
  "up": (-1, 0),
  "down": (+1, 0),
  "left": (0, -1),
  "right": (0, +1),
  "right_down": (+1, +1),
  "left_down": (+1, -1),
  "right_up": (-1, +1),
  "left_up": (-1, -1),
}


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class SokobanGridAbstractionTermCfg(GridAbstractionTermCfg):
  """Configuration for SokobanGridAbstraction.

  Inherits all GridAbstractionTermCfg fields (grid_frame, map,
  wall_center_weight, debug_vis).  direction_method is inherited but unused.

  Attributes:
      robot_entity:        Scene entity name for the robot.
      ball_entity:         Scene entity name for the ball.
      hysteresis_fraction: Fraction of cell_size that a position must penetrate
                           past a grid boundary before the cell index is updated.
                           Prevents rapid flickering near boundaries. Range: 0.1–0.3.
      replan_timeout_s:    Seconds to wait for the solver before declaring failure.
      allow_diagonal:      If True, use the free-diagonal Sokoban problem formulation
                           (robot can move and push diagonally). If False, cardinal only.
  """

  robot_entity: str = "robot"
  ball_entity: str = "ball"
  hysteresis_fraction: float = 0.2
  replan_timeout_s: float = 5.0
  allow_diagonal: bool = True

  def build(self, env: AbstractionBasedEnv) -> SokobanGridAbstraction:
    return SokobanGridAbstraction(cfg=self, env=env)


# ── Term ───────────────────────────────────────────────────────────────────────


class SokobanGridAbstraction(GridAbstraction):
  """Grid abstraction that uses Sokoban-style planning.

  Overrides _compute_direction_map to return None (no direction field),
  then provides its own per-env plan-tracking state and query interface.

  Plan cache
  ----------
  ``plan_cache`` is a shared dict keyed on (_PlanKey) that maps abstract
  configurations to plans.  The first env to encounter a new config solves it
  (via unified_planning + up_symk) and caches the result; all subsequent envs
  with the same config reuse it.  Call ``clear_cache()`` to discard cached
  plans (e.g. when the maze changes).

  Per-env runtime state (tensors of shape [N])
  ---------------------------------------------
  * ``_confirmed_robot_cells``   — hysteresis-filtered robot grid cell.
  * ``_confirmed_ball_cells``    — hysteresis-filtered ball grid cell.
  * ``_robot_prev_cells``        — robot confirmed cell from the previous step;
                                   encodes approach direction during co-occupancy.
  * ``_current_action_is_push``  — True when the current plan step is a PUSH.
  * ``_current_direction_grid``  — (di, dj) direction of the current action.
  * ``plan_length``              — length of each env's current plan (0 if no plan).
  * ``current_step``             — index into the env's current plan.
  * ``expected_next_robot_cell`` — robot must reach this cell to complete current action.
  * ``expected_next_ball_cell``  — ball must reach this cell to complete a PUSH action.
  * ``_move_start_robot_cell``   — robot confirmed cell when MOVE action started.
  * ``_push_start_ball_cell``    — ball confirmed cell when PUSH action started.
  * ``deviation_detected``       — True when replanning failed (no plan / timeout).
                                   Consumed by the termination term.
  """

  cfg: SokobanGridAbstractionTermCfg  # type: ignore[override]

  def __init__(self, cfg: SokobanGridAbstractionTermCfg, env: AbstractionBasedEnv):
    super().__init__(cfg, env)

    self._confirmed_robot_cells = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )
    self._confirmed_ball_cells = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )
    self._robot_prev_cells = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )

    self.current_plan: list[Optional[SokobanPlan]] = [None] * self.num_envs
    self.plan_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.current_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    self._current_action_is_push = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._current_direction_grid = torch.zeros(
      (self.num_envs, 2), dtype=torch.float32, device=self.device
    )

    self.expected_next_robot_cell = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )
    self.expected_next_ball_cell = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )
    self._move_start_robot_cell = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )
    self._push_start_ball_cell = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )
    self.deviation_detected = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.plan_cache: dict[_PlanKey, Optional[SokobanPlan]] = {}

  # ── Override: no direction field ────────────────────────────────────────────

  def _compute_direction_map(
    self,
    cost_maps: torch.Tensor,
    unique_goal_cells: torch.Tensor,
  ) -> None:
    """Opt out of direction-field navigation; plans are used instead."""
    return None

  # ── Lifecycle ────────────────────────────────────────────────────────────────

  def reset(self, env_ids: torch.Tensor) -> None:
    """Rebuild cost map if goal or map changed, then initialise plans for env_ids."""
    super().reset(env_ids)
    if self._env_goal_cells is not None and len(env_ids) > 0:
      robot_pos = self._get_entity_pos_local(self.cfg.robot_entity)
      ball_pos = self._get_entity_pos_local(self.cfg.ball_entity)
      raw_robot = self._local_to_grid(robot_pos)
      raw_ball = self._local_to_grid(ball_pos)
      self._confirmed_robot_cells[env_ids] = raw_robot[env_ids]
      self._confirmed_ball_cells[env_ids] = raw_ball[env_ids]
      self._robot_prev_cells[env_ids] = raw_robot[env_ids]
      self.plan_length[env_ids] = 0
      self.deviation_detected[env_ids] = False
      self._initialize_plans(env_ids)

  def _update_signals(self, env_ids: torch.Tensor) -> None:
    """Advance plan steps, detect deviations, and replan when needed.

    Called every step with env_ids = all envs.
    """
    del env_ids
    if self.settings is None:
      return

    robot_pos = self._get_entity_pos_local(self.cfg.robot_entity)
    ball_pos = self._get_entity_pos_local(self.cfg.ball_entity)

    prev_robot_snapshot = self._confirmed_robot_cells.clone()

    self._confirmed_robot_cells = self._apply_hysteresis(
      robot_pos, self._confirmed_robot_cells
    )
    self._confirmed_ball_cells = self._apply_hysteresis(
      ball_pos, self._confirmed_ball_cells
    )

    # Update prev only when the NEW positions are not co-occupying, so that
    # prev always holds the last pre-contact approach cell.
    not_cooccupying = ~(self._confirmed_robot_cells == self._confirmed_ball_cells).all(dim=1)
    self._robot_prev_cells[not_cooccupying] = prev_robot_snapshot[not_cooccupying]

    robot = self._confirmed_robot_cells  # [N, 2]
    ball = self._confirmed_ball_cells    # [N, 2]

    active = self.current_step < self.plan_length  # [N]
    is_push = self._current_action_is_push & active
    is_move = ~self._current_action_is_push & active

    # ── MOVE completion / deviation ───────────────────────────────────────────
    robot_moved    = ~(robot == self._move_start_robot_cell).all(dim=1)
    robot_reached  = (robot == self.expected_next_robot_cell).all(dim=1)
    move_done      = is_move & robot_moved & robot_reached
    move_deviation = is_move & robot_moved & ~robot_reached

    # ── PUSH completion / deviation ───────────────────────────────────────────
    ball_moved    = ~(ball == self._push_start_ball_cell).all(dim=1)
    ball_reached  = (ball == self.expected_next_ball_cell).all(dim=1)
    push_done     = is_push & ball_moved & ball_reached
    push_deviation = is_push & ball_moved & ~ball_reached

    # ── Online replanning for deviating envs ─────────────────────────────────
    deviated = move_deviation | push_deviation
    if deviated.any():
      deviated_ids = deviated.nonzero(as_tuple=True)[0]
      for env_idx in deviated_ids.tolist():
        action_type = "PUSH" if self._current_action_is_push[env_idx].item() else "MOVE"
        if action_type == "PUSH":
          logger.debug(
            f"[env {env_idx}] PUSH deviation: "
            f"ball at {ball[env_idx].tolist()}, "
            f"expected {self.expected_next_ball_cell[env_idx].tolist()}"
          )
        else:
          logger.debug(
            f"[env {env_idx}] MOVE deviation: "
            f"robot at {robot[env_idx].tolist()}, "
            f"expected {self.expected_next_robot_cell[env_idx].tolist()}"
          )
      self._initialize_plans(deviated_ids)
      # _initialize_plans clears deviation_detected on success and sets it on failure

    # ── Advance step counter for completed actions ────────────────────────────
    advance = move_done | push_done
    self.current_step[advance] += 1
    for env_idx in advance.nonzero(as_tuple=True)[0].tolist():
      self._load_step_targets(env_idx, robot[env_idx], ball[env_idx])

  # ── Hysteresis ───────────────────────────────────────────────────────────────

  def _apply_hysteresis(
    self,
    positions: torch.Tensor,
    confirmed_cells: torch.Tensor,
  ) -> torch.Tensor:
    """Return updated confirmed cell indices with hysteresis applied.

    A cell transition is accepted only when the position has penetrated at least
    ``hysteresis_fraction × cell_size`` past the boundary.  Jumps of more than
    one cell (teleport on reset) are always accepted immediately.

    Args:
        positions:       [N, 2] local (x, y) positions.
        confirmed_cells: [N, 2] current confirmed (i, j) indices.

    Returns:
        [N, 2] updated confirmed (i, j) indices.
    """
    h = self.cfg.hysteresis_fraction * self.grid_frame.cell_size
    cs = self.grid_frame.cell_size
    xmc = self.grid_frame.x_map_center
    ymc = self.grid_frame.y_map_center
    cx = self.grid_frame.center_x
    cy = self.grid_frame.center_y

    raw_cells = self._local_to_grid(positions)
    new_cells = confirmed_cells.clone()

    pos_x = positions[:, 0]
    pos_y = positions[:, 1]
    old_i = confirmed_cells[:, 0].float()
    old_j = confirmed_cells[:, 1].float()
    raw_i = raw_cells[:, 0].float()
    raw_j = raw_cells[:, 1].float()

    # j-axis (x ↔ j, same direction)
    going_right = raw_j > old_j
    going_left  = raw_j < old_j
    large_j     = (raw_j - old_j).abs() > 1

    bnd_right = (old_j + 1) * cs - xmc + cx
    bnd_left  = old_j * cs - xmc + cx

    accept_j = (
      large_j
      | (going_right & (pos_x >= bnd_right + h))
      | (going_left  & (pos_x <= bnd_left  - h))
    )
    new_cells[:, 1] = torch.where(accept_j, raw_cells[:, 1], confirmed_cells[:, 1])

    # i-axis (y ↔ i inverted: i increases downward, y increases upward)
    going_down = raw_i > old_i
    going_up   = raw_i < old_i
    large_i    = (raw_i - old_i).abs() > 1

    bnd_down = ymc - (old_i + 1) * cs + cy
    bnd_up   = ymc - old_i * cs + cy

    accept_i = (
      large_i
      | (going_down & (pos_y <= bnd_down - h))
      | (going_up   & (pos_y >= bnd_up   + h))
    )
    new_cells[:, 0] = torch.where(accept_i, raw_cells[:, 0], confirmed_cells[:, 0])

    return new_cells

  # ── Plan management ──────────────────────────────────────────────────────────

  def _initialize_plans(self, env_ids: torch.Tensor) -> None:
    """Look up or solve plans for env_ids and reset their step counters.

    2-MDP convention: when robot and ball share a cell, the robot is treated as
    being at ``_robot_prev_cells`` (the last cell it occupied before entering the
    ball's cell).  This keeps Sokoban state well-defined during co-occupancy.
    """
    assert self._env_goal_cells is not None
    goal_cells = self._env_goal_cells
    map_cpu = self.map.cpu()

    for env_idx in env_ids.tolist():
      ball_cell = self._confirmed_ball_cells[env_idx]
      bi, bj    = int(ball_cell[0].item()), int(ball_cell[1].item())

      # 2-MDP: use confirmed robot cell unless co-occupying, then use prev.
      robot_cell = self._confirmed_robot_cells[env_idx]
      ri, rj     = int(robot_cell[0].item()), int(robot_cell[1].item())
      if ri == bi and rj == bj:
        robot_cell = self._robot_prev_cells[env_idx]
        ri, rj     = int(robot_cell[0].item()), int(robot_cell[1].item())
        if ri == bi and rj == bj:
          # prev is also at ball's cell — ball bounced into robot's post-push
          # position. Skip replanning; plan stays active and will recover.
          logger.debug(f"[env {env_idx}] Co-occupancy: prev also at ball cell, skipping replan")
          continue

      if map_cpu[ri, rj].item() or map_cpu[bi, bj].item():
        logger.debug(
          f"[env {env_idx}] Skipping plan init: robot({ri},{rj}) or ball({bi},{bj}) in wall"
        )
        continue

      key: _PlanKey = (ri, rj, bi, bj, int(goal_cells[env_idx, 0].item()), int(goal_cells[env_idx, 1].item()))

      if key not in self.plan_cache:
        self.plan_cache[key] = self._solve_sokoban(key)

      plan = self.plan_cache[key]

      if plan is None:
        self.deviation_detected[env_idx] = True
        logger.warning(f"[env {env_idx}] No Sokoban plan found for key {key}; episode will terminate")
        continue

      self.current_plan[env_idx] = plan
      self.plan_length[env_idx]  = len(plan)
      self.current_step[env_idx] = 0
      self.deviation_detected[env_idx] = False
      self._load_step_targets(env_idx, robot_cell, ball_cell)

  def _load_step_targets(
    self,
    env_idx: int,
    current_robot_cell: torch.Tensor,
    current_ball_cell: torch.Tensor,
  ) -> None:
    """Update expected_next_*_cell and start snapshots from the current plan step."""
    plan = self.current_plan[env_idx]
    step = int(self.current_step[env_idx].item())

    if plan is None or step >= len(plan):
      return

    action = plan[step]
    is_push = action.action_type == "PUSH"
    self._current_action_is_push[env_idx] = is_push
    self._current_direction_grid[env_idx] = torch.tensor(
      [float(action.direction_grid[0]), float(action.direction_grid[1])],
      device=self.device, dtype=torch.float32,
    )
    self.expected_next_robot_cell[env_idx] = torch.tensor(
      action.robot_target_cell, device=self.device, dtype=torch.long
    )

    if is_push:
      assert action.ball_target_cell is not None
      self.expected_next_ball_cell[env_idx] = torch.tensor(
        action.ball_target_cell, device=self.device, dtype=torch.long
      )
      self._push_start_ball_cell[env_idx] = current_ball_cell.clone()
    else:
      self._move_start_robot_cell[env_idx] = current_robot_cell.clone()

  # ── Solver ───────────────────────────────────────────────────────────────────

  def _build_level_string(self, config: _PlanKey) -> str:
    """Build a Sokoban level string from a plan key and the obstacle map."""
    robot_row, robot_col, ball_row, ball_col, goal_row, goal_col = config
    rows = self.grid_frame.num_rows
    cols = self.grid_frame.num_cols
    map_cpu = self.map.cpu()

    grid = [
      [ENCODING["WALL"] if map_cpu[r, c].item() else ENCODING["EMPTY"]
       for c in range(cols)]
      for r in range(rows)
    ]
    grid[robot_row][robot_col] = ENCODING["ROBOT"]
    grid[ball_row][ball_col]   = ENCODING["BLOCK"]
    grid[goal_row][goal_col]   = ENCODING["GOAL"]
    return maze_to_string(grid)

  def _solve_sokoban(self, config: _PlanKey) -> Optional[SokobanPlan]:
    """Solve a Sokoban configuration using unified_planning.

    Returns a SokobanPlan (possibly empty if ball is already at goal),
    or None on solver failure / timeout.
    """
    robot_row, robot_col, ball_row, ball_col, goal_row, goal_col = config

    if ball_row == goal_row and ball_col == goal_col:
      return []

    level_str = self._build_level_string(config)

    if self.cfg.allow_diagonal:
      problem = create_free_diagonal_sokoban_problem(level_str)
    else:
      problem = create_sokoban_problem(level_str)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      future = executor.submit(solve_sokoban_problem, problem)
      try:
        up_plan = future.result(timeout=self.cfg.replan_timeout_s)
      except concurrent.futures.TimeoutError:
        logger.warning(f"Sokoban solver timed out (>{self.cfg.replan_timeout_s}s) for key {config}")
        return None

    if up_plan is None:
      logger.warning(f"Sokoban solver found no plan for key {config}")
      return None

    return self._convert_plan(up_plan.actions, robot_row, robot_col, ball_row, ball_col)

  def _convert_plan(
    self,
    actions: list,
    robot_row: int,
    robot_col: int,
    ball_row: int,
    ball_col: int,
  ) -> SokobanPlan:
    """Convert unified_planning actions to SokobanPlan.

    Walks through direction-label strings and tracks (robot, ball) cell positions
    to build SokobanAction objects.  Coordinate conversion:
      maze-branch (dx,dy) = (col_delta, row_delta)
      this branch  (di,dj) = (row_delta, col_delta)
      → di = dy, dj = dx  (swap)
    """
    steps = plan_to_simple_steps(actions)
    plan: SokobanPlan = []
    r_row, r_col = robot_row, robot_col
    b_row, b_col = ball_row, ball_col

    for step_label in steps:
      if step_label.startswith("move_"):
        direction = step_label[len("move_"):]
        di, dj = _DIRECTION_TO_IJ[direction]
        new_r_row, new_r_col = r_row + di, r_col + dj
        plan.append(SokobanAction(
          action_type="MOVE",
          direction_grid=(di, dj),
          robot_target_cell=(new_r_row, new_r_col),
          ball_target_cell=None,
        ))
        r_row, r_col = new_r_row, new_r_col

      elif step_label.startswith("push_"):
        direction = step_label[len("push_"):]
        di, dj = _DIRECTION_TO_IJ[direction]
        new_b_row, new_b_col = b_row + di, b_col + dj
        plan.append(SokobanAction(
          action_type="PUSH",
          direction_grid=(di, dj),
          robot_target_cell=(b_row, b_col),  # robot ends where ball was
          ball_target_cell=(new_b_row, new_b_col),
        ))
        r_row, r_col = b_row, b_col
        b_row, b_col = new_b_row, new_b_col

      else:
        raise ValueError(f"Unknown step label: {step_label!r}")

    return plan

  # ── Entity helpers ───────────────────────────────────────────────────────────

  def _get_entity_pos_local(self, entity_name: str) -> torch.Tensor:
    """Return [N, 2] local (x, y) positions for the named entity."""
    entity = self._env.scene[entity_name]
    pos_w = entity.data.root_link_pos_w[:, :2]
    return pos_w - self._env.scene.env_origins[:, :2]

  # ── Query interface ──────────────────────────────────────────────────────────

  @property
  def current_actions(self) -> list[Optional[SokobanAction]]:
    """Current SokobanAction per env, or None if plan is done/uninitialized."""
    result: list[Optional[SokobanAction]] = []
    for env_idx in range(self.num_envs):
      plan = self.current_plan[env_idx]
      step = int(self.current_step[env_idx].item())
      result.append(plan[step] if (plan is not None and step < len(plan)) else None)
    return result

  def clear_cache(self) -> None:
    """Invalidate all cached plans. Call when the maze layout changes."""
    self.plan_cache.clear()
    logger.info("SokobanGridAbstraction: plan cache cleared")

  # ── Debug visualization ──────────────────────────────────────────────────────

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    super()._debug_vis_impl(visualizer)

    env_idx = visualizer.env_idx
    if env_idx >= self.num_envs:
      return

    plan = self.current_plan[env_idx]
    step = int(self.current_step[env_idx].item())
    if plan is None or step >= len(plan):
      return

    env_origin = self._env.scene.env_origins[env_idx, :2].cpu().numpy()

    def cell_world(row: int, col: int, z: float) -> np.ndarray:
      idx = torch.tensor([[row, col]], dtype=torch.long, device=self.device)
      xy = self._grid_to_local(idx, center=True)[0].cpu().numpy()
      return np.array([xy[0] + env_origin[0], xy[1] + env_origin[1], z])

    action = plan[step]
    if action.action_type == "MOVE":
      r_row = int(self._confirmed_robot_cells[env_idx, 0].item())
      r_col = int(self._confirmed_robot_cells[env_idx, 1].item())
      tgt_row, tgt_col = action.robot_target_cell
      visualizer.add_arrow(
        start=cell_world(r_row, r_col, 0.5),
        end=cell_world(tgt_row, tgt_col, 0.5),
        color=(0.1, 0.9, 0.1, 1.0),
        width=0.08,
        label=f"MOVE→({tgt_row},{tgt_col})",
      )
    else:  # PUSH
      assert action.ball_target_cell is not None
      b_row = int(self._confirmed_ball_cells[env_idx, 0].item())
      b_col = int(self._confirmed_ball_cells[env_idx, 1].item())
      tgt_row, tgt_col = action.ball_target_cell
      visualizer.add_arrow(
        start=cell_world(b_row, b_col, 0.2),
        end=cell_world(tgt_row, tgt_col, 0.2),
        color=(1.0, 0.55, 0.0, 1.0),
        width=0.08,
        label=f"PUSH→({tgt_row},{tgt_col})",
      )
