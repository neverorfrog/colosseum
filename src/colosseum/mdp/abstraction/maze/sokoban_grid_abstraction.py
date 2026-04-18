"""Sokoban-style planning abstraction for the soccer-maze task.

Inherits Dijkstra cost map from GridAbstraction but opts out of the direction
field entirely (returns None from _compute_direction_map).  Instead it maintains
a per-env discrete plan (MOVE robot / PUSH ball sequence) looked up from a shared
cache keyed on the abstract configuration (robot_cell, ball_cell, goal_cell).

Lifecycle:
  reset(env_ids)     → cost map rebuilt if goal or obstacles changed, then plans
                       initialized for env_ids at their current (post-reset) positions.
  compute(dt)        → _update_signals(all_envs): advance plan step when the
                       current action's goal is reached; detect deviations.

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

Deviation detection (symmetric for MOVE and PUSH)
  MOVE: robot left ``_move_start_robot_cell`` but ended up in a cell that is not
        ``expected_next_robot_cell`` → deviation.
  PUSH: ball left ``_push_start_ball_cell`` but ended up in a cell that is not
        ``expected_next_ball_cell`` → deviation.
  Both set ``deviation_detected`` which is consumed by the termination term.

Vectorization
  ``_update_signals`` is fully tensor-parallel (no Python loop over envs in the
  hot path).  ``_load_step_targets`` is called only for envs whose plan step
  actually advanced (rare), so its Python loop is negligible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import torch
from loguru import logger

from colosseum.mdp.abstraction.maze.grid_abstraction import (
  GridAbstraction,
  GridAbstractionTermCfg,
)

if TYPE_CHECKING:
  from colosseum.envs.abstraction_based_env import AbstractionBasedEnv


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class SokobanAction:
  """One step in a Sokoban plan.

  Attributes:
      action_type:       "MOVE" (robot navigates to a position) or
                         "PUSH" (robot pushes the ball one cell).
      direction_grid:    (di, dj) primary direction of the action in grid space.
                         For PUSH this is the exact push direction (cardinal unit).
                         For MOVE this is the displacement to the target (not
                         necessarily unit; SokobanCommand normalises via
                         GridFrame.grid_direction_to_local).
      robot_target_cell: (row, col) where the robot should be after the action
                         completes.  For PUSH the robot ends up in the ball's
                         former cell (co-occupancy until ball moves).
      ball_target_cell:  (row, col) where the ball lands.  Only set for PUSH;
                         None for MOVE.
  """

  action_type: Literal["MOVE", "PUSH"]
  direction_grid: tuple[int, int]
  robot_target_cell: tuple[int, int]
  ball_target_cell: Optional[tuple[int, int]]


SokobanPlan = list[SokobanAction]

# 6-int key: (robot_row, robot_col, ball_row, ball_col, goal_row, goal_col)
_PlanKey = tuple[int, int, int, int, int, int]


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class SokobanGridAbstractionTermCfg(GridAbstractionTermCfg):
  """Configuration for SokobanGridAbstraction.

  Inherits all GridAbstractionTermCfg fields (grid_frame, map,
  wall_center_weight, debug_vis).  direction_method is inherited but unused.

  Attributes:
      robot_entity:       Scene entity name for the robot.
      ball_entity:        Scene entity name for the ball.
      hysteresis_fraction: Fraction of cell_size that a position must penetrate
                           past a grid boundary before the cell index is updated.
                           Prevents rapid flickering near boundaries.
                           Typical range: 0.1–0.3.
  """

  robot_entity: str = "robot"
  ball_entity: str = "ball"
  hysteresis_fraction: float = 0.2

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
  and caches the result; all subsequent envs with the same config reuse it.
  Call ``clear_cache()`` to discard cached plans (e.g. when the maze changes).

  Per-env runtime state (tensors of shape [N])
  ---------------------------------------------
  * ``_confirmed_robot_cells``   — hysteresis-filtered robot grid cell.
  * ``_confirmed_ball_cells``    — hysteresis-filtered ball grid cell.
  * ``_robot_prev_cells``        — robot confirmed cell from the previous step;
                                   encodes approach direction during co-occupancy.
  * ``_current_action_is_push``  — True when the current plan step is a PUSH.
  * ``plan_length``              — length of each env's current plan (0 if no plan).
  * ``current_step``             — index into the env's current plan.
  * ``expected_next_robot_cell`` — robot must reach this cell to complete current action.
  * ``expected_next_ball_cell``  — ball must reach this cell to complete a PUSH action.
  * ``_move_start_robot_cell``   — robot confirmed cell when MOVE action started;
                                   used to detect when the robot has actually moved.
  * ``_push_start_ball_cell``    — ball confirmed cell when PUSH action started;
                                   used to detect when the ball has actually moved.
  * ``deviation_detected``       — True when robot (MOVE) or ball (PUSH) ended up in
                                   the wrong cell.  Consumed by the termination term;
                                   cleared on next reset.
  """

  cfg: SokobanGridAbstractionTermCfg  # type: ignore[override]

  def __init__(self, cfg: SokobanGridAbstractionTermCfg, env: AbstractionBasedEnv):
    super().__init__(cfg, env)

    # Hysteresis-filtered cell indices
    self._confirmed_robot_cells = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )
    self._confirmed_ball_cells = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )
    # Previous-step robot cell (2-MDP: records approach direction during co-occupancy)
    self._robot_prev_cells = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )

    # Per-env plan state
    self.current_plan: list[Optional[SokobanPlan]] = [None] * self.num_envs
    self.plan_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.current_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # Current action type and direction (needed for vectorized command computation)
    self._current_action_is_push = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # Grid-space direction (di, dj) of the current action, as floats.
    # Read by SokobanCommand to produce velocity commands without Python loops.
    self._current_direction_grid = torch.zeros(
      (self.num_envs, 2), dtype=torch.float32, device=self.device
    )

    self.expected_next_robot_cell = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )
    self.expected_next_ball_cell = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )
    # Snapshots at action-load time (detect actual movement vs. start position)
    self._move_start_robot_cell = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )
    self._push_start_ball_cell = torch.zeros(
      (self.num_envs, 2), dtype=torch.long, device=self.device
    )
    self.deviation_detected = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # Shared plan cache across all envs
    self.plan_cache: dict[_PlanKey, SokobanPlan] = {}

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
    super().reset(env_ids)  # _update_settings → _maybe_rebuild (Dijkstra)
    if self._env_goal_cells is not None and len(env_ids) > 0:
      robot_pos = self._get_entity_pos_local(self.cfg.robot_entity)
      ball_pos = self._get_entity_pos_local(self.cfg.ball_entity)
      # Bypass hysteresis on reset: snap confirmed cells to raw positions.
      raw_robot = self._local_to_grid(robot_pos)
      raw_ball = self._local_to_grid(ball_pos)
      self._confirmed_robot_cells[env_ids] = raw_robot[env_ids]
      self._confirmed_ball_cells[env_ids] = raw_ball[env_ids]
      self._robot_prev_cells[env_ids] = raw_robot[env_ids]  # no history yet
      self._initialize_plans(env_ids)

  def _update_signals(self, env_ids: torch.Tensor) -> None:
    """Advance plan steps and detect deviations, fully vectorised over all envs.

    Called every step with env_ids = all envs (from AbstractionTerm.compute).
    """
    del env_ids  # Always operate on all envs
    if self.settings is None:
      return

    robot_pos = self._get_entity_pos_local(self.cfg.robot_entity)
    ball_pos = self._get_entity_pos_local(self.cfg.ball_entity)

    # 2-MDP: record previous robot cell before updating
    self._robot_prev_cells = self._confirmed_robot_cells.clone()

    # Apply hysteresis to get stable confirmed cell indices
    self._confirmed_robot_cells = self._apply_hysteresis(
      robot_pos, self._confirmed_robot_cells
    )
    self._confirmed_ball_cells = self._apply_hysteresis(
      ball_pos, self._confirmed_ball_cells
    )

    robot = self._confirmed_robot_cells  # [N, 2]
    ball = self._confirmed_ball_cells    # [N, 2]

    # Only operate on envs that have an active plan step
    active = self.current_step < self.plan_length  # [N]

    is_push = self._current_action_is_push & active   # [N]
    is_move = ~self._current_action_is_push & active  # [N]

    # ── MOVE: robot must leave start cell and arrive at target ────────────────
    robot_moved   = ~(robot == self._move_start_robot_cell).all(dim=1)   # [N]
    robot_reached  = (robot == self.expected_next_robot_cell).all(dim=1)  # [N]
    move_done      = is_move & robot_moved & robot_reached
    move_deviation = is_move & robot_moved & ~robot_reached

    # ── PUSH: ball must leave start cell and arrive at target ─────────────────
    ball_moved    = ~(ball == self._push_start_ball_cell).all(dim=1)     # [N]
    ball_reached   = (ball == self.expected_next_ball_cell).all(dim=1)   # [N]
    push_done      = is_push & ball_moved & ball_reached
    push_deviation = is_push & ball_moved & ~ball_reached

    # ── Debug logging (only when deviations occur) ────────────────────────────
    if move_deviation.any():
      for env_idx in move_deviation.nonzero(as_tuple=True)[0].tolist():
        logger.debug(
          f"[env {env_idx}] MOVE deviation: "
          f"robot at {robot[env_idx].tolist()}, "
          f"expected {self.expected_next_robot_cell[env_idx].tolist()}"
        )
    if push_deviation.any():
      for env_idx in push_deviation.nonzero(as_tuple=True)[0].tolist():
        prev = self._robot_prev_cells[env_idx]
        approach = (
          int((ball[env_idx, 0] - prev[0]).item()),
          int((ball[env_idx, 1] - prev[1]).item()),
        )
        logger.debug(
          f"[env {env_idx}] PUSH deviation: "
          f"ball at {ball[env_idx].tolist()}, "
          f"expected {self.expected_next_ball_cell[env_idx].tolist()}, "
          f"robot approach direction {approach}"
        )

    self.deviation_detected |= move_deviation | push_deviation

    # ── Advance step counter for completed actions ────────────────────────────
    advance = move_done | push_done  # [N]
    self.current_step[advance] += 1

    # Load targets for the new step (Python loop only over advancing envs)
    for env_idx in advance.nonzero(as_tuple=True)[0].tolist():
      self._load_step_targets(env_idx, robot[env_idx], ball[env_idx])

  # ── Hysteresis ───────────────────────────────────────────────────────────────

  def _apply_hysteresis(
    self,
    positions: torch.Tensor,
    confirmed_cells: torch.Tensor,
  ) -> torch.Tensor:
    """Return updated confirmed cell indices with hysteresis applied.

    A cell transition from confirmed cell c to raw cell c' is accepted only when
    the continuous position has penetrated at least ``hysteresis_fraction × cell_size``
    past the boundary in the direction of c'.  Jumps of more than one cell
    (teleport, fast movement) are always accepted immediately.

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

    raw_cells = self._local_to_grid(positions)  # [N, 2]
    new_cells = confirmed_cells.clone()

    pos_x = positions[:, 0]  # [N]
    pos_y = positions[:, 1]  # [N]
    old_i = confirmed_cells[:, 0].float()
    old_j = confirmed_cells[:, 1].float()
    raw_i = raw_cells[:, 0].float()
    raw_j = raw_cells[:, 1].float()

    # ── j-axis  (x ↔ j, same direction) ──────────────────────────────────────
    # Boundary right: x = (old_j + 1) * cs - xmc + cx
    # Boundary left:  x = old_j * cs - xmc + cx
    going_right = raw_j > old_j
    going_left = raw_j < old_j
    large_j = (raw_j - old_j).abs() > 1  # teleport / fast: bypass hysteresis

    bnd_right = (old_j + 1) * cs - xmc + cx
    bnd_left = old_j * cs - xmc + cx

    accept_j = (
      large_j
      | (going_right & (pos_x >= bnd_right + h))
      | (going_left & (pos_x <= bnd_left - h))
    )
    new_cells[:, 1] = torch.where(accept_j, raw_cells[:, 1], confirmed_cells[:, 1])

    # ── i-axis  (y ↔ i, inverted: i increases downward, y increases upward) ───
    # Boundary down (i→i+1): y = ymc - (old_i + 1) * cs + cy  (lower y)
    # Boundary up   (i→i-1): y = ymc - old_i * cs + cy        (higher y)
    going_down = raw_i > old_i  # larger i = lower y
    going_up = raw_i < old_i
    large_i = (raw_i - old_i).abs() > 1

    bnd_down = ymc - (old_i + 1) * cs + cy
    bnd_up = ymc - old_i * cs + cy

    accept_i = (
      large_i
      | (going_down & (pos_y <= bnd_down - h))
      | (going_up & (pos_y >= bnd_up + h))
    )
    new_cells[:, 0] = torch.where(accept_i, raw_cells[:, 0], confirmed_cells[:, 0])

    return new_cells

  # ── Plan management ──────────────────────────────────────────────────────────

  def _initialize_plans(self, env_ids: torch.Tensor) -> None:
    """Look up or solve plans for env_ids and reset their step counters.

    Uses already-updated ``_confirmed_robot_cells`` and ``_confirmed_ball_cells``
    (set in reset() before this is called).
    """
    assert self._env_goal_cells is not None
    goal_cells = self._env_goal_cells  # [N, 2] — set by _update_settings

    for env_idx in env_ids.tolist():
      robot_cell = self._confirmed_robot_cells[env_idx]
      ball_cell = self._confirmed_ball_cells[env_idx]

      key: _PlanKey = (
        int(robot_cell[0].item()),
        int(robot_cell[1].item()),
        int(ball_cell[0].item()),
        int(ball_cell[1].item()),
        int(goal_cells[env_idx, 0].item()),
        int(goal_cells[env_idx, 1].item()),
      )

      if key not in self.plan_cache:
        self.plan_cache[key] = self._solve_sokoban(key)

      plan = self.plan_cache[key]
      self.current_plan[env_idx] = plan
      self.plan_length[env_idx] = len(plan)
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
      # Snapshot ball's cell at push start to detect actual ball movement.
      self._push_start_ball_cell[env_idx] = current_ball_cell.clone()
    else:
      # Snapshot robot's cell at move start to detect actual robot movement.
      self._move_start_robot_cell[env_idx] = current_robot_cell.clone()

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

  # ── Stub solver ──────────────────────────────────────────────────────────────

  def _solve_sokoban(self, config: _PlanKey) -> SokobanPlan:
    """Generate a plan for the given abstract configuration.

    Stub: one greedy PUSH toward goal along the dominant Manhattan axis.
    Real solver: override this method with a PDDL call.

    The MOVE action targets the cell adjacent to the ball on the opposite side
    from the push direction (standard Sokoban approach position).  When the
    robot and ball share a cell (co-occupancy on reset), the initial MOVE is
    omitted and the PUSH action is issued directly.

    Args:
        config: (robot_row, robot_col, ball_row, ball_col, goal_row, goal_col)

    Returns:
        List of SokobanActions (possibly empty if ball is already at goal).
    """
    robot_row, robot_col, ball_row, ball_col, goal_row, goal_col = config

    if ball_row == goal_row and ball_col == goal_col:
      return []

    # Dominant axis toward goal
    dr = goal_row - ball_row
    dc = goal_col - ball_col
    if abs(dr) >= abs(dc):
      push_di, push_dj = (1 if dr > 0 else -1), 0
    else:
      push_di, push_dj = 0, (1 if dc > 0 else -1)

    # Standard Sokoban approach: robot must be on the opposite side of the ball.
    robot_push_pos = (ball_row - push_di, ball_col - push_dj)
    ball_target = (ball_row + push_di, ball_col + push_dj)

    actions: SokobanPlan = []

    if (robot_row, robot_col) != robot_push_pos:
      move_di = robot_push_pos[0] - robot_row
      move_dj = robot_push_pos[1] - robot_col
      actions.append(
        SokobanAction(
          action_type="MOVE",
          direction_grid=(move_di, move_dj),
          robot_target_cell=robot_push_pos,
          ball_target_cell=None,
        )
      )

    # PUSH: robot enters ball_cell (co-occupancy) → ball moves to ball_target.
    actions.append(
      SokobanAction(
        action_type="PUSH",
        direction_grid=(push_di, push_dj),
        robot_target_cell=(ball_row, ball_col),  # robot ends up where ball was
        ball_target_cell=ball_target,
      )
    )

    return actions
