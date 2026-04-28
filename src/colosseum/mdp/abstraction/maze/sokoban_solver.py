"""Sokoban PDDL solver using unified_planning + up_symk.

Provides:
  create_sokoban_problem()               – cardinal moves/pushes only
  create_diagonal_sokoban_problem()      – cardinal + diagonal (free corners)
  create_free_diagonal_sokoban_problem() – cardinal + free diagonal (corner-aware)
  solve_sokoban_problem()                – call OneshotPlanner, return SequentialPlan or None
  plan_to_simple_steps()                 – convert UP plan to List[str] direction labels
"""

from __future__ import annotations

import re
from typing import List, Tuple

import unified_planning as up
import unified_planning.engines.results
from unified_planning.shortcuts import (
  And,
  BoolType,
  DerivedBoolType,
  Exists,
  Not,
  OneshotPlanner,
  Variable,
)

# ── Grid encoding ─────────────────────────────────────────────────────────────

ENCODING = {
  "WALL": "#",
  "EMPTY": ".",
  "ROBOT": "r",
  "GOAL": "@",
  "BLOCK": "b",
  "ELBOW": ".",
}


def maze_to_string(maze: List[List]) -> str:
  return "\n".join("".join(str(cell) for cell in row) for row in maze) + "\n"


# ── Problem builders ──────────────────────────────────────────────────────────


def create_sokoban_problem(level: str, with_axioms: bool = False):
  problem = up.model.Problem("sokoban")
  loc = up.shortcuts.UserType("location")

  has_player = up.model.Fluent("has_player", BoolType(), l=loc)
  has_box = up.model.Fluent("has_box", BoolType(), l=loc)
  adjacent = up.model.Fluent("adjacent", BoolType(), l1=loc, l2=loc)
  adjacent_2 = up.model.Fluent("adjacent_2", BoolType(), l1=loc, l2=loc)

  problem.add_fluent(has_player, default_initial_value=False)
  problem.add_fluent(has_box, default_initial_value=False)
  problem.add_fluent(adjacent, default_initial_value=False)
  problem.add_fluent(adjacent_2, default_initial_value=False)

  if with_axioms:
    can_reach = up.model.Fluent("can_reach", DerivedBoolType(), l=loc)
    problem.add_fluent(can_reach, default_initial_value=False)

    a1 = up.model.Axiom("reach-axiom-1", l=loc)
    a1.set_head(can_reach(a1.parameters[0]))
    a1.add_body_condition(has_player(a1.parameters[0]))
    problem.add_axiom(a1)

    a2 = up.model.Axiom("reach-axiom-2", to=loc)
    to, fr = a2.parameters[0], Variable("from", loc)
    a2.set_head(can_reach(to))
    a2.add_body_condition(Not(has_box(to)))
    a2.add_body_condition(Exists(And(can_reach(fr), adjacent(fr, to)), fr))
    problem.add_axiom(a2)

  cost_dict = {}

  if not with_axioms:
    move = up.model.InstantaneousAction("move", fr=loc, to=loc)
    fr, to = move.parameters
    move.add_precondition(adjacent(fr, to))
    move.add_precondition(has_player(fr))
    move.add_precondition(Not(has_box(to)))
    move.add_effect(has_player(fr), False)
    move.add_effect(has_player(to), True)
    cost_dict[move] = 0
    problem.add_action(move)

    push_box = up.model.InstantaneousAction("push-box", x=loc, y=loc, z=loc)
    x, y, z = push_box.parameters
    push_box.add_precondition(adjacent(x, y))
    push_box.add_precondition(adjacent(y, z))
    push_box.add_precondition(adjacent_2(x, z))
    push_box.add_precondition(has_player(x))
    push_box.add_precondition(has_box(y))
    push_box.add_precondition(Not(has_box(z)))
    push_box.add_effect(has_player(x), False)
    push_box.add_effect(has_player(y), True)
    push_box.add_effect(has_box(y), False)
    push_box.add_effect(has_box(z), True)
    cost_dict[push_box] = 1
    problem.add_action(push_box)

  else:
    push_box = up.model.InstantaneousAction("push-box", l=loc, x=loc, y=loc, z=loc)
    l, x, y, z = push_box.parameters
    push_box.add_precondition(adjacent(x, y))
    push_box.add_precondition(adjacent(y, z))
    push_box.add_precondition(adjacent_2(x, z))
    push_box.add_precondition(has_player(l))
    push_box.add_precondition(can_reach(x))
    push_box.add_precondition(has_box(y))
    push_box.add_precondition(Not(has_box(z)))
    push_box.add_effect(has_player(l), False)
    push_box.add_effect(has_player(y), True)
    push_box.add_effect(has_box(y), False)
    push_box.add_effect(has_box(z), True)
    cost_dict[push_box] = 1
    problem.add_action(push_box)

  problem.add_quality_metric(up.model.metrics.MinimizeActionCosts(cost_dict))

  level_array = [[ch for ch in line] for line in level.splitlines()]
  num_col = len(level_array[0])
  num_row = len(level_array)
  loc_objs: dict[str, up.model.Object] = {}

  for x in range(num_col):
    for y in range(num_row):
      entry = level_array[y][x]
      if entry != ENCODING["WALL"]:
        loc_id = f"loc-{x}-{y}"
        obj = up.model.Object(loc_id, loc)
        loc_objs[loc_id] = obj
        problem.add_object(obj)
        if entry == ENCODING["ROBOT"]:
          problem.set_initial_value(has_player(obj), True)
        elif entry == ENCODING["BLOCK"]:
          problem.set_initial_value(has_box(obj), True)
        elif entry == ENCODING["GOAL"]:
          problem.add_goal(has_box(obj))

  for obj1 in loc_objs:
    for obj2 in loc_objs:
      x1, y1 = map(int, str(obj1).split("-")[1:])
      x2, y2 = map(int, str(obj2).split("-")[1:])
      dist = abs(x1 - x2) + abs(y1 - y2)
      if dist == 1:
        problem.set_initial_value(adjacent(loc_objs[obj1], loc_objs[obj2]), True)
      elif dist == 2 and (x1 == x2 or y1 == y2):
        problem.set_initial_value(adjacent_2(loc_objs[obj1], loc_objs[obj2]), True)

  return problem


def create_diagonal_sokoban_problem(
  level: str,
  with_axioms: bool = False,
  with_diagonal_push: bool = True,
):
  """Cardinal + diagonal moves; diagonal adjacency requires corner cell to be free."""
  problem = up.model.Problem("sokoban")
  loc = up.shortcuts.UserType("location")

  has_player = up.model.Fluent("has_player", BoolType(), l=loc)
  has_box = up.model.Fluent("has_box", BoolType(), l=loc)
  adjacent = up.model.Fluent("adjacent", BoolType(), l1=loc, l2=loc)
  adjacent_2 = up.model.Fluent("adjacent_2", BoolType(), l1=loc, l2=loc)
  adjacent_diag = up.model.Fluent("adjacent_diag", BoolType(), l1=loc, l2=loc)

  for fl in [has_player, has_box, adjacent, adjacent_2, adjacent_diag]:
    problem.add_fluent(fl, default_initial_value=False)

  if with_diagonal_push:
    adjacent_2_diag = up.model.Fluent("adjacent_2_diag", BoolType(), l1=loc, l2=loc)
    problem.add_fluent(adjacent_2_diag, default_initial_value=False)

  cost_dict = {}

  move = up.model.InstantaneousAction("move", fr=loc, to=loc)
  fr, to = move.parameters
  move.add_precondition(adjacent(fr, to))
  move.add_precondition(has_player(fr))
  move.add_precondition(Not(has_box(to)))
  move.add_effect(has_player(fr), False)
  move.add_effect(has_player(to), True)
  cost_dict[move] = 0
  problem.add_action(move)

  move_diag = up.model.InstantaneousAction("move-diag", fr=loc, to=loc)
  fr, to = move_diag.parameters
  move_diag.add_precondition(adjacent_diag(fr, to))
  move_diag.add_precondition(has_player(fr))
  move_diag.add_precondition(Not(has_box(to)))
  move_diag.add_effect(has_player(fr), False)
  move_diag.add_effect(has_player(to), True)
  cost_dict[move_diag] = 0
  problem.add_action(move_diag)

  push_box = up.model.InstantaneousAction("push-box", x=loc, y=loc, z=loc)
  x, y, z = push_box.parameters
  push_box.add_precondition(adjacent(x, y))
  push_box.add_precondition(adjacent(y, z))
  push_box.add_precondition(adjacent_2(x, z))
  push_box.add_precondition(has_player(x))
  push_box.add_precondition(has_box(y))
  push_box.add_precondition(Not(has_box(z)))
  push_box.add_effect(has_player(x), False)
  push_box.add_effect(has_player(y), True)
  push_box.add_effect(has_box(y), False)
  push_box.add_effect(has_box(z), True)
  cost_dict[push_box] = 1
  problem.add_action(push_box)

  if with_diagonal_push:
    push_box_diag = up.model.InstantaneousAction("push-box-diag", x=loc, y=loc, z=loc)
    x, y, z = push_box_diag.parameters
    push_box_diag.add_precondition(adjacent_diag(x, y))
    push_box_diag.add_precondition(adjacent_diag(y, z))
    push_box_diag.add_precondition(adjacent_2_diag(x, z))
    push_box_diag.add_precondition(has_player(x))
    push_box_diag.add_precondition(has_box(y))
    push_box_diag.add_precondition(Not(has_box(z)))
    push_box_diag.add_effect(has_player(x), False)
    push_box_diag.add_effect(has_player(y), True)
    push_box_diag.add_effect(has_box(y), False)
    push_box_diag.add_effect(has_box(z), True)
    cost_dict[push_box_diag] = 1
    problem.add_action(push_box_diag)

  problem.add_quality_metric(up.model.metrics.MinimizeActionCosts(cost_dict))

  level_array = [[ch for ch in line] for line in level.splitlines()]
  num_col = len(level_array[0])
  num_row = len(level_array)
  loc_objs: dict[str, up.model.Object] = {}

  for x in range(num_col):
    for y in range(num_row):
      entry = level_array[y][x]
      if entry != ENCODING["WALL"]:
        loc_id = f"loc-{x}-{y}"
        obj = up.model.Object(loc_id, loc)
        loc_objs[loc_id] = obj
        problem.add_object(obj)
        if entry == ENCODING["ROBOT"]:
          problem.set_initial_value(has_player(obj), True)
        elif entry == ENCODING["BLOCK"]:
          problem.set_initial_value(has_box(obj), True)
        elif entry == ENCODING["GOAL"]:
          problem.add_goal(has_box(obj))

  for obj1 in loc_objs:
    for obj2 in loc_objs:
      x1, y1 = map(int, str(obj1).split("-")[1:])
      x2, y2 = map(int, str(obj2).split("-")[1:])
      dist = abs(x1 - x2) + abs(y1 - y2)
      if dist == 1:
        problem.set_initial_value(adjacent(loc_objs[obj1], loc_objs[obj2]), True)
      elif dist == 2 and (x1 == x2 or y1 == y2):
        problem.set_initial_value(adjacent_2(loc_objs[obj1], loc_objs[obj2]), True)
      elif abs(x1 - x2) == 1 and abs(y1 - y2) == 1:
        corner_id = f"loc-{x2}-{y1}"
        if corner_id in loc_objs:
          problem.set_initial_value(adjacent_diag(loc_objs[obj1], loc_objs[obj2]), True)
      elif abs(x1 - x2) == 2 and abs(y1 - y2) == 2 and with_diagonal_push:
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        w1, w2 = f"loc-{mid_x}-{y1}", f"loc-{x2}-{mid_y}"
        if w1 in loc_objs and w2 in loc_objs:
          problem.set_initial_value(adjacent_2_diag(loc_objs[obj1], loc_objs[obj2]), True)

  return problem


def create_free_diagonal_sokoban_problem(
  level: str,
  with_axioms: bool = False,
  with_diagonal_push: bool = True,
):
  """Cardinal + free diagonal moves; corner cells don't need to be free."""
  problem = up.model.Problem("sokoban")
  loc = up.shortcuts.UserType("location")

  has_player = up.model.Fluent("has_player", BoolType(), l=loc)
  has_box = up.model.Fluent("has_box", BoolType(), l=loc)
  adjacent = up.model.Fluent("adjacent", BoolType(), l1=loc, l2=loc)
  adjacent_2 = up.model.Fluent("adjacent_2", BoolType(), l1=loc, l2=loc)
  adjacent_diag = up.model.Fluent("adjacent_diag", BoolType(), l1=loc, l2=loc)

  for fl in [has_player, has_box, adjacent, adjacent_2, adjacent_diag]:
    problem.add_fluent(fl, default_initial_value=False)

  if with_diagonal_push:
    adjacent_2_diag = up.model.Fluent("adjacent_2_diag", BoolType(), l1=loc, l2=loc)
    problem.add_fluent(adjacent_2_diag, default_initial_value=False)

  cost_dict = {}

  move = up.model.InstantaneousAction("move", fr=loc, to=loc)
  fr, to = move.parameters
  move.add_precondition(adjacent(fr, to))
  move.add_precondition(has_player(fr))
  move.add_precondition(Not(has_box(to)))
  move.add_effect(has_player(fr), False)
  move.add_effect(has_player(to), True)
  cost_dict[move] = 0
  problem.add_action(move)

  move_diag = up.model.InstantaneousAction("move-diag", fr=loc, to=loc)
  fr, to = move_diag.parameters
  move_diag.add_precondition(adjacent_diag(fr, to))
  move_diag.add_precondition(has_player(fr))
  move_diag.add_precondition(Not(has_box(to)))
  move_diag.add_effect(has_player(fr), False)
  move_diag.add_effect(has_player(to), True)
  cost_dict[move_diag] = 0
  problem.add_action(move_diag)

  push_box = up.model.InstantaneousAction("push-box", x=loc, y=loc, z=loc)
  x, y, z = push_box.parameters
  push_box.add_precondition(adjacent(x, y))
  push_box.add_precondition(adjacent(y, z))
  push_box.add_precondition(adjacent_2(x, z))
  push_box.add_precondition(has_player(x))
  push_box.add_precondition(has_box(y))
  push_box.add_precondition(Not(has_box(z)))
  push_box.add_effect(has_player(x), False)
  push_box.add_effect(has_player(y), True)
  push_box.add_effect(has_box(y), False)
  push_box.add_effect(has_box(z), True)
  cost_dict[push_box] = 1
  problem.add_action(push_box)

  if with_diagonal_push:
    push_box_diag = up.model.InstantaneousAction("push-box-diag", x=loc, y=loc, z=loc)
    x, y, z = push_box_diag.parameters
    push_box_diag.add_precondition(adjacent_diag(x, y))
    push_box_diag.add_precondition(adjacent_diag(y, z))
    push_box_diag.add_precondition(adjacent_2_diag(x, z))
    push_box_diag.add_precondition(has_player(x))
    push_box_diag.add_precondition(has_box(y))
    push_box_diag.add_precondition(Not(has_box(z)))
    push_box_diag.add_effect(has_player(x), False)
    push_box_diag.add_effect(has_player(y), True)
    push_box_diag.add_effect(has_box(y), False)
    push_box_diag.add_effect(has_box(z), True)
    cost_dict[push_box_diag] = 1
    problem.add_action(push_box_diag)

  problem.add_quality_metric(up.model.metrics.MinimizeActionCosts(cost_dict))

  level_array = [[ch for ch in line] for line in level.splitlines()]
  num_col = len(level_array[0])
  num_row = len(level_array)
  loc_objs: dict[str, up.model.Object] = {}

  for x in range(num_col):
    for y in range(num_row):
      entry = level_array[y][x]
      if entry != ENCODING["WALL"]:
        loc_id = f"loc-{x}-{y}"
        obj = up.model.Object(loc_id, loc)
        loc_objs[loc_id] = obj
        problem.add_object(obj)
        if entry == ENCODING["ROBOT"]:
          problem.set_initial_value(has_player(obj), True)
        elif entry == ENCODING["BLOCK"]:
          problem.set_initial_value(has_box(obj), True)
        elif entry == ENCODING["GOAL"]:
          problem.add_goal(has_box(obj))

  for obj1 in loc_objs:
    for obj2 in loc_objs:
      x1, y1 = map(int, str(obj1).split("-")[1:])
      x2, y2 = map(int, str(obj2).split("-")[1:])
      dist = abs(x1 - x2) + abs(y1 - y2)
      if dist == 1:
        problem.set_initial_value(adjacent(loc_objs[obj1], loc_objs[obj2]), True)
      elif dist == 2 and (x1 == x2 or y1 == y2):
        problem.set_initial_value(adjacent_2(loc_objs[obj1], loc_objs[obj2]), True)
      elif abs(x1 - x2) == 1 and abs(y1 - y2) == 1:
        # Free diagonal: no corner-free check
        problem.set_initial_value(adjacent_diag(loc_objs[obj1], loc_objs[obj2]), True)
      elif abs(x1 - x2) == 2 and abs(y1 - y2) == 2 and with_diagonal_push:
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        w1, w2 = f"loc-{mid_x}-{y1}", f"loc-{x2}-{mid_y}"
        if w1 in loc_objs and w2 in loc_objs:
          problem.set_initial_value(adjacent_2_diag(loc_objs[obj1], loc_objs[obj2]), True)

  return problem


# ── Solver ────────────────────────────────────────────────────────────────────


def solve_sokoban_problem(problem):
  """Solve and return a SequentialPlan, or None if unsolvable."""
  up.shortcuts.get_environment().credits_stream = None
  with OneshotPlanner(problem_kind=problem.kind) as planner:
    result = planner.solve(problem)
    if result.status in unified_planning.engines.results.POSITIVE_OUTCOMES:
      return result.plan
    return None


# ── Plan conversion ───────────────────────────────────────────────────────────

# (dx, dy) = (col_delta, row_delta) in maze-branch convention (x=col, y=row).
_DELTA_TO_DIRECTION: dict[tuple[int, int], str] = {
  (0, -1): "up",
  (0, 1): "down",
  (-1, 0): "left",
  (1, 0): "right",
  (1, 1): "right_down",
  (-1, 1): "left_down",
  (1, -1): "right_up",
  (-1, -1): "left_up",
}


def parse_plan_action(action_string: str) -> Tuple[str, List[str]]:
  match = re.match(r"([\w-]+)\((.*)\)", action_string.strip())
  if not match:
    return None, None
  action_name = match.group(1)
  params = [p.strip() for p in match.group(2).split(",")]
  return action_name, params


def location_to_coords(loc_id: str) -> Tuple[int, int]:
  """Convert 'loc-x-y' → (x, y) = (col, row)."""
  parts = str(loc_id).split("-")
  return int(parts[1]), int(parts[2])


def plan_to_simple_steps(plan_actions: List) -> List[str]:
  """Convert UP plan actions to direction-label strings.

  Returns labels like "move_up", "push_right", "move_right_down", etc.
  Delta convention: (dx, dy) = (col_delta, row_delta).
  """
  steps: List[str] = []
  for action in plan_actions:
    action_name, params = parse_plan_action(str(action))

    if action_name in ("move", "move-diag"):
      fr_coords = location_to_coords(params[0])
      to_coords = location_to_coords(params[1])
      delta = (to_coords[0] - fr_coords[0], to_coords[1] - fr_coords[1])
      if delta not in _DELTA_TO_DIRECTION:
        raise ValueError(f"Unexpected move delta {delta} in {action}")
      steps.append(f"move_{_DELTA_TO_DIRECTION[delta]}")

    elif action_name in ("push-box", "push-box-diag"):
      if len(params) == 3:
        box_coords = location_to_coords(params[1])
        new_box_coords = location_to_coords(params[2])
      elif len(params) == 4:
        box_coords = location_to_coords(params[2])
        new_box_coords = location_to_coords(params[3])
      else:
        raise ValueError(f"Unexpected push-box param count {len(params)} in {action}")
      delta = (new_box_coords[0] - box_coords[0], new_box_coords[1] - box_coords[1])
      if delta not in _DELTA_TO_DIRECTION:
        raise ValueError(f"Unexpected push delta {delta} in {action}")
      steps.append(f"push_{_DELTA_TO_DIRECTION[delta]}")

    else:
      raise ValueError(f"Unknown action: {action_name}")

  return steps
