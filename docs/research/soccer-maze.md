# HumanoidSoccerMaze

> A novel benchmark requiring a humanoid robot to navigate a maze while
> dribbling a ball to a distant target, with a baseline solution that
> integrates planning and learned locomotion.

---

## What this is

HumanoidSoccerMaze tests a Booster T1 humanoid on goal-conditioned maze
navigation combined with ball dribbling. The robot must find a path through
a maze with walls and obstacles, and execute it using a learned
locomotion-dribbling policy.

The benchmark is built on [mjlab](https://github.com/mujocolab/mjlab)
(GPU-accelerated RL via MuJoCo Warp) and uses the
[Booster T1](https://www.boostspace.com/) 23-DOF humanoid — the platform used
in the [RoboCup Humanoid Soccer League](https://www.robocup.org/leagues/35).

An example of evaluation after approximately 200 million steps of training on a RTX 4090 in approximately 3 hours:

<iframe
  src="https://drive.google.com/file/d/1CgFFBQNQtTevHFo_cPRcrJF6HEFSO9g2/preview"
  width="720" height="405"
  allow="autoplay"
  style="border:none; border-radius:4px;"
></iframe>

---

## The planning–learning interface

The core challenge is that a low-level dribbling policy cannot reason about
walls several meters away, while a planner cannot control 23 joints to keep
the ball moving smoothly. The solution couples them through a shared
**direction signal**.

```
┌─────────────────┐          direction vector         ┌──────────────────────┐
│    Planner      │  ─────────────────────────────>   │  Dribbling Policy    │
│  (discrete,     │                                   │  (continuous,        │
│   global)       │  <─────────────────────────────   │   local)             │
│                 │     current robot/ball position   │                      │
└─────────────────┘                                   └──────────────────────┘
```

**Planner side** — a classical planner solves the maze using a Sokoban
abstraction: it models the ball-dribbling problem as a discrete push-planning
problem on a grid. Given the current ball cell and the goal cell, it computes
a feasible path through free cells considering the Sokoban constraints (
the ball can only be pushed, not pulled). The planner outputs a direction
vector pointing from the robot's current cell toward the next cell in the plan.

**Learning side** — the dribbling policy (trained as in the dribbling task)
observes a local direction vector pointing toward the next waypoint in the
plan. It uses this to steer its dribbling (or just walking).

**AbstractionManager bridge** — at each environment step, the
`GridAbstractionTerm` re-reads the robot position and ball position, maps them
to grid cells, looks up the pre-computed cost map, and provides a
`direction_at_pos()` signal to observation functions. The planner only
replans when the goal cell or map layout changes (change-detection via
`AbstractionSettings`), avoiding expensive replanning every step.

```
env step:
  abstraction_manager.compute(dt)
    └── GridAbstractionTerm._update_signals()
          └── look up direction_map at current robot cell
                → feeds "maze_direction" observation term
```

The maze layout and goal are provided as part of the environment config and
can be randomised across episodes for generalisation.

---

## Task structure

The environment extends the standard dribbling task:

- **Same robot and ball setup** as `t1-dribbling`.
- **Additional scene elements**: maze walls encoded as a binary occupancy grid.
- **Additional observation term**: `maze_direction` — a 2D unit vector in robot
  body frame pointing toward the next grid waypoint.
- **Same reward structure**: ball velocity tracking + locomotion regularization.
  The planner direction acts as a soft constraint through the observation, not
  as a separate reward term.

---

## Code structure

```
src/colosseum/tasks/soccer_maze/     # Benchmark environment (scene, MDP, rewards)

src/colosseum/mdp/abstraction/maze/
└── grid_abstraction.py              # GridAbstractionTerm — planner + signal cache
```

---

## Quick start

```bash
# Train
pixi run train task:t1-soccer-maze

# Evaluate
pixi run play task:t1-soccer-maze --checkpoint ./logs/<run>/checkpoints/latest.pt
```

---

## Paper

The paper describes the technical contribution of the benchmark (environment features, performance metrics) and the baseline solution approach. It is under submission at **RoboCup Symposium 2026**.

> **HumanoidSoccerMaze: a novel benchmark and a hierarchical solution approach**  
> Flavio Maiorana, Daniel Gigliotti, Luca Iocchi  
> *RoboCup Symposium 2026 (under submission)*
