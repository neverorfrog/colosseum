# Colosseum Internals

Colosseum extends mjlab's `ManagerBasedRlEnv` with three optional managers
that are not part of mjlab itself. All of them live in `src/colosseum/managers/`
and are activated automatically by `ColosseumEnv` based on the task config.

---

## ColosseumEnv

`ColosseumEnv` is the single environment class used by every colosseum task.
It conditionally loads managers depending on which config sections are non-empty.

```python
@dataclass(kw_only=True)
class ColosseumEnvCfg(ManagerBasedRlEnvCfg):
    encoders:     dict[str, RmaTermCfg]         = field(default_factory=dict)
    abstractions: dict[str, AbstractionTermCfg] = field(default_factory=dict)
    constraints:  dict[str, ConstraintTermCfg]  = field(default_factory=dict)
    use_depth_camera: bool = False
```

A manager is only instantiated when its dict is non-empty:

| Config field | Manager created | When to use |
|---|---|---|
| `encoders` | `RmaManager` | Phase 1 / Phase 2 RMA training |
| `abstractions` | `AbstractionManager` | Planning-guided guidance signals (Soccer Maze) |
| `constraints` | `ConstraintManager` | CaT soft constraints (experimental) |

### Step order

Every `env.step()` runs in this order:

```
1.  ManagerBasedRlEnv.step()      ← physics + rewards + default managers (mjlab)
2.  rma_manager.update()          ← encoder forward pass (if present)
3.  constraint_manager.compute()  ← scale rewards, soft terminations (if present)
4.  abstraction_manager.compute() ← update guidance signals for NEXT step (if present)
```

### Reset order

```
1.  Snapshot episode lengths       ← before super() zeros them
2.  ManagerBasedRlEnv._reset_idx() ← standard mjlab reset
3.  abstraction_manager.reset()
4.  constraint_manager.reset()     ← receives pre-reset episode lengths
5.  rma_manager.reset()
```

---

## RmaManager

`RmaManager` implements the two-phase
[Rapid Motor Adaptation](https://arxiv.org/abs/2107.04034) training paradigm.

### Concept

The actor network takes as input `[proprioceptive_obs, latent_z]`. The latent
`z` is produced by one of two encoders depending on the training phase:

- **Phase 1** — a **privileged encoder** (small MLP) reads ground-truth
  observations that are not available at deployment (ball position, obstacle
  states). The policy and encoder train jointly via PPO.
- **Phase 2** — an **adaptation encoder** (CNN + GRU for depth) reads only
  sensor observations. Its output is supervised to match the Phase 1 latent
  via MSE regression. The policy and privileged encoder are frozen.

### Term interface

```python
@dataclass(kw_only=True)
class RmaTermCfg(ABC):
    privileged_obs_group: str        # GT obs group (always present)
    adaptation_obs_group: str | None # Sensor obs group (Phase 2 only)
    latent_dim: int = 8
    latent_noise_std: float = 0.0    # Phase 1 noise for Phase 2 robustness

    @abstractmethod
    def build(self, env) -> RmaTerm: ...
```

Each `RmaTerm` owns both encoders and is responsible for:

- `encode_privileged(obs_dict)` — GT obs → latent (Phase 1 forward pass)
- `encode_adaptation(obs_dict)` — sensor obs → latent (Phase 2 forward pass)
- `update()` — called every env step by `RmaManager`
- `reset(env_ids)` — reset recurrent state (e.g. GRU hidden) for terminated envs

### Wiring into the algorithm

`RmaPPO` (in `src/colosseum/algorithm/rma_ppo.py`) reads the manager to:

1. Widen the actor input by `rma_manager.total_latent_dim`.
2. Add encoder parameters to the Phase 1 optimizer.
3. Switch to the adaptation encoder for Phase 2 inference.

The dribbling task uses one term (`DribblingRmaTermCfg`) with a 64-dimensional
latent encoding ball position/velocity and obstacle state. See
[Dribbling](../research/dribbling.md) for details.

### Configuration example

```python
encoders = {
    "dribbling": DribblingRmaTermCfg(
        privileged_obs_group="privileged_ball",
        obstacle_privileged_obs_group="privileged_obstacles",
        adaptation_obs_group="depth_frames",  # None during Phase 1
        latent_dim=64,
        latent_noise_std=0.1,
    ),
}
```

---

## AbstractionManager

`AbstractionManager` provides symbolic guidance signals derived from a
higher-level planner. It is used by the Soccer Maze task to supply a
direction vector from the current robot position to the next waypoint in a
discrete grid plan.

### Concept

The manager holds a collection of `AbstractionTerm` instances. Each term
maintains a cache of guidance signals (e.g. cost maps, direction maps) and
rebuilds them lazily when the underlying settings change (e.g. goal cell or
maze layout changes).

The term lifecycle follows a change-detection pattern:

```
reset(env_ids)
  └─ _update_settings(env_ids)   ← read goal/map from env
  └─ _maybe_rebuild(env_ids)     ← rebuild cost/direction map if settings changed

compute(dt)
  └─ _update_signals(all_envs)   ← read current robot pos, look up direction
```

### Term interface

```python
@dataclass(kw_only=True)
class AbstractionTermCfg(ABC):
    debug_vis: bool = False

    @abstractmethod
    def build(self, env) -> AbstractionTerm: ...
```

Concrete subclasses expose their signals as methods (e.g.
`term.direction_map`, `term.cost_map`). Observation functions retrieve the
term from the manager and call those methods directly:

```python
def maze_direction(env, abstraction_name: str) -> torch.Tensor:
    term = env.abstraction_manager.get_term(abstraction_name)
    return term.direction_at_pos(env.scene["robot"].data.root_pos_w[:, :2])
```

### AbstractionSettings and change detection

`AbstractionSettings` is an immutable frozen dataclass that snapshots the
inputs that drive a rebuild. Subclasses add exactly the fields that, when
changed, should invalidate the cached plan:

```python
@dataclass(frozen=True, eq=False)
class GridAbstractionSettings(AbstractionSettings):
    goal_cell: tuple[int, int]
    map: torch.Tensor   # object identity used for equality (no tensor copy)
```

This avoids redundant replanning across the thousands of parallel envs that
share the same goal and map.

### The GridAbstractionTerm

The concrete implementation used by Soccer Maze is `GridAbstractionTerm`
(in `src/colosseum/mdp/abstraction/maze/grid_abstraction.py`). It computes:

- A **cost map** via Dijkstra / harmonic potential over the discrete grid.
- A **direction map** pointing each cell toward the goal.

Both are cached per unique `(goal_cell, map)` pair and reused across envs
with the same goal.

---

## Adding a new manager

All three managers follow the same term-based pattern as mjlab's built-in
managers. To add your own:

1. Subclass `AbstractionTermCfg` (or `RmaTermCfg`) and implement `build()`.
2. Implement the corresponding `AbstractionTerm` (or `RmaTerm`) subclass.
3. Add the term to `ColosseumEnvCfg.abstractions` (or `.encoders`) in your
   task config.

`ColosseumEnv` will instantiate the manager automatically.
