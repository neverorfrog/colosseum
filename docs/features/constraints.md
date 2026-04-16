# Constraints as Terminations (CaT)

Implementation of the algorithm from:
> *"Constraints as Terminations: Reinforcement Learning with Soft Constraints"*
> Pardo et al., 2024 — [arXiv:2403.18765](https://arxiv.org/abs/2403.18765)

## Core Idea

Instead of using Lagrangian penalties to enforce safety constraints, CaT converts
constraint violations into stochastic episode terminations. At each step, a
per-env termination probability `δ ∈ [0, 1]` is derived from the violation
magnitude and used to both scale the reward and probabilistically terminate the
episode.

This removes the need for constrained optimization while providing a dense
learning signal: the agent is incentivized to avoid violations because they
collapse future returns.

---

## Algorithm

### Step 1 — Constraint evaluation

Each constraint function returns a raw violation tensor:

```
c_i(s, a)  ∈ R^{num_envs × num_dims}
```

Values > 0 indicate violation magnitude; values ≤ 0 are safe.

### Step 2 — Polyak-averaged running maximum (Eq. 7)

For each constraint `i`, a per-dimension running maximum is maintained:

```
c_i^max  ←  τ · c_i^max  +  (1 - τ) · max_{batch}(c_i^+)
```

where `τ = 0.95` and `c_i^+ = clamp(c_i, 0, ∞)`. This normalizes violation
magnitudes into `[0, 1]` in a scale-invariant way across training.

### Step 3 — Termination probability (Eq. 6)

```
p_i  =  p_i^max · clip(c_i^+ / c_i^max, 0, 1)

δ    =  max_i  p_i
```

`p_i^max` controls constraint severity:

| `p_i^max` | Meaning |
|---|---|
| `1.0` | Hard constraint — full termination on any violation |
| `0.25` | Soft constraint — at most 25% termination chance per step |
| `0.0` | Disabled |

### Step 4 — Reward scaling and done signal

```
reward  ←  clip(reward · (1 - δ),  min=0)
done    ←  f(δ)   # see Option A / B below
```

The effective per-step discount becomes `γ · (1 - δ)` instead of `γ`,
so a violation at time `t` collapses all future returns from that point.

---

## Implementation Options

Two options exist for translating the float probability `δ` into the done signal
consumed by the PPO trainer. Option A is currently active.

### Option A — Bernoulli Sampling (current)

Sample a binary termination from the probability at each step:

```python
cstr_terminated = torch.bernoulli(cstr_prob).bool()
terminated = terminated | cstr_terminated
```

`terminated` stays `bool` so the PPO interface `(obs, rewards, terminated: bool, truncated: bool, extras)`
is unchanged. In expectation the GAE is identical to Option B. Variance is
negligible at 4096+ parallel environments.

#### Files created

| File | Contents |
|---|---|
| `src/colosseum/managers/constraint_manager.py` | `ConstraintTermCfg` dataclass; `CaT` core algorithm class; `ConstraintManager` mjlab manager |
| `src/colosseum/mdp/constraints.py` | Constraint function library (11 functions) |
| `src/colosseum/envs/constraint_based_env.py` | `ConstraintBasedEnvCfg` and `ConstraintBasedEnv` |

#### Files modified

| File | Change |
|---|---|
| `src/colosseum/managers/__init__.py` | Added exports: `CaT`, `ConstraintManager`, `ConstraintTermCfg` |
| `src/colosseum/envs/__init__.py` | Added exports: `ConstraintBasedEnv`, `ConstraintBasedEnvCfg` |
| `src/colosseum/mdp/__init__.py` | Added export: `constraints` module |

No changes to `src/colosseum/algorithm/ppo.py` or the rollout buffer.

---

### Option B — Float Dones (not yet implemented)

The key conceptual difference from Option A: soft violations (0 < δ < 1) do
**not** reset the environment. The episode continues; only the GAE discount is
softened. Only δ = 1 (hard constraint or timeout) triggers an actual reset.
This is the faithful implementation of the paper's training objective.

#### Files to create

Same three new files as Option A, with one change in `constraint_based_env.py`:

**`src/colosseum/envs/constraint_based_env.py` — `step()` method:**
replace the Bernoulli sampling block with:

```python
# Return float terminated instead of sampling a bool
terminated_float = cstr_prob.clone()
terminated_float[terminated] = 1.0   # hard resets stay at 1.0
return obs, rewards, terminated_float, truncated, extras
```

#### Files to modify

| File | Location | Change |
|---|---|---|
| `src/colosseum/managers/__init__.py` | — | Same exports as Option A |
| `src/colosseum/envs/__init__.py` | — | Same exports as Option A |
| `src/colosseum/mdp/__init__.py` | — | Same exports as Option A |
| `src/colosseum/algorithm/ppo.py` | line 280 | Detect float `terminated` and compute dones without bitwise OR |
| `src/colosseum/algorithm/ppo.py` | line 306 | Use `(dones >= 1.0)` for episode tracking, not `dones.nonzero()` |
| `src/colosseum/algorithm/ppo.py` | line 316 | Cast float `terminated` to bool before `update_episode_counts` |

**`src/colosseum/algorithm/ppo.py` — exact diffs:**

```python
# line 280 — replace:
#   dones = (terminated | truncated).float()
# with:
if terminated.is_floating_point():
    dones = torch.clamp(terminated + truncated.float(), 0.0, 1.0)
else:
    dones = (terminated | truncated).float()

# line 306 — replace:
#   done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
# with:
episode_done_ids = (dones >= 1.0).nonzero(as_tuple=False).squeeze(-1)

# lines 307–313 — replace done_ids with episode_done_ids throughout:
if len(episode_done_ids) > 0:
    self.rewbuffer.extend(self.cur_reward_sum[episode_done_ids].cpu().numpy().tolist())
    self.cur_reward_sum[episode_done_ids] = 0.0
    self.episode_lengths.extend(
        self.episode_length_buf[episode_done_ids].cpu().numpy().tolist()
    )
    self.episode_length_buf[episode_done_ids] = 0

# line 316 — replace:
#   self.update_episode_counts(terminated, truncated)
# with:
hard_terminated = terminated >= 1.0 if terminated.is_floating_point() else terminated
self.update_episode_counts(hard_terminated, truncated)
```

`src/colosseum/algorithm/rollout_buffer.py` — **no change needed**.
`compute_returns_and_advantages` already uses `1.0 - dones` as a float mask,
which handles values in `[0, 1]` correctly.

---

## Usage

### Define constraint terms

```python
from mjlab.managers.scene_entity_config import SceneEntityCfg
from colosseum.managers import ConstraintTermCfg
from colosseum.mdp import constraints

@dataclass(kw_only=True)
class MyEnvCfg(ConstraintBasedEnvCfg):
    constraints: dict = field(default_factory=lambda: {
        # Hard: full termination on upside-down
        "upsidedown": ConstraintTermCfg(
            func=constraints.upsidedown,
            max_p=1.0,
            params={"limit": 0.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Soft: 25% termination chance when torque exceeds 40 Nm
        "joint_torque": ConstraintTermCfg(
            func=constraints.joint_torque,
            max_p=0.25,
            params={"limit": 40.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Soft: encourage upright base
        "base_orientation": ConstraintTermCfg(
            func=constraints.base_orientation,
            max_p=0.25,
            params={"limit": 0.5, "asset_cfg": SceneEntityCfg("robot")},
        ),
    })
```

### Available constraint functions (`colosseum.mdp.constraints`)

| Function | Violation condition | Returns |
|---|---|---|
| `joint_torque` | `\|torque\| > limit` | `[N, num_joints]` |
| `joint_velocity` | `\|vel\| > limit` | `[N, num_joints]` |
| `joint_acceleration` | `\|acc\| > limit` | `[N, num_joints]` |
| `joint_position` | `\|pos\| > limit` | `[N, num_joints]` |
| `joint_range` | `\|pos - default\| > limit` | `[N, num_joints]` |
| `base_orientation` | horizontal gravity component `> limit` | `[N]` |
| `upsidedown` | gravity z-component `> limit` | `[N]` |
| `min_base_height` | base height `< limit` | `[N]` |
| `contact` | any body contact force `> 1 N` | `[N]` |
| `foot_contact_force` | peak foot contact force `> limit` | `[N, num_feet]` |
| `air_time` | foot air time `> limit` when moving | `[N, num_feet]` |

### Logged metrics

When environments reset, `ConstraintManager.reset()` adds to `extras["log"]`:

- `Episode_Constraint_violation/<name>` — % of steps where the constraint was violated
- `Episode_Constraint_probability/<name>` — mean termination probability per step

---

## Design Notes

- `ConstraintManager` follows the same `ManagerBase` pattern as other mjlab
  managers. It is initialized in `load_managers()` and its `reset()` is called
  inside `_reset_idx()` so episode statistics are logged at the right time.
- `CaT` is a pure-torch class with no mjlab dependencies and can be unit-tested
  independently.
- Setting `constraints={}` in the config is a no-op: the manager is not
  instantiated and `step()` behaves identically to the base env.
