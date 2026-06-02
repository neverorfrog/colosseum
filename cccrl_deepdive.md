# CCRL Implementation Deep Dive

This document traces the complete Cascaded Compositional Residual Learning pipeline from
[Kumar, Essa, Ha (2022)](https://arxiv.org/abs/2212.08954), as implemented in
`external/rsl_rl` with example configs from `external/legged_gym`.

We follow the execution flow top-down: config → runner → algorithm → actor-critic module.

## What Problem Does CCRL Solve?

Suppose you have a policy that already walks well (trained with PPO). Now you want the robot
to push an object toward a target while walking. Training from scratch is slow, and
fine-tuning the locomotion policy risks losing the walking style.

**CCRL's approach:** freeze the walking policy. Add a residual policy that learns corrective
actions for object interaction, plus a weight network that blends the two at each timestep.
The robot walks normally by default and only engages the residual when the task reward
justifies it.

## The Three Mechanisms

| Mechanism | What it learns | Trainable? |
|---|---|---|
| Skill Library | Frozen locomotion policy | No (frozen) |
| Residual Policy | Corrective push/interaction actions | Yes |
| Weight Network | When to use each skill (gating) | Yes |

Note: the paper's Goal Network (MetaBackbone for synthetic observations) lives in
`MultiSkillActorCriticv3`. It is only needed for complex hierarchical compositions (e.g.,
indoor navigation + door opening). For walk+push with 2 skills, we use the simpler
`ResidualActorCritic`.

---

## Full Architecture End-to-End

```
┌─────────────────────────────────────────────────────────────────────┐
│ CONFIG (a1_config.py)                                               │
│   obs_sizes = {"base_lin_vel": 3, ..., "object_position": 2}        │
│   actor_obs = [["base_lin_vel", ...], ["base_lin_vel", ..., "obj"]] │
│   actor_hidden_dims = [[512,256,128], [512,256,128]]                │
│   skill_paths = [".../locomotion.pt"]                               │
│   residual_action_penalty_coef = 0.01                               │
│   residual_weight_penalty_coef = 0.01                               │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ResidualSkillOnPolicyRunner.__init__()                               │
│   1. ResidualActorCritic(2, obs_sizes, actor_obs, ...)               │
│      ├── actor[0] = Policy([512,256,128], 45, 12)                   │
│      ├── actor[1] = Policy([512,256,128], 49, 12)                   │
│      ├── critic = MLP(49 → 1)                                       │
│      ├── weights = Weights(49 → softmax(2))                         │
│      └── obs_indices: [tensor(0:45), tensor(0:49)]                  │
│   2. ResidualPPO(actor_critic, penalty_coefs=0.01)                   │
│      └── optimizer = Adam(actor[1] + critic + weights)               │
│   3. load_skills([.../locomotion.pt])                                │
│      ├── Load checkpoint → actor[0].load_state_dict(strict=True)     │
│      ├── Freeze actor[0] (requires_grad=False)                       │
│      └── weight_init bias = [0.8, 0.01]                             │
└───────────────────────┬──────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Training Loop (runner.learn)                                         │
│                                                                      │
│  ROLLOUT: for each step:                                             │
│    action = alg.act(obs, critic_obs)                                 │
│      → actor_critic.act(obs)                                         │
│        → update_distribution(obs)                                    │
│          → actor[0](obs[:, 0:45]) → (μ₀, σ₀)    [FROZEN]           │
│          → actor[1](obs[:, 0:49]) → (μ₁, σ₁)    [TRAINABLE]         │
│          → weights(obs[:, 0:49])  → [w₀, w₁]    [TRAINABLE]         │
│          → residual_action_magnitude = ‖μ₁‖₂                          │
│          → residual_weights_ = |w₁|                                   │
│          → combine_skills → Normal(μ_comb, σ_comb)                    │
│          → sample(μ_comb, σ_comb) → action                            │
│    env.step(action) → obs', reward                                    │
│                                                                      │
│  UPDATE: alg.update()                                                │
│    for each mini-batch:                                              │
│      actor_critic.act(obs_batch)     # re-eval with current params   │
│      PPO surrogate loss + value loss + entropy                       │
│      + residual_action_magnitude * 0.01   ← penalizes large μ₁       │
│      + residual_weights_ * 0.01          ← penalizes high w₁         │
│      → backprop through actor[1], critic, weights only               │
│    return value_loss, surrogate_loss, residual_weight, action_mag    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 1. OnPolicyRunner (Base)

File: `external/rsl_rl/rsl_rl/runners/on_policy_runner.py`

The upstream base runner. Key characteristics:

### `__init__()` (line 46)

```python
actor_critic_class = eval(self.cfg["policy_class_name"])  # "ActorCritic"
actor_critic = actor_critic_class(self.env.num_obs,
                                  num_critic_obs,
                                  self.env.num_actions,
                                  **self.policy_cfg)
alg_class = eval(self.cfg["algorithm_class_name"])  # "PPO"
self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, ...)
```

- Dynamic instantiation via `eval()` of class name strings → allows swapping actor-critic/algo classes from config.
- Observation dimensions come from `self.env.num_obs` (flat), `self.env.num_privileged_obs` (privileged), and `self.env.num_actions`.
- Storage is initialized with observation shape lists (wrapped in singleton lists for the flat-obs case).

### `learn()` (line 107)

The standard RSL-RL training loop:

```
for each iteration:
    1. ROLLOUT (inference_mode):
       for each step in num_steps_per_env:
           actions = self.alg.act(obs, critic_obs)
           obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
           self.alg.process_env_step(rewards, dones, infos)
           # bookkeeping: cur_reward_sum, rewbuffer, lenbuffer
    2. compute_returns (bootstrap from last observation)
    3. self.alg.update()  → mean_value_loss, mean_surrogate_loss
    4. log + periodic save
```

Returns `mean_value_loss` and `mean_surrogate_loss` (2 values).

---

## 2. ResidualSkillOnPolicyRunner

File: `external/rsl_rl/rsl_rl/runners/on_policy_runner_residualskill.py` (163 lines)

Extends `OnPolicyRunner`. Inheritance:

```
OnPolicyRunner                     # rsl_rl base: env loop + learn()
  └── ResidualSkillOnPolicyRunner  # adds: load_skills(), ResidualActorCritic, ResidualPPO
```

### What's Different from the Base Runner

#### A: Instantiates `ResidualActorCritic` instead of `ActorCritic` (lines 63-68)

```python
actor_critic = actor_critic_class(
    2,                           # num_skills: HARDCODED to 2 (base + residual)
    self.cfg["obs_sizes"],       # dict: observation segment → dimension
    self.cfg["actor_obs"],       # list[list[str]]: which segments each skill sees
    self.cfg["critic_obs"],      # list[list[str]]: which segments critic sees
    self.env.num_actions,
    **self.policy_cfg
)
```

Instead of one network that reads `[num_obs, num_critic_obs, num_actions]`, we get a
multi-branch architecture that internally slices observations by column segments.

#### B: Instantiates `ResidualPPO` instead of `PPO` (line 71)

```python
alg_class = eval(self.cfg["algorithm_class_name"])  # "ResidualPPO"
```

Both class names come from config strings so the same runner could pair different
combinations. In practice, residual training always uses this pair.

#### C: Calls `load_skills()` before `env.reset()` (lines 84-85)

```python
self.load_skills(self.cfg["skill_paths"])
_, _ = self.env.reset()
```

### `load_skills()` — The Freezing Mechanism (lines 88-101)

Two key operations:

**Part A: Load and freeze pretrained skills** (lines 89-98):

```python
for i, path in enumerate(paths):
    loaded_dict = torch.load(path)
    model_state_dict = {}
    for name, params in loaded_dict["model_state_dict"].items():
        if "actor" in name:
            name_ = name[6:]          # strip "actor." prefix
            model_state_dict[name_] = params
    self.alg.actor_critic.actor[i].load_state_dict(model_state_dict, True)
    for parms in self.alg.actor_critic.actor[i].parameters():
        parms.requires_grad = False   # FROZEN
```

Step by step:
1. Load a standard PPO checkpoint (keys like `actor.mean_head.weight`, `actor.std`,
   `critic.0.weight`, etc.)
2. **Filter**: only extract keys starting with `"actor"` — the locomotion policy backbone
3. **Strip the prefix**: `"actor.mean_head.weight"` → `"mean_head.weight"` so it matches
   the `Policy` nn.Module inside `ResidualActorCritic.actor[0]`
4. **Strict load**: `load_state_dict(..., True)` ensures exact key match
5. **Freeze**: `requires_grad = False` prevents the optimizer from touching these params

Since `actor[0]` has `requires_grad=False`, Adam ignores it automatically when building
its parameter groups from `self.actor_critic.parameters()`.

**Part B: Initialize weight bias toward the frozen skill** (lines 99-100):

```python
weight_init = (len(paths) * [1/len(paths) - 0.2]
               + (len(self.alg.actor_critic.actor) - len(paths)) * [0.01])
# For 1 frozen skill + 1 residual: [1/1 - 0.2, 0.01] = [0.8, 0.01]
self.alg.actor_critic.weights.data = torch.tensor(weight_init, device=self.device)
```

For the typical walk+push case with 1 frozen skill and 1 residual:
- `actor[0]` (walk): weight bias = `1/1 - 0.2 = 0.8`
- `actor[1]` (residual/push): weight bias = `0.01`

The raw bias values pass through softmax, giving the locomotion branch ~99% initial weight.
The residual only activates when the reward signal justifies the penalty costs.

### `learn()` — Differences from the Base Runner (lines 104-163)

Nearly identical to `OnPolicyRunner.learn()`, with one key difference in return values
(line 153):

```python
# Base runner returns 2 values:
mean_value_loss, mean_surrogate_loss = self.alg.update()

# Residual runner returns 4 values:
mean_value_loss, mean_surrogate_loss, residual_weight, residual_action_magnitude = self.alg.update()
```

`ResidualPPO.update()` computes two additional diagnostics that are logged for monitoring.

Other minor differences:
- Does **not** save env config (`self.env.save_config()` missing — likely an oversight)
- Does **not** return `rewbuffer` from learn (unlike base runner, line 170)
- Uses `rewbuffer[new_ids][:, 0]` instead of `rewbuffer[new_ids]` for reward extraction

---

## 3. Config Wiring Example: `A1MultiSkillObjectPushCfgPPO`

File: `external/legged_gym/legged_gym/envs/a1/a1_config.py` (lines 1593-1668)

This is the canonical walk+push example. Each config section maps directly to a component:

### `algorithm` section (lines 1594-1602)

```python
class algorithm:
    entropy_coef = 0.01
    residual_weight_penalty_coef = 0.5 * 0.02   # = 0.01
    residual_action_penalty_coef = 0.5 * 0.02   # = 0.01
```

These go into `ResidualPPO.__init__(**self.alg_cfg)`. The penalty coefficients control how
much the residual branch is taxed:
- Higher `residual_action_penalty_coef` → residual corrections stay small → locomotion
  style preserved
- Higher `residual_weight_penalty_coef` → weight network prefers locomotion → residual
  only activates when strongly needed

### `policy` section (lines 1603-1608)

```python
class policy:
    init_noise_std = 1.0
    actor_hidden_dims = [[512, 256, 128], [512, 256, 128]]  # 2 skills, each 3-layer MLP
    critic_hidden_dims = [512, 256, 128]
    weight_network_dims = [512, 256]
    activation = 'elu'
```

`actor_hidden_dims` is a **list of lists** — one per skill branch. The first `[512,256,128]`
is for the frozen locomotion policy; the second is for the trainable residual policy.

### `runner` section (lines 1610-1668) — the critical wiring

```python
class runner:
    algorithm_class_name = 'ResidualPPO'        # eval'd to create the algo
    policy_class_name = 'ResidualActorCritic'   # eval'd to create the network

    obs_sizes = {
        "scaled_base_lin_vel": 3,   "scaled_base_ang_vel": 3,
        "projected_gravity": 3,     "target_position": 2,
        "object_position": 2,       "relative_dof": 12,
        "scaled_dof_vel": 12,       "actions": 12,
    }
    # Total = 3+3+3+2+2+12+12+12 = 49

    actor_obs = [
        # skill[0] (= locomotion): NO target/object position
        ["scaled_base_lin_vel", "scaled_base_ang_vel", "projected_gravity",
         "relative_dof", "scaled_dof_vel", "actions"],         # 3+3+3+12+12+12 = 45

        # skill[1] (= residual/push): INCLUDES target and object position
        ["scaled_base_lin_vel", "scaled_base_ang_vel", "projected_gravity",
         "target_position", "object_position",
         "relative_dof", "scaled_dof_vel", "actions"],         # 45 + 2 + 2 = 49
    ]

    critic_obs = [["scaled_base_lin_vel", "scaled_base_ang_vel",
                   "projected_gravity", "target_position",
                   "object_position", "relative_dof",
                   "scaled_dof_vel", "actions"]]               # full 49

    skill_paths = [".../model_1500.pt"]  # path to frozen locomotion checkpoint
```

Key design decisions:
- **Locomotion branch sees 45 dims** (no object info) — exactly what it was trained on
- **Residual branch sees 49 dims** — includes `target_position` and `object_position` so
  it can react to the object
- **Critic and weights see 49 dims** — full context for value estimation and gating

### Task registry wiring

`task_registry.make_residual_alg_runner()` (`task_registry.py:263`) creates the runner:

```python
train_cfg_dict = class_to_dict(train_cfg)
runner = ResidualSkillOnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
```

---

## 4. Reward System

### One Environment, One Combined Signal

The push task environment inherits from a base environment class (`PushingRobot` extends
`LeggedRobot`) and **adds** task rewards on top. There is only **one** `compute_reward()`
call per step, and all policies (frozen + residual) optimize against the same combined
scalar reward signal.

### Mechanism

`LeggedRobot.compute_reward()` (`legged_robot.py:194`) iterates every reward function
whose scale is non-zero in the config and sums them:

```python
self.rew_buf[:] = 0.
for i in range(len(self.reward_functions)):
    rew = self.reward_functions[i]() * self.reward_scales[name]
    self.rew_buf += rew
```

The reward functions are discovered dynamically via `_prepare_reward_function()` — it scans
`cfg.rewards.scales`, drops zero-scale entries, and maps each remaining key to
`self._reward_<NAME>()`. Python's MRO handles the rest: if a subclass defines
`_reward_object_target_dist()`, that method is called; if not, the base method is used.

### How Locomotion Tracking Rewards Are Killed

The environment naturally provides the locomotion observation (`base_lin_vel`, `dof_pos`,
etc.) plus any task-specific observations (`target_position`, `object_position`).
Locomotion tracking rewards are zeroed in the task config (`tracking_lin_vel = 0.0`,
`tracking_ang_vel = 0.0`) so that only task rewards drive learning:

```
Pretrained locomotion policy
    ├── Trained on tracking_lin_vel = 1.0, tracking_ang_vel = 0.3 (active velocity tracking)
    └── Frozen, continues to output "walk forward" actions because those weights are frozen

Residual policy
    ├── Trained on tracking_lin_vel = ~0, object_target_dist = 2.0 (only task reward matters)
    └── Learns to override locomotion when pushing the object yields reward
```

### Comparison: `A1FlatCfg` (locomotion) vs `A1TargetObjectPushCfg` (push task)

File: `external/legged_gym/legged_gym/envs/a1/a1_config.py`

| Reward term | A1FlatCfg | A1TargetObjectPushCfg | Change |
|---|---|---|---|
| `tracking_lin_vel` | **1.0** | 0.01 | nearly killed |
| `tracking_ang_vel` | **0.3** | 0.01 | nearly killed |
| `lin_vel_z` | -2.0 | -2.0 | same |
| `ang_vel_xy` | -0.05 | **0.0** | removed |
| `orientation` | -0.1 | **0.0** | removed |
| `base_height` | **0.0** | -0.5 | newly enabled |
| `feet_air_time` | 1.0 | 1.0 | same |
| `collision` | -1.0 | -1.0 | same |
| `action_rate` | -0.01 | -0.01 | same |
| `torques` | -0.0002 | **-0.0001** | halved |
| `dof_acc` | -2.5e-7 | -2.5e-7 | same |
| `dof_pos_limits` | -10.0 | -10.0 | same |
| `object_target_dist` | — | **2.0** | new task reward |

| Commands | A1FlatCfg | A1TargetObjectPushCfg |
|---|---|---|
| `lin_vel_x` | [0.5, 0.5] (walk forward) | **[0.0, 0.0]** (no velocity target) |

What this means:
- **Velocity tracking is nearly zeroed** — the frozen policy still outputs "walk forward at
  0.5 m/s" actions (its weights encode this), but the environment no longer rewards it
- **Base height penalty is introduced** (`-0.5`) to keep the robot upright during object
  interaction
- **Torque penalty is halved** to allow more aggressive actions near the puck
- **One new task reward**: `object_target_dist = 2.0` — reward for moving the puck toward
  the target
- **Regularization penalties stay the same** (`lin_vel_z`, `collision`, `action_rate`,
  `feet_air_time`, `dof_acc`, `dof_pos_limits`) — these encode fundamental physical
  constraints that apply universally regardless of task

### Why the Frozen Policy Still Works

The frozen locomotion policy was trained on a different reward structure (with
`tracking_lin_vel` active). During residual training, the environment gives it the **same
observation slice** it was trained on (proprioception only — no object info), so its
output is the same as what it would produce in a pure locomotion setting. The residual
learns the *difference*: when the puck needs pushing, the environment reward
(`object_target_dist`) outweighs the residual penalty, so the residual branch activates.

### The Push Task Deliberately Breaks the Locomotion Reward Structure

Velocity tracking (the primary signal the frozen policy optimized for) is nearly zeroed
out. The frozen policy still outputs "walk forward at 0.5 m/s" actions because that's what
its weights encode. But the environment no longer rewards that behavior.

Instead, the environment rewards `object_target_dist` — getting the puck near the target.
The only way to get that reward is through the residual branch, since the frozen policy
has no idea there's even a puck.

The residual branch has to learn to **override** the frozen policy's forward-walking
actions with puck-pushing actions. The weight network has to learn **when** (i.e., when
the puck is nearby and the object-target reward is achievable) and **how much** to let
the residual take control.

---

## 5. ResidualPPO Algorithm

File: `external/rsl_rl/rsl_rl/algorithms/residual_ppo.py` (209 lines)

A **minimal extension** of the upstream `PPO` class — only 2 things differ: the
constructor gets 2 extra parameters, and `update()` adds 2 penalty terms to the loss.
Everything else (`act`, `process_env_step`, `compute_returns`, `init_storage`) is
identical line-for-line to the base `PPO`.

### Constructor — the 2 new parameters (lines 55-56)

```python
residual_action_penalty_coef = 0.02
residual_weight_penalty_coef = 10.0
```

These flow from the config:
```
a1_config.py
  → class algorithm:
      residual_action_penalty_coef = 0.01
      residual_weight_penalty_coef = 0.01
  → train_cfg_dict = class_to_dict(train_cfg)
  → runner = ResidualSkillOnPolicyRunner(env, train_cfg_dict, ...)
  → self.alg_cfg = train_cfg["algorithm"]          # includes both coefs
  → ResidualPPO(actor_critic, device=..., **self.alg_cfg)  # expanded as kwargs
```

### `update()` — the only method that differs (lines 126-209)

Standard PPO loss:

```python
loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy
```

ResidualPPO adds **2 penalty terms** (lines 186-188):

```python
loss += self.actor_critic.residual_action_magnitude * self.residual_action_penalty_coef
loss += self.actor_critic.residual_weights_     * self.residual_weight_penalty_coef
```

And returns **4 values** instead of 2 (line 209):

```python
# PPO:
return mean_value_loss, mean_surrogate_loss

# ResidualPPO:
return mean_value_loss, mean_surrogate_loss, residual_weight, residual_action_magnitude
```

### The Hidden Coupling: Side-Effect State Passing

This is the trickiest part of the architecture. The penalty values don't come from the PPO
update itself — they're computed **during the rollout** and stored as instance attributes
of `ResidualActorCritic`, then read during the update. Here's the full chain:

```
ROLLOUT PHASE (runner.learn(), line 128):
  actions = self.alg.act(obs, critic_obs)
    └── ResidualPPO.act(obs, critic_obs)                   # residual_ppo.py:96
          ├── self.transition.actions = self.actor_critic.act(obs).detach()
          │     └── ResidualActorCritic.act(observations)  # residual_actor_critic.py:214
          │           └── self.update_distribution(observations)  # line 159
          │                 ├── skill_means, skill_std = [branch(obs_slice) for each branch]
          │                 ├── self.residual_action_magnitude = ‖skill_means[-1]‖₂  # line 164
          │                 ├── weights = self.weights(observations)
          │                 ├── self.residual_weights_ = |weights[:,-1]|.mean()      # line 172
          │                 ├── combined = combine_skills(means, stds, weights)  # PoE fusion
          │                 └── self.distribution = Normal(combined_mean, combined_std)
          ├── self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
          ├── self.transition.actions_log_prob = ...detach()
          └── return self.transition.actions

... (env.step, collect all timesteps) ...

LEARNING PHASE (runner.learn(), line 153):
  self.alg.update()
    └── ResidualPPO.update()                              # residual_ppo.py:126
          └── for each mini-batch:
                ├── self.actor_critic.act(obs_batch)       # re-runs update_distribution
                │     └── sets self.residual_action_magnitude  ← overwritten!
                │     └── sets self.residual_weights_           ← overwritten!
                ├── compute PPO loss...
                ├── loss += self.actor_critic.residual_action_magnitude * penalty_coef  # line 187
                ├── loss += self.actor_critic.residual_weights_ * penalty_coef          # line 188
                ├── residual_action_magnitude += self.actor_critic.residual_action_magnitude  # line 185
                └── residual_weight += self.actor_critic.residual_weights_                    # line 186

          return mean_value_loss, mean_surrogate_loss, residual_weight, residual_action_magnitude
```

The key insight: `self.actor_critic.residual_action_magnitude` and
`self.residual_weights_` are **side effects** of calling `.act()` on the
`ResidualActorCritic`. They get set in `update_distribution()` at line 164 and line 172,
then read back in `ResidualPPO.update()` at lines 185-188. This works because `.act()`
is called inside the mini-batch loop (line 139) right before the penalty terms are
accessed.

The values used during updates are from the **current mini-batch** (re-evaluated with
current parameters), not from the rollout buffer. This is correct — the penalty should
reflect the current policy's behavior, not old behavior.

### What the Penalties Actually Do

| Penalty | Formula | Purpose |
|---|---|---|
| `residual_action_magnitude` | `mean(‖a_res‖₂)` across batch | Keep residual corrections small — prevents the residual from taking over completely |
| `residual_weights_` | `mean(|w_res|)` across batch | Keep the residual's gating weight small — penalizes overuse of the residual branch |

During backprop:

- `residual_action_magnitude` gradient flows through `actor[1]` (the residual policy) →
  pushes its action means toward zero
- `residual_weights_` gradient flows through the weight network → pushes it toward
  preferring the locomotion branch

Both penalties are **always active**, even when the residual should NOT be used. This
creates a tension: the task reward (e.g., `object_target_dist = 2.0`) must be large enough
to overcome these penalties for the residual to justify activating. The frozen locomotion
policy provides a stable baseline — the residual only deviates when the net benefit
(task reward minus penalties) is positive.

### Return Value Connection to the Runner

```python
# runner.learn(), line 153:
mean_value_loss, mean_surrogate_loss, residual_weight, residual_action_magnitude = self.alg.update()

# These are passed to self.log(locals()), which writes to TensorBoard:
#   'Loss/value_function'        → mean_value_loss
#   'Loss/surrogate'              → mean_surrogate_loss
#   'Residual/residual_weight'    → residual_weight
#   'Residual/residual_action'    → residual_action_magnitude
```

### Summary: what ResidualPPO adds vs PPO

```
Base PPO                          ResidualPPO
─────────────────────────────────────────────────────────
Constructor:
  clip_param, gamma, lam, ...      + residual_action_penalty_coef
  learning_rate, schedule          + residual_weight_penalty_coef

act():
  actor_critic.act(obs)            SAME (but calls ResidualActorCritic.act
  actor_critic.evaluate(crit)       which sets magnitude/weight as side effect)

process_env_step():
  storage.add_transitions(...)     SAME

compute_returns():
  storage.compute_returns(...)     SAME

update():
  PPO clip loss + value loss        + residual_action_magnitude * coef
  + entropy bonus                   + residual_weights_ * coef
  return (value_loss, surrogate)   return (value_loss, surrogate, weight, action_mag)
```

---

## 6. ResidualActorCritic Module

File: `external/rsl_rl/rsl_rl/modules/residual_actor_critic.py` (262 lines)

This is the `nn.Module` that sits at the center of everything — the runner creates it,
the algo calls `.act()` / `.evaluate()` / `.get_actions_log_prob()` on it, and it owns
the side-effect attributes that the penalty system reads.

### Building Blocks

**`Policy` (line 10):** MLP mapping observations → `(action_mean, action_std)`.

```
obs → Linear → ELU → ... → action_means, action_stds
```

```python
class Policy(nn.Module):
    def __init__(self, hidden_dims, input_dim, num_actions, activation):
        actor_layers = []
        actor_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                pass
            else:
                actor_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.base_extractor = nn.ModuleList(actor_layers)
        self.action_means = nn.Linear(hidden_dims[-1], num_actions)
        self.action_stds = nn.Linear(hidden_dims[-1], num_actions)
        self.action_stds.bias.data.fill_(1.0)

    def forward(self, obs):
        x = obs
        for layer in self.base_extractor:
            x = layer(x)
        return self.action_means(x), torch.abs(self.action_stds(x))
```

Note: the final layer is intentionally skipped (no activation after the last hidden
layer). This is a common RSL-RL pattern.

**`Weights` (line 35):** MLP mapping observations → softmax over N skills.

```python
class Weights(nn.Module):
    def __init__(self, hidden_dims, input_dim, num_skills, activation):
        actor_layers = []
        actor_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                actor_layers.append(nn.Linear(hidden_dims[-1], num_skills))
            else:
                actor_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.layers = nn.Sequential(*actor_layers)

    def forward(self, obs):
        return nn.functional.softmax(self.layers(obs))
```

### Constructor — `__init__` (lines 59-97)

```python
ResidualActorCritic.__init__(
    num_skills=2,                       # from runner, hardcoded
    obs_sizes=Dict[str, int],           # {"base_lin_vel": 3, "dof_pos": 12, ...}
    actor_obs=List[List[str]],          # [["base_lin_vel", ...], ["base_lin_vel", ..., "ball_pos"]]
    critic_obs=List[List[str]],         # [["base_lin_vel", ..., "ball_pos"]]
    num_actions=int,                    # action dim
    actor_hidden_dims=List[List[int]],  # [[512,256,128], [512,256,128]]
    critic_hidden_dims=List[int],       # [512,256,128]
    weight_network_dims=List[int],      # [512,256]
    activation='elu',
    init_noise_std=1.0,
)
```

### What It Builds

**1. Actor branches** (lines 75-81):

```python
actor_obs_size = [self.get_obs_size(obs_sizes, actor_obs_) for actor_obs_ in actor_obs]
# For walk+push: [45, 49]  → sum of dims for each skill's observation segment list

self.actor_branches = [Policy(hidden_dims[i], actor_obs_size[i], num_actions, activation)
                       for i in range(num_skills)]
self.actor = nn.ModuleList(self.actor_branches)
# actor[0]: Policy([512,256,128], 45, 12)  → locomotion (frozen later)
# actor[1]: Policy([512,256,128], 49, 12)  → residual (trainable)
```

**2. Critic** (line 83):

```python
self.critic = self.create_network(critic_obs_size[-1], 1, critic_hidden_dims, activation)
# MLP: 49 → 512 → 256 → 128 → 1
```

A single value network. No multi-branch here — the critic sees the full observation.

**3. Weight network** (line 85):

```python
self.weights = Weights(weight_network_dims, critic_obs_size[-1], num_skills, activation)
# Weights: 49 → 512 → 256 → softmax → [w₀, w₁]
```

Also sees the full observation. Output is a softmax over `num_skills` (2) categories.

### Observation Slicing — the Column-Index Mechanism (lines 102-135)

This is how the same flat observation tensor gets split into different sub-vectors for
each skill.

```python
# Step 1: compute the start:end index for each named segment
def get_obs_segment_start_end(self, obs_sizes):
    start = np.cumsum([0] + [length for obs_name, length in obs_sizes.items()][:-1])
    end = np.cumsum([length for obs_name, length in obs_sizes.items()])
    segments = {k: (start_, end_) for k, start_, end_ in zip(obs_sizes.keys(), start, end)}
    return segments
    # e.g., {"base_lin_vel": (0,3), "dof_pos": (3,15), "ball_pos": (15,18), ...}

# Step 2: for each skill's observation list, collect the column indices
def get_obs_indices(self, obs_sizes, actor_obs):
    segments = self.get_obs_segment_start_end(obs_sizes)
    obs_indices = []
    for actor_obs_ in actor_obs:
        start_end_indices = [segments[obs_name] for obs_name in actor_obs_]
        indices = [range(start, end) for start, end in start_end_indices]
        indices = [item for sublist in indices for item in sublist]
        obs_indices.append(torch.tensor(indices, device=next(self.parameters()).device))
    return obs_indices
    # actor_obs[0] → indices [0,1,2, ..., 47] (48 dims, no ball info)
    # actor_obs[1] → indices [0,1,2, ..., 53] (54 dims, with ball info)
```

At runtime, `select_obs()` slices the flat tensor (line 134-135):

```python
def select_obs(self, observations, i):
    return torch.index_select(observations, -1, self.actor_obs_indices[i])
    # observations: [N, 54]
    # actor_obs_indices[0]: [0,1,2, ..., 47]  → selects cols for skill 0 → [N, 48]
    # actor_obs_indices[1]: [0,1,2, ..., 53]  → selects cols for skill 1 → [N, 54]
```

### How Different Observation Spaces Are Reconciled

**The environment produces one big flat vector** containing all observation segments
concatenated together. Each skill receives a **subset of columns** via `torch.index_select`.

```
Full observation (e.g., 54 dims):
┌───────────────┬───────────────┬───────────────┬─────────┬─────────┬─────────┬───────────┬───────────┬──────────────┐
│ base_lin_vel  │ base_ang_vel  │ proj_gravity  │ dof_pos │ dof_vel │ actions │ command   │ ball_pos  │ ball_target  │
│ 3             │ 3             │ 3             │ 12      │ 12      │ 12      │ 3         │ 3         │ 3            │
└───────────────┴───────────────┴───────────────┴─────────┴─────────┴─────────┴───────────┴───────────┴──────────────┘
```

Each skill's config specifies which named segments it receives:

```python
actor_obs = [
    # locomotion: NO ball info → 48 dims
    ["base_lin_vel", "base_ang_vel", "projected_gravity",
     "dof_pos", "dof_vel", "actions", "command"],

    # residual/push: INCLUDES ball info → 54 dims
    ["base_lin_vel", "base_ang_vel", "projected_gravity",
     "dof_pos", "dof_vel", "actions", "ball_pos", "ball_target"],
]
```

Column indices are precomputed at init time. At runtime, `select_obs` calls
`torch.index_select` to pull only the relevant columns. The locomotion checkpoint was
trained on 48 dims and still receives exactly 48 dims — no modification, no retraining,
no padding.

**Why the weight network sees everything:** the weight network and critic receive the full
observation (all dims), so they have full context for gating decisions and value estimation.

### Forward Pass — `update_distribution()` (lines 159-180)

This is called by `.act()` and is the core of the architecture:

```python
def update_distribution(self, observations):
    # 1. Run each frozen/trainable branch on its own observation slice
    skill_outputs = [branch(self.select_obs(observations, i))
                     for i, branch in enumerate(self.actor)]
    skill_means, skill_std = zip(*skill_outputs)
    # skill_means = (mean₀, mean₁)    each [N, num_actions]
    # skill_std   = (std₀, std₁)      each [N, num_actions]

    self.std = torch.stack(skill_std, 1)

    # 2. Store side effects for penalty system
    self.residual_action_magnitude = torch.norm(skill_means[-1], p=2, dim=1).mean()
    # scalar: how large the residual branch's action proposal is

    # 3. Run weight network on FULL observation
    weights = self.weights(observations)         # [N, 2] softmax
    self.residual_weights_ = torch.abs(weights[:, -1]).mean()
    # scalar: average weight assigned to the residual branch
    self.instance_weights = weights

    # 4. Product-of-Experts fusion
    combined_mean, combined_std = self.combine_skills(
        torch.stack(skill_means, 1),           # [N, num_skills, num_actions]
        self.std,                                # [N, num_skills, num_actions]
        torch.stack(num_actions * [weights], dim=-1))  # [N, num_skills, num_actions]

    self.distribution = Normal(combined_mean, combined_std)
```

### Product-of-Experts Fusion — `combine_skills()` (lines 137-157)

This is the core of multiplicative policy composition:

```python
def combine_skills(self, means, stds, weights):
    stds = stds + 1e-2                              # prevent division by zero
    scaled_weights = weights / stds                 # weight proportional to precision (1/std)
    combined_std   = 1 / scaled_weights.sum(dim=1)  # harmonic mean of variances
    combined_mean  = combined_std * (means * scaled_weights).sum(dim=1)
    return combined_mean, combined_std
```

For each timestep `n` and each action dimension `d`:

```
Given:  [(mean₀, σ₀), (mean₁, σ₁)]   and weights [w₀, w₁]

Step 1: σ_i = σ_i + 0.01              # prevent division by zero
Step 2: sw_i = w_i / σ_i              # weight scaled by precision
Step 3: σ_combined = 1 / Σ sw_i       # harmonic mean of variances
Step 4: μ_combined = σ_combined * Σ (μ_i * sw_i)  # precision-weighted mean
```

**Intuition:** each skill "votes" with weight proportional to `w_i / σ_i`. A skill that's
confident (small σ) gets more say. A skill the weight network favors (large w) gets more
say. When both are confident and weighted equally, the combined action is the midpoint.
When one dominates (high confidence + high weight), the other's vote is negligible.

Example: `w₀=0.7, σ₀=0.1, w₁=0.3, σ₁=0.5` → `scaled_weights = [7.0, 0.6]`. The walk
policy contributes ~92% of the final action.

### Deployment / Inference — `act_inference()` (lines 221-237)

Nearly identical to `update_distribution()` but returns the **deterministic** combined
mean (no sampling, no critic):

```python
def act_inference(self, observations):
    skill_outputs = [branch(self.select_obs(observations, i))
                     for i, branch in enumerate(self.actor)]
    skill_means, std = zip(*skill_outputs)
    weights = self.weights(observations)
    combined_mean, combined_std = self.combine_skills(
        torch.stack(skill_means, 1),
        torch.stack(std, 1),
        torch.stack(num_actions * [weights], dim=-1))
    return combined_mean  # deterministic — no sampling, no critic
```

This is what runs on the real robot. The full pipeline — slice obs, run both branches,
weight network, PoE fusion — runs every control cycle.

### The API Surface — Methods ResidualPPO Calls

| Method | Called by | What it does |
|---|---|---|
| `act(observations)` | `ResidualPPO.act()` during rollout **and** update loop | calls `update_distribution()` → sets `self.distribution` → samples from it |
| `evaluate(critic_observations)` | `ResidualPPO.act()` for values, `ResidualPPO.compute_returns()` for last value | `self.critic(obs) → scalar` |
| `get_actions_log_prob(actions)` | `ResidualPPO.act()` for old_log_prob, `ResidualPPO.update()` for new_log_prob | `self.distribution.log_prob(actions).sum(-1)` |
| `action_mean` (property) | `ResidualPPO.act()` for storing `old_mu_batch` | `self.distribution.mean` |
| `action_std` (property) | `ResidualPPO.act()` for storing `old_sigma_batch` | `self.distribution.stddev` |
| `entropy` (property) | `ResidualPPO.update()` for entropy bonus | `self.distribution.entropy().sum(-1)` |

### What Gets Gradients vs What Doesn't

```
ResidualActorCritic
├── actor (nn.ModuleList)
│   ├── actor[0]  → Policy (locomotion)
│   │   ├── base_extractor   ✗ FROZEN (set by load_skills in runner)
│   │   ├── action_means     ✗ FROZEN
│   │   └── action_stds      ✗ FROZEN
│   └── actor[1]  → Policy (residual)
│       ├── base_extractor   ✓ TRAINABLE
│       ├── action_means     ✓ TRAINABLE
│       └── action_stds      ✓ TRAINABLE
├── critic                   ✓ TRAINABLE  (MLP)
└── weights                  ✓ TRAINABLE  (Weights MLP → softmax)
```

The optimizer receives: `actor[1].parameters() + critic.parameters() +
weights.parameters()`. Since `actor[0]` has `requires_grad=False`, Adam ignores it
automatically.

---

## 7. File Index

| Concept | File |
|---|---|
| ResidualSkillOnPolicyRunner | `external/rsl_rl/rsl_rl/runners/on_policy_runner_residualskill.py` |
| OnPolicyRunner (base) | `external/rsl_rl/rsl_rl/runners/on_policy_runner.py` |
| ResidualActorCritic (2-branch, basic) | `external/rsl_rl/rsl_rl/modules/residual_actor_critic.py` |
| ResidualPPO algorithm | `external/rsl_rl/rsl_rl/algorithms/residual_ppo.py` |
| PPO algorithm (base) | `external/rsl_rl/rsl_rl/algorithms/ppo.py` |
| PoE fusion `combine_skills()` | `residual_actor_critic.py:137-157` |
| Residual action penalty | `residual_actor_critic.py:164` |
| Residual weight penalty | `residual_actor_critic.py:172` |
| Column-level observation slicing | `residual_actor_critic.py:108-135` |
| Weight bias initialization | `on_policy_runner_residualskill.py:99-100` |
| Task registry (runner factory) | `external/legged_gym/legged_gym/utils/task_registry.py:263-314` |
| A1MultiSkillObjectPushCfgPPO (config) | `external/legged_gym/legged_gym/envs/a1/a1_config.py:1593-1668` |
| A1TargetObjectPushCfg (env config) | `external/legged_gym/legged_gym/envs/a1/a1_config.py:1671-1803` |
| A1FlatCfg (locomotion env config) | `external/legged_gym/legged_gym/envs/a1/a1_config.py:139-198` |
| Base LeggedRobot (rewards) | `external/legged_gym/legged_gym/envs/base/legged_robot.py` |
| PushingRobot (object push rewards) | `external/legged_gym/legged_gym/envs/base/object_pushing.py` |
| TargetReachingRobot (target reach rewards) | `external/legged_gym/legged_gym/envs/base/target_reaching.py` |
| RolloutStorage | `external/rsl_rl/rsl_rl/storage/rollout_storage.py` |

---

## 8. Appendix: ResNet-Style Residual vs CCRL

A natural question: ResNets learn residuals as `y = x + F(x)` where `F` sees the input it
corrects. In CCRL, the residual policy does not see `a_loco` as explicit input — it only
sees the observation vector. Would a skip connection help?

### What CCRL Does: Residual in Action Space

```
a_combined = PoE_fusion(a_loco, a_res, w)
           = combine(f_loco(obs_loco), f_res(obs_full), g(obs_full))
```

The residual policy `f_res` takes `obs_full` and directly outputs `a_res` — a full action
vector. It has **no explicit knowledge** of `a_loco` during its forward pass. The
"residual" relationship only emerges indirectly through the PoE fusion + penalty terms.

### What ResNets Do: Residual in Feature Space

```
y = x + F(x)
```

`F(x)` sees `x` directly and learns the *delta* from it. The skip connection ensures
`F(x)` only needs to model the difference.

### Arguments For a Skip Connection

- The residual network explicitly knows what it's correcting — it can learn to say "push
  the hip 0.1 rad more than walking would," rather than independently computing a full
  action and hoping PoE blending works out
- The `‖a_res‖²` penalty becomes less necessary — the architecture naturally biases toward
  small corrections (if initialized near zero)
- Potentially faster learning, especially when the base policy changes behavior (walking
  fast vs slow vs turning)

### Arguments Against (Why CCRL Didn't Do It)

- **Architecture coupling**: the residual policy's input dimension now depends on the
  locomotion policy's action dimension — you can't swap in a different base skill with
  more/fewer actions without changing the network
- The locomotion policy's output is already in the observation vector via the `actions`
  segment (the previous timestep's action). The residual policy does see what was executed
  last step — just not what the locomotion policy *currently wants* to do
- The PoE fusion already creates a functional relationship: `combined_mean` blends both
  actions proportionally to their confidence. If the residual is confident (`σ_res` small)
  and the locomotion is uncertain (`σ_loco` large), the residual dominates
- In practice, the penalty-based approach works well enough that the extra complexity isn't
  justified

### Potential Middle Ground

The most natural modification would be to **concatenate `a_loco` into the residual
policy's observation**:

```
a_res = f_res([obs_full, f_loco(obs_loco).detach()])
```

This keeps the PoE fusion unchanged but gives the residual branch explicit context about
what the base policy is trying to do *this timestep* (not just last timestep's action).
The locomotion output is already computed and cheap (one frozen forward pass), so this
adds negligible cost.

Whether this helps in practice depends on how much the locomotion policy's intent varies
from step to step. If the frozen policy mostly outputs the same "walk forward at
0.5 m/s" pattern, the residual doesn't gain much. If the frozen policy outputs diverse
behaviors (walking, turning, standing), knowing the current intent could help the
residual make more targeted corrections.
