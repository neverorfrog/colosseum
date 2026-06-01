# CCRL Residual Learning Implementation

This document explains the Cascaded Compositional Residual Learning framework from [Kumar, Essa, Ha (2022)](https://arxiv.org/abs/2212.08954), as implemented in `external/legged_gym` and `external/rsl_rl`.

## What Problem Does CCRL Solve?

Suppose you have a policy that already walks well (trained with PPO). Now you want the robot to kick a ball toward a target while walking. Training from scratch is slow, and fine-tuning the locomotion policy risks losing the walking style.

**CCRL's approach:** freeze the walking policy. Add a residual policy that learns corrective actions for kicking, plus a weight network that blends the two at each timestep. The robot walks normally by default and only kicks when the task reward justifies it.

## The Three Mechanisms

| Mechanism | What it learns | Trainable? |
|---|---|---|
| Skill Library | Frozen locomotion policy | No (frozen) |
| Residual Policy | Corrective kick/dribble actions | Yes |
| Weight Network | When to use each skill (gating) | Yes |

Note: the paper's Goal Network (MetaBackbone for synthetic observations) lives in `MultiSkillActorCriticv3`. It is only needed for complex hierarchical compositions (e.g., indoor navigation + door opening). For walk+kick with 2 skills, we use the simpler `ResidualActorCritic`.

## Architecture: `ResidualActorCritic`

File: `rsl_rl/rsl_rl/modules/residual_actor_critic.py`

### Building Blocks

**`Policy` (line 10):** MLP mapping observations → `(action_mean, action_std)`.
```
obs → Linear → ELU → ... → action_means, action_stds
```

**`Weights` (line 35):** MLP mapping observations → softmax over N skills.
```
critic_obs → Linear → ELU → ... → softmax → [w_0, ..., w_{N-1}]
```

### Forward Pass

```
                    Environment Observation (flat concatenated vector)
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      actor[0] (WALK)     actor[1] (KICK)     Weights Network
      48-dim slice        51-dim slice         54-dim full obs
      FROZEN              TRAINABLE            TRAINABLE
      → (mean₀, std₀)     → (mean₁, std₁)     → softmax → [w₀, w₁]
              │                  │                  │
              └────────┬─────────┘                  │
                       ▼                            │
              combine_skills(means, stds, weights) ◄┘
                       │
              ┌────────┴────────┐
              │ combined_mean   │
              │ combined_std    │
              └────────┬────────┘
                       │
         Training: Normal(mean, std) → sample
         Deployment: return combined_mean (deterministic)
```

### `combine_skills()` — Product-of-Experts Fusion (line 137)

This is the core of multiplicative policy composition:

```python
stds = stds + 1e-2                              # prevent division by zero
scaled_weights = weights / stds                 # weight proportional to precision (1/std)
combined_std   = 1 / scaled_weights.sum(dim=1)  # harmonic mean of variances
combined_mean  = combined_std * (means * scaled_weights).sum(dim=1)
```

**Intuition:** a skill with low variance (high certainty) contributes more. When the locomotion policy is confident about walking (small σ₀), it dominates. When the kick policy becomes confident near the ball (small σ₁), it takes over. This is a smooth continuous blend — no hard switching.

Example: `w₀=0.7, σ₀=0.1, w₁=0.3, σ₁=0.5` → `scaled_weights = [7.0, 0.6]`. The walk policy contributes ~92% of the final action.

### `update_distribution()` — Training Step (line 159)

```python
def update_distribution(self, observations):
    # 1. Run each actor on its OWN observation slice
    skill_outputs = [branch(self.select_obs(observations, i))
                     for i, branch in enumerate(self.actor)]
    skill_means, skill_std = zip(*skill_outputs)

    # 2. Track residual magnitude for penalty
    self.residual_action_magnitude = torch.norm(skill_means[-1], p=2, dim=1).mean()

    # 3. Run weight network on full observation
    weights = self.weights(observations)
    self.residual_weights_ = torch.abs(weights[:, -1]).mean()

    # 4. Fuse into single Gaussian
    combined_mean, combined_std = self.combine_skills(
        torch.stack(skill_means, 1),
        torch.stack(skill_std, 1),
        torch.stack(num_actions * [weights], dim=-1))

    self.distribution = Normal(combined_mean, combined_std)
```

## How Different Observation Spaces Are Reconciled

**The environment produces one big flat vector** containing all observation segments concatenated together. Each skill receives a **subset of columns** via `torch.index_select`.

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
    # locomotion: NO ball info
    ["base_lin_vel", "base_ang_vel", "projected_gravity",
     "dof_pos", "dof_vel", "actions", "command"],

    # kick: INCLUDES ball info
    ["base_lin_vel", "base_ang_vel", "projected_gravity",
     "dof_pos", "dof_vel", "actions", "ball_pos", "ball_target"],
]
```

Column indices are precomputed at init time (`residual_actor_critic.py:108-116`). At runtime, `select_obs` calls `torch.index_select` to pull only the relevant columns. The locomotion checkpoint was trained on 48 dims and still receives exactly 48 dims — no modification, no retraining, no padding.

### Why the Weight Network Sees Everything

The weight network and critic receive the full observation (all 54 dims), so they have full context for gating decisions and value estimation.

## Training with `ResidualPPO`

File: `rsl_rl/rsl_rl/algorithms/residual_ppo.py`

### Standard PPO Loss + Residual Regularization (line 186-188)

```python
loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy
loss += self.actor_critic.residual_action_magnitude * self.residual_action_penalty_coef
loss += self.actor_critic.residual_weights_     * self.residual_weight_penalty_coef
```

| Penalty | Purpose | Typical values |
|---|---|---|
| `residual_action_penalty_coef` | Keep residual actions small — only correct locomotion when needed | 0.01–5.0 |
| `residual_weight_penalty_coef` | Prevent over-reliance on residual branch | 0.0–5.0 |

These penalties preserve locomotion style without demonstrations. The residual only deviates when the task reward (e.g., "ball reached target") justifies the penalty cost.

### What Gets Gradients

```
  ✓ actor[1] (kick residual)  — action_means, action_stds, base_extractor
  ✗ actor[0] (locomotion)     — requires_grad = False (frozen)
  ✓ weights network           — learned gating
  ✓ critic                    — value function
```

## The Training Runner

File: `rsl_rl/rsl_rl/runners/on_policy_runner_residualskill.py`

### `load_skills()` — Loading and Freezing (line 88)

```python
def load_skills(self, paths, load_optimizer=False):
    for i, path in enumerate(paths):
        # Load pretrained locomotion into actor[i]
        loaded_dict = torch.load(path)
        model_state_dict = {}
        for name, params in loaded_dict["model_state_dict"].items():
            if "actor" in name:
                name_ = name[6:]  # strip "actor." prefix
                model_state_dict[name_] = params
        self.alg.actor_critic.actor[i].load_state_dict(model_state_dict, True)
        for parms in self.alg.actor_critic.actor[i].parameters():
            parms.requires_grad = False  # FROZEN

    # Initialize weight bias to favor locomotion
    weight_init = len(paths)*[1/len(paths)-0.2] + \
                  (num_branches - len(paths))*[0.01]
    # e.g., [0.3, 0.01] — residual starts with near-zero weight
    self.alg.actor_critic.weights.data = torch.tensor(weight_init, device=self.device)
```

Key points:
- The locomotion checkpoint is loaded and frozen
- The weight network is **not** loaded from any checkpoint — initialized from scratch with bias favoring locomotion
- The residual branch (`actor[1]`) starts with random weights, trained from scratch
- `num_branches` hardcoded to 2 in the runner constructor (`on_policy_runner_residualskill.py:63`)

## Deployment / Inference

### `act_inference()` vs `act()`

| Method | Used in | Returns |
|---|---|---|
| `act(obs)` | Training | Sampled action from Normal(combined_mean, combined_std) |
| `act_inference(obs)` | Deployment | Deterministic `combined_mean` |

```python
# residual_actor_critic.py:221-237
def act_inference(self, observations):
    # Runs ALL branches + weights + fusion — same as training
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

### Real Robot Deployment Pipeline

```
1. DDS LowState → extract IMU, joint pos/vel/torque
2. Build observation vector (same format as training, same column layout)
3. policy(obs) → deterministic combined_mean
4. action = net_output × action_scale + default_joint_pos
5. Remap from sim joint order → hardware joint order
6. Publish LowCmd with q_targets + PD gains at 500 Hz
```

The policy runs at ~50 Hz. The model is typically exported via `torch.jit.script()` (TorchScript `.pt`) or ONNX for C++ deployment.

## How This Differs from Imitation Learning

| Aspect | Imitation Learning | CCRL Residual Learning |
|---|---|---|
| Training signal | Expert demonstrations | Environment reward (PPO) |
| Base skill | Often fine-tuned or discarded | **Frozen**, never modified |
| What is learned | Full policy from demos | Only the difference from what the base skill can do |
| Style preservation | Requires explicit regularization | Emergent — residual penalty pulls corrections toward zero |
| Needs demonstrations? | Yes | No |
| Catastrophic forgetting | Risk if fine-tuned | Impossible — base frozen |

**For walk+kick specifically:**
- IL: Record teleoperated demonstrations, train to reproduce them. Fails if deployment differs from demos.
- CCRL: Define a reward (+1 ball at target, +0.1 approaching ball, -0.01 residual magnitude). PPO discovers how to kick. Walk policy handles all locomotion; residual only activates near the ball.

## Walk+Kick Configuration Template

Based on `A1MultiSkillObjectPushCfgPPO` in `a1_config.py:1593-1668`:

```python
class WalkAndKickCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        residual_action_penalty_coef = 0.02
        residual_weight_penalty_coef = 0.01

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [[512, 256, 128], [512, 256, 128]]
        critic_hidden_dims = [512, 256, 128]
        weight_network_dims = [512, 256]
        activation = 'elu'

    class runner(LeggedRobotCfgPPO.runner):
        algorithm_class_name = 'ResidualPPO'
        policy_class_name = 'ResidualActorCritic'
        max_iterations = 5000

        obs_sizes = {
            "base_lin_vel": 3,  "base_ang_vel": 3,  "projected_gravity": 3,
            "dof_pos": 12,      "dof_vel": 12,       "actions": 12,
            "command": 3,       "ball_pos": 3,        "ball_target": 3,
        }

        actor_obs = [
            ["base_lin_vel", "base_ang_vel", "projected_gravity",
             "dof_pos", "dof_vel", "actions", "command"],           # walk

            ["base_lin_vel", "base_ang_vel", "projected_gravity",
             "dof_pos", "dof_vel", "actions", "ball_pos", "ball_target"],  # kick
        ]

        critic_obs = [["base_lin_vel", "base_ang_vel", "projected_gravity",
                        "dof_pos", "dof_vel", "actions",
                        "command", "ball_pos", "ball_target"]]

        skill_paths = ["/path/to/pretrained_locomotion.pt"]
        experiment_name = "walk_and_kick"
```

## Cascading Explained

The paper's name includes "Cascaded" because residual policies can become frozen primitives for higher-level residual policies. This is needed for complex tasks in the paper (indoor navigation + door opening), but **not** for simple walk+kick.

```
Level 0 (standalone PPO):
  walk, stand, turn_left, turn_right

Level 1 (residual on Level 0):
  target_reach  = residual on [walk, stand, turn_left, turn_right]
  door_open     = residual on [walk]
  crouching     = residual on [walk]

Level 2 (residual on Level 1):
  door_openv2 = residual on [walk, stand, turn_left, turn_right,
                              target_reach, door_open]

Level 3 (residual on Levels 0-2):
  interactive_targetreach = residual on everything above
```

The mechanism: when loading a previously composed skill as a frozen primitive, only its **residual branch** is extracted, not the full composition. This is handled in `MultiSkillOnPolicyRunnerv2.load_skills()` (`on_policy_runner_multiskillv2.py:113-119`).

## File Index

| Concept | File |
|---|---|
| ResidualActorCritic (2-branch, basic) | `rsl_rl/rsl_rl/modules/residual_actor_critic.py` |
| MultiSkillActorCriticv2 (named skills, hierarchy) | `rsl_rl/rsl_rl/modules/multiskill_actor_criticv2.py` |
| MultiSkillActorCriticv3 (MetaBackbone, synthetic obs) | `rsl_rl/rsl_rl/modules/multiskill_actor_criticv3.py` |
| ResidualPPO algorithm | `rsl_rl/rsl_rl/algorithms/residual_ppo.py` |
| ResidualSkillOnPolicyRunner | `rsl_rl/rsl_rl/runners/on_policy_runner_residualskill.py` |
| MultiSkillOnPolicyRunnerv2 | `rsl_rl/rsl_rl/runners/on_policy_runner_multiskillv2.py` |
| PoE fusion `combine_skills()` | `residual_actor_critic.py:137-157` |
| Residual action penalty | `residual_actor_critic.py:164` |
| Residual weight penalty | `residual_actor_critic.py:172` |
| Column-level observation slicing | `residual_actor_critic.py:108-135` |
| Weight bias initialization | `on_policy_runner_residualskill.py:99-100` |
| Task configurations (all) | `legged_gym/legged_gym/envs/a1/a1_config.py` |
| Runner factory methods | `legged_gym/legged_gym/utils/task_registry.py:263-314` |
| JIT export for deployment | `legged_gym/legged_gym/utils/helpers.py:184-194` |
