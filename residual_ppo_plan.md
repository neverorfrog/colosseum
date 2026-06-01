# Residual PPO Integration Plan

## Overview

Integrate CCRL-style residual PPO (frozen base policy + trainable residual policy with gating weights) into Colosseum's PPO architecture. Target use case: kick a ball while walking with a pre-trained locomotion policy.

## Key Design Decisions

1. **Dict-based, no column indices**: All observation composition uses named groups (e.g., `["actor"]`, `["actor", "task_obs"]`). Per-skill tensors are composed at collection time. No numeric column slicing anywhere.
2. **Separate ResidualActor and critic**: `ResidualActor` owns skill branches + weight network + PoE fusion. Critic reuses `PpoValueNet` unchanged.
3. **Weight network + critic observe the "critic" observation group**: Matches colosseum's existing pattern and avoids privileged info problems during deployment.
4. **Per-skill observation normalization**: Each skill gets its own `EmpiricalNormalization` since different skills see different observation groups (different dims and distributions).

## File Structure

```
src/colosseum/algorithm/
├── networks/
│   ├── residual_actor.py       # NEW: ResidualActor (multi-skill branches + PoE fusion)
│   └── weight_network.py       # NEW: WeightNetwork (gating MLP → softmax)
├── residual_ppo.py             # NEW: ResidualPPO (extends PPO, adds penalties + skill loading)
src/colosseum/config/types/
├── algorithm.py                # ADD: ResidualPpoConfig
```

## Component Details

### 1. WeightNetwork (`networks/weight_network.py`)

Simple MLP observing the critic observation group, outputting softmax weights over skills.

```python
class WeightNetwork(Network):
    """MLP: critic_obs → softmax → [w_skill0, w_skill1, ...]"""
    def __init__(self, input_dim, hidden_layers, num_skills, activation):
        # backcone: Network base (orthogonal init)
        # value_head → Linear(last_dim, num_skills)
    def forward(self, critic_obs) -> Tensor:  # [B, num_skills], post-softmax
```

Extends colosseum's `Network` base class.

### 2. ResidualActor (`networks/residual_actor.py`)

Combines multiple `PpoActor` branches + `WeightNetwork` + PoE fusion.

```
ResidualActor
├── skill_branches: nn.ModuleDict[str, PpoActor]
│   ├── "locomotion": PpoActor(obs_dim[locomotion], action_dim)  # FROZEN
│   └── "kick":       PpoActor(obs_dim[kick], action_dim)        # TRAINABLE
├── weight_network: WeightNetwork
├── frozen_skills: set[str]                    # which branches are frozen
├── trainable_skill_names: list[str]           # which branches get penalties
├── residual_action_magnitude: float           # side-effect for penalty
└── residual_weights_: float                   # side-effect for penalty
```

```python
class ResidualActor(nn.Module):
    def __init__(
        self,
        skill_configs: dict[str, tuple[PpoActorConfig, int]],
        #  ^^  {"locomotion": (cfg, obs_dim), "kick": (cfg, obs_dim)}
        action_dim: int,
        weight_network: WeightNetwork,
        frozen_skills: set[str] | None = None,
    )

    # ---- Core methods ----
    def update_distribution(self, skill_obs: dict[str, Tensor], critic_obs: Tensor) -> None
        # 1. Run each branch on its named obs: branch(skill_obs[name]) → (mean, std)
        # 2. Run weight network on critic_obs → [w_0, w_1, ...]
        # 3. PoE fusion: combine_skills(means, stds, weights)
        # 4. Set self.distribution = Normal(combined_mean, combined_std)
        # 5. Track residual_action_magnitude = mean(‖μ_trainable‖₂)
        # 6. Track residual_weights_ = mean(|w_trainable|)

    def act(self, skill_obs, critic_obs) -> Tensor
    def act_inference(self, skill_obs, critic_obs) -> Tensor
    def evaluate(self, skill_obs, critic_obs, actions) -> Tuple[Tensor, Tensor]
    def get_distribution_params(self, skill_obs, critic_obs) -> Tuple[Tensor, Tensor]

    # ---- Skill management ----
    def freeze_skill(self, name: str) -> None
    def load_skill(self, name: str, state_dict: dict) -> None
    def trainable_parameters(self) -> Iterator[Parameter]

    # ---- Fusion ----
    @staticmethod
    def combine_skills(means, stds, weights) -> Tuple[Tensor, Tensor]
```

**PoE fusion** (same math as rsl_rl CCRL):
```
σ_i = σ_i + 1e-2
sw_i = w_i / σ_i                  # weight scaled by precision
σ_combined = 1 / Σ sw_i            # harmonic mean of variances
μ_combined = σ_combined * Σ (μ_i * sw_i)  # precision-weighted mean
```

**Penalty tracking** (side effects set during `update_distribution`):
- `residual_action_magnitude = mean(‖μ_trainable‖₂)` — penalizes large residual actions
- `residual_weights_ = mean(|w_trainable|)` — penalizes high residual weight

### 3. ResidualPpoConfig (`config/types/algorithm.py`)

```python
@dataclass(frozen=True)
class ResidualPpoConfig(PpoConfig):
    name: str = "ResidualPPO"
    target: str = "colosseum.algorithm.residual_ppo:ResidualPPO"

    # Observation group names that each skill sees
    # e.g. {"locomotion": ["actor"], "kick": ["actor", "task_obs"]}
    skill_obs_groups: dict[str, list[str]]

    # Actor network config per skill
    # e.g. {"locomotion": PpoActorConfig(...), "kick": PpoActorConfig(...)}
    skill_actor_configs: dict[str, PpoActorConfig]

    # Which skills are frozen (first loaded, then requires_grad=False)
    frozen_skills: list[str] = []

    # Weight network architecture
    weight_network_hidden_layers: list[int] = [512, 256]
    weight_network_activation: str = "relu"

    # Penalty coefficients (rsl_rl CCRL defaults)
    residual_action_penalty_coef: float = 0.01
    residual_weight_penalty_coef: float = 0.01

    # Paths to frozen skill checkpoints (colosseum PPO checkpoints)
    frozen_skill_checkpoints: dict[str, str] = {}
```

### 4. ResidualPPO (`residual_ppo.py`)

```python
@register_algorithm("residual_ppo", config_class=ResidualPpoConfig)
class ResidualPPO(PPO):
```

**Overrides from PPO:**

| Method | Change |
|--------|--------|
| `_build_networks()` | Creates `ResidualActor` (replaces `PpoActor`). Creates `PpoValueNet` (unchanged). |
| `_build_optimizers()` | Optimizer over `residual_actor.trainable_parameters()` + `critic.parameters()`. Still separate actor/critic AdamW. |
| `_build_rollout_buffer()` | `actor_obs_dim` = combined dim (sum of all group dims). `extras` includes per-skill obs storage keys. |
| `_build_normalizer()` | Per-skill normalizers: `dict[str, EmpiricalNormalization]` for actor. Single `EmpiricalNormalization` for critic (unchanged). |
| `_collect_rollout()` | Composes per-skill obs tensors from named groups. Stores them in buffer extras. |
| `_learning_step()` | Passes per-skill obs dict + critic obs to ResidualActor. Adds penalty terms to loss. |
| `save()` | Includes per-skill normalizer states, weight network state. |
| `load()` | Restores residual-specific state. |
| `_compose_actor_input()` | Identity (no encoder latents to append). |
| `_prewarm_actor_obs()` | Handles per-skill normalizer prewarming. |

**New methods:**

```python
def load_frozen_skills(self) -> None:
    """Load colosseum PPO checkpoints, install + freeze skill branches."""
    for skill_name, path in self.config.frozen_skill_checkpoints.items():
        checkpoint = torch.load(path)
        actor_state = checkpoint["actor_state_dict"]
        self.residual_actor.load_skill(skill_name, actor_state)
        self.residual_actor.freeze_skill(skill_name)
    self._init_weight_network_bias()

def _compose_skill_obs(self, obs_dict) -> dict[str, torch.Tensor]:
    """Build per-skill tensors from named observation groups."""
    return {
        name: torch.cat([obs_dict[group] for group in groups], dim=-1)
        for name, groups in self.config.skill_obs_groups.items()
    }

def _init_weight_network_bias(self) -> None:
    """Initialize weight network output bias to favor frozen skills."""
    # e.g., for 1 frozen + 1 residual: bias = [0.8, 0.01]
    # Softmax of these logits heavily favors the frozen skill initially
```

## Observation Flow (End-to-End)

### Configuration (task config)

```python
# In observation_cfg.py:
observations = {
    "actor": ObservationGroupCfg(terms=proprio_terms, concatenate_terms=True),
    "critic": ObservationGroupCfg(terms=all_terms, concatenate_terms=True),
    "task_obs": ObservationGroupCfg(terms=task_terms, concatenate_terms=True),
}
```

### ResidualPpoConfig wiring

```python
ResidualPpoConfig(
    skill_obs_groups={
        "locomotion": ["actor"],                # proprioception only
        "kick":       ["actor", "task_obs"],    # proprio + ball/target info
    },
    skill_actor_configs={
        "locomotion": PpoActorConfig(hidden_layers=[512, 256, 128]),
        "kick":       PpoActorConfig(hidden_layers=[512, 256, 128]),
    },
    frozen_skills=["locomotion"],
    frozen_skill_checkpoints={
        "locomotion": "/path/to/locomotion.pt",
    },
    # PPO params inherited from PpoConfig...
)
```

### Collection (per step)

```
env.step(action) → obs_dict = {
    "actor":     tensor[B, 48],   # proprioception
    "critic":    tensor[B, 54],   # proprio + privileged
    "task_obs":  tensor[B, 6],    # ball pos, target pos
}

_compose_skill_obs(obs_dict) → {
    "locomotion": tensor[B, 48],           # obs_dict["actor"]
    "kick":       tensor[B, 54],           # cat(obs_dict["actor"], obs_dict["task_obs"])
}

critic_obs = obs_dict["critic"]             # tensor[B, 54]

# Normalize and get action
norm_skill_obs = {name: normalizers[name](obs) for name, obs in skill_obs.items()}
norm_critic_obs = critic_normalizer(critic_obs)
action = residual_actor.act(norm_skill_obs, norm_critic_obs)

# Store in buffer
buffer.add(
    actor_obs=combined_obs,                  # cat of all groups (for completeness)
    critic_obs=critic_obs_raw,               # stored raw, normalized during learning
    extras={"skill_obs:locomotion": skill_obs["locomotion"],
            "skill_obs:kick": skill_obs["kick"]},
    ...actions, rewards, dones, values, log_probs, means, stds...
)
```

### Learning (per mini-batch)

```
batch = generator.next()
skill_obs = {name: batch[f"skill_obs:{name}"] for name in skill_names}

# Normalize
norm_skill_obs = {name: normalizers[name](obs) for name, obs in skill_obs.items()}
norm_critic_obs = critic_normalizer(batch["critic_obs"])

# Re-evaluate with current params
new_log_probs, entropy = residual_actor.evaluate(norm_skill_obs, norm_critic_obs, actions)
new_values = critic(norm_critic_obs)

# Standard PPO loss + residual penalties
loss = surrogate_loss + value_loss - entropy_coef * entropy
     + residual_action_penalty_coef * residual_actor.residual_action_magnitude
     + residual_weight_penalty_coef * residual_actor.residual_weights_

# Backward through trainable params only (frozen branches have requires_grad=False)
```

## Open Questions

1. **Weight network + critic observation**: Should they observe the "critic" group only, or the full combined obs? Proposal: "critic" group (matches existing pattern, avoids privileged info during deployment).

2. **Per-skill normalization**: Separate `EmpiricalNormalization` per skill (since each skill sees different dims/distributions). Or single normalizer over the combined obs? Proposal: per-skill normalizers.

3. **Combined actor_obs buffer slot**: Store full combined obs (unused by ResidualActor) or make the slot a dummy? Proposal: store combined obs for backward compatibility with buffer structure.

## Implementation Order

1. `WeightNetwork` — small, standalone
2. `ResidualActor` — core module (multi-branch + PoE fusion)
3. `ResidualPpoConfig` — config dataclass
4. `ResidualPPO` — algorithm class
5. Test with simple setup (pretend frozen skill, verify gradients flow correctly)
