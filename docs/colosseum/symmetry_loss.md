# Symmetry Loss

> **Status:** planned — not yet implemented.
> Reference implementation: `external/colosseum_can/` (commit `8ba1803`).
> Physics rationale and step-by-step explanation: `external/colosseum_can/docs/dribbling/symmetry_loss.md`.

Humanoid locomotion is bilaterally symmetric: mirroring a valid behavior left↔right produces another valid behavior. The symmetry loss enforces this at the policy level during PPO training:

```
policy(mirror(obs)) ≈ mirror(policy(obs))
```

Violations are penalized via MSE. Additionally, during rollout collection the policy samples from a symmetrized distribution whose mean averages both sides, ensuring the training data itself is symmetric.

---

## Files to create / modify

| File | Action |
|---|---|
| `src/colosseum/mdp/symmetry.py` | **Create** — generic mirror fns + spec builder |
| `src/colosseum/robots/t1_23dof/mdp/symmetry.py` | **Create** — T1-specific joint permutation + sign mask |
| `src/colosseum/config/types/algorithm.py` | **Modify** — add `symmetry_loss_coef` to `PpoConfig` |
| `src/colosseum/algorithm/ppo.py` | **Modify** — build spec at init, symmetric rollout, symmetry loss |
| `src/colosseum/algorithm/networks/ppo_networks.py` | **Modify** — add `act_symmetric_with_log_prob` |
| `src/colosseum/tasks/velocity/config/t1_23dof/observation_cfg.py` | **Modify** — annotate actor terms with `mirror_fn` |

---

## Step 1 — `src/colosseum/mdp/symmetry.py` (new file)

Copy verbatim from `external/colosseum_can/src/colosseum/mdp/symmetry.py`.

Then add one function that the fork lacks — the gait phase mirror. Our velocity task
includes a `gait_phase` actor observation `[cos(φ_L), cos(φ_R), sin(φ_L), sin(φ_R)]`
that the fork's task does not have. Mirroring swaps left↔right feet:

```python
def mirror_gait_phase(x: torch.Tensor) -> torch.Tensor:
    """Mirror gait phase clock under left-right reflection.

    [cos_L, cos_R, sin_L, sin_R] → [cos_R, cos_L, sin_R, sin_L]
    """
    return x[..., [1, 0, 3, 2]]
```

No sign changes: cosine and sine are evaluated on phases, not joint angles, and the
reflection simply swaps which foot is "left" and which is "right".

Full exports from this file:

```python
MirrorableObservationTermCfg
TermMirrorSpec
build_symmetry_spec
mirror_obs
mirror_ang_vel
mirror_projected_gravity
mirror_velocity_command
mirror_gait_phase          # ← our addition
```

---

## Step 2 — `src/colosseum/robots/t1_23dof/mdp/symmetry.py` (new file)

Copy verbatim from `external/colosseum_can/src/colosseum/robots/t1_23dof/mdp/symmetry.py`.

The joint ordering matches our `JOINT_NAMES` in `constants.py` exactly (verified by
comparing both XML files and both `JOINT_NAMES` lists — they are identical).

Joint anatomy reference (indices into the 23-DOF `JOINT_NAMES` ordering):

```
 0  Left_Hip_Pitch        6  Right_Hip_Pitch
 1  Left_Hip_Roll         7  Right_Hip_Roll
 2  Left_Hip_Yaw          8  Right_Hip_Yaw
 3  Left_Knee_Pitch       9  Right_Knee_Pitch
 4  Left_Ankle_Pitch     10  Right_Ankle_Pitch
 5  Left_Ankle_Roll      11  Right_Ankle_Roll
12  Waist
13  Left_Shoulder_Pitch  17  Right_Shoulder_Pitch
14  Left_Shoulder_Roll   18  Right_Shoulder_Roll
15  Left_Elbow_Pitch     19  Right_Elbow_Pitch
16  Left_Elbow_Yaw       20  Right_Elbow_Yaw
21  AAHead_yaw
22  Head_pitch
```

Sign convention: pitch joints (rotation axis ≈ y) use `+1` (symmetric); roll and yaw
joints (axes ≈ x, z) use `-1` (pseudovectors pick up `det(P) = -1` under reflection).

---

## Step 3 — `src/colosseum/config/types/algorithm.py`

Add one field to `PpoConfig` after the existing loss coefficients:

```python
# Symmetry loss
symmetry_loss_coef: float = 0.0
"""Coefficient for the symmetry loss. 0.0 disables it entirely.
Requires actor obs terms to use MirrorableObservationTermCfg with mirror_fn set."""
```

---

## Step 4 — `src/colosseum/algorithm/ppo.py`

### 4a — Imports (top of file)

```python
from colosseum.mdp.symmetry import TermMirrorSpec, build_symmetry_spec, mirror_obs
```

### 4b — `__init__`: build symmetry spec once

After `self._build_normalizer()`, add:

```python
# Symmetry loss: compile obs mirror layout once from the env's observation manager.
self._actor_sym_spec: list[TermMirrorSpec] | None = None
self._action_mirror_fn = None
if config.symmetry_loss_coef > 0.0:
    obs_manager = self.env.observation_manager
    self._actor_sym_spec = build_symmetry_spec(obs_manager, "actor")
    if "actions" in obs_manager.active_terms.get("actor", []):
        actions_cfg = obs_manager.get_term_cfg("actor", "actions")
        self._action_mirror_fn = getattr(actions_cfg, "mirror_fn", None)
```

### 4c — `_collect_rollout`: symmetric action sampling

The existing call to `self.actor.act_with_log_prob(norm_actor_obs)` becomes:

```python
if (
    self.config.symmetry_loss_coef > 0.0
    and self._actor_sym_spec is not None
    and self._action_mirror_fn is not None
):
    mirrored_actor_obs = mirror_obs(norm_actor_obs, self._actor_sym_spec)
    actions, log_probs, action_means, action_stds = self.actor.act_symmetric_with_log_prob(
        norm_actor_obs, mirrored_actor_obs, self._action_mirror_fn
    )
else:
    actions, log_probs, action_means, action_stds = self.actor.act_with_log_prob(
        norm_actor_obs
    )
```

### 4d — `_learning_step`: symmetry loss term

Inside the mini-batch loop, after the value loss and before the total loss, add:

```python
# --- Symmetry loss ---
# Enforce policy(mirror(obs)) == mirror(policy(obs)).
# Uses deterministic action means; mu_batch is detached so gradients only
# flow through actions_on_mirrored (conflicting gradients otherwise).
symmetry_loss = torch.tensor(0.0, device=self.device)
if (
    self.config.symmetry_loss_coef > 0.0
    and self._actor_sym_spec is not None
    and self._action_mirror_fn is not None
):
    mirrored_obs = mirror_obs(actor_obs, self._actor_sym_spec)
    actions_on_mirrored = self.actor.forward(mirrored_obs)
    mirror_of_actions = self._action_mirror_fn(mu_batch.detach())
    symmetry_loss = torch.nn.functional.mse_loss(actions_on_mirrored, mirror_of_actions)
```

Update the total loss line:

```python
loss = (
    surrogate_loss
    + self.config.value_loss_coef * value_loss
    - self.config.entropy_coef * entropy.mean()
    + self.config.symmetry_loss_coef * symmetry_loss
)
```

Add `total_symmetry_loss` accumulation and log it in the returned dict alongside the
other losses.

### 4e — `_eval_get_action`: symmetric inference

```python
def _eval_get_action(self, normalized_obs: torch.Tensor) -> torch.Tensor:
    if (
        self.config.symmetry_loss_coef > 0.0
        and self._actor_sym_spec is not None
        and self._action_mirror_fn is not None
    ):
        mirrored = mirror_obs(normalized_obs, self._actor_sym_spec)
        mu = self.actor.act_inference(normalized_obs)
        mu_mirrored = self.actor.act_inference(mirrored)
        return 0.5 * (mu + self._action_mirror_fn(mu_mirrored))
    return self.actor.act_inference(normalized_obs)
```

---

## Step 5 — `src/colosseum/algorithm/networks/ppo_networks.py`

Add `act_symmetric_with_log_prob` to `PpoActor`:

```python
def act_symmetric_with_log_prob(
    self,
    obs: torch.Tensor,
    mirrored_obs: torch.Tensor,
    action_mirror_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample from the symmetrized distribution.

    sym_mu = 0.5 * (mu(obs) + mirror(mu(mirror(obs))))

    Log probs are computed under Normal(sym_mu, std) so the PPO importance
    ratio exp(new_log_prob - old_log_prob) stays self-consistent.
    """
    mu = self.forward(obs)
    mu_mirrored = self.forward(mirrored_obs)
    sym_mu = 0.5 * (mu + action_mirror_fn(mu_mirrored))
    std = torch.clamp(self.std, min=self.min_noise_std).expand_as(sym_mu)
    dist = torch.distributions.Normal(sym_mu, std)
    actions = dist.sample()
    log_probs = dist.log_prob(actions).sum(dim=-1)
    return actions, log_probs, sym_mu, std
```

Requires adding `from typing import Callable` if not already imported.

---

## Step 6 — `src/colosseum/tasks/velocity/config/t1_23dof/observation_cfg.py`

Replace all `ObservationTermCfg` in `actor_terms` with `MirrorableObservationTermCfg`
and assign the appropriate `mirror_fn` to each term:

```python
from colosseum.mdp.symmetry import (
    MirrorableObservationTermCfg,
    mirror_ang_vel,
    mirror_projected_gravity,
    mirror_velocity_command,
    mirror_gait_phase,
)
from colosseum.robots.t1_23dof.mdp.symmetry import mirror_joints

actor_terms = {
    "base_ang_vel": MirrorableObservationTermCfg(
        func=builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        mirror_fn=mirror_ang_vel,
    ),
    "projected_gravity": MirrorableObservationTermCfg(
        func=projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
        mirror_fn=mirror_projected_gravity,
    ),
    "joint_pos": MirrorableObservationTermCfg(
        func=joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        mirror_fn=mirror_joints,
    ),
    "joint_vel": MirrorableObservationTermCfg(
        func=joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        mirror_fn=mirror_joints,
    ),
    "actions": MirrorableObservationTermCfg(
        func=last_action,
        mirror_fn=mirror_joints,  # also used as the action mirror function in PPO
    ),
    "command": MirrorableObservationTermCfg(
        func=generated_commands,
        params={"command_name": "twist"},
        mirror_fn=mirror_velocity_command,
    ),
    "gait_phase": MirrorableObservationTermCfg(
        func=generated_commands,
        params={"command_name": "gait_phase"},
        mirror_fn=mirror_gait_phase,   # ← not in the fork; our addition
    ),
}
```

Critic terms (`base_lin_vel`, `foot_height`, etc.) stay as plain `ObservationTermCfg` —
the symmetry loss only applies to the actor.

---

## Enabling

Set `symmetry_loss_coef` in `algo_cfg.py`:

```python
PpoConfig(
    ...
    symmetry_loss_coef=1.0,
)
```

Three conditions must all be true for the loss to activate:

- `symmetry_loss_coef > 0.0`
- All actor obs terms use `MirrorableObservationTermCfg` with `mirror_fn` set (or `None` for invariant terms)
- The `"actions"` term has a non-`None` `mirror_fn`

Set `symmetry_loss_coef=0.0` to disable with zero overhead.
