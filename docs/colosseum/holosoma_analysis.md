# Holosoma vs Colosseum: Standing-Still Analysis

> Reference implementation analyzed: `external/holosoma/src/holosoma/holosoma/config_values/loco/t1/`
> Our config: `src/colosseum/tasks/velocity/config/t1_23dof/`

---

## TL;DR

Two distinct problems, two distinct causes:

1. **Walking in place at zero velocity**: We removed all foot-timing learning signal when
   standing — we zero the phase observation AND gate off all phase rewards. The policy has
   no feedback about desired foot contact and keeps shuffling. Holosoma freezes the phase
   at a meaningful value and never gates rewards.

2. **"Elvis Presley" pose / foot on edge**: We measure foot posture only through ankle joint
   angles (relative to default). Holosoma directly measures foot orientation in world frame
   by projecting gravity through the foot's quaternion — `penalty_feet_ori: -5.0`. This
   catches tilted or edge-loaded feet that joint-angle penalties miss entirely, because
   foot flatness is a world-frame property, not just a function of the ankle joint angle.

---

## 1. What Holosoma Does When Standing

### Phase Freezing (`LocomotionGait.step()`)

When `‖cmd_xy‖ < 0.01` and `|ω_z| < 0.01`, the phase is frozen at `stand_phase_value = π`:

```python
stand_mask = torch.logical_and(
    torch.linalg.norm(command_tensor[:, :2], dim=1) < 0.01,
    torch.abs(command_tensor[:, 2]) < 0.01,
)
if stand_mask.any():
    self.phase[stand_mask] = torch.full(
        (int(stand_mask.sum().item()), 2), self.stand_phase_value, device=env.device
    )
```

At φ = π:
- `cos(π) = -1`, `sin(π) = 0`
- The policy **still sees meaningful phase information** — it is not zeroed out
- Both feet are snapped to the same phase value

### Feet Phase Reward (Not Gated, Weight = 5.0)

`feet_phase()` uses a cubic Bézier curve to define an expected foot height profile over [0, 2π].
At φ = π the curve evaluates to **0** (expected foot height = ground level).

The reward is:

```
r = exp(-(actual_height - expected_height)² / 0.008)
```

Sigma = 0.008 is **very tight** — any foot lift above ground is heavily penalized.

**Key property**: this reward is **never gated**. When standing, both phases freeze at π,
the expected height is 0, and the reward punishes any foot movement. The formulation handles
standing naturally without a velocity gate.

### What the Policy Learns

The policy sees `cos = -1, sin = 0` → "this means both feet should be on the ground."
During the many standing episodes, this association is reinforced with a strong 5.0-weight signal.
When walking resumes, phases diverge and the reward drives the expected lift/contact schedule.

### Alive Reward

A constant `alive = 1.0` bonus provides a baseline incentive for survival, which matters
especially during standing episodes when most other rewards are near zero.

---

## 2. What We Do When Standing

### Phase Observation Is Zeroed (`GaitPhaseCommand.command`)

```python
@property
def command(self) -> torch.Tensor:
    phase_cmd = torch.cat([torch.cos(self._phase), torch.sin(self._phase)], dim=-1)
    if self._gate_cmd is not None:
        cmd = self._env.command_manager.get_command(self._gate_cmd)
        moving = (torch.norm(cmd[:, :2], dim=-1) > self.cfg.gate_speed_threshold).float()
        phase_cmd = phase_cmd * moving.unsqueeze(-1)  # ← zeroed when standing
    return phase_cmd
```

When standing, the policy sees `[0, 0, 0, 0]`. No phase information at all.

### Phase Rewards Are Gated Off (`rewards.py`)

```python
if command_name is not None:
    cmd = env.command_manager.get_command(command_name)
    moving = (torch.norm(cmd[:, :2], dim=-1) > command_threshold).float()
    reward = reward * moving  # ← swing_phase (3.0) + stance_phase (2.0) → 0
```

When standing, `swing_phase` (weight 3.0) and `stance_phase` (weight 2.0) are both zeroed.

### What Remains When Standing

| Reward | Weight | Notes |
|--------|--------|-------|
| track_linear_velocity | 2.0 | → 0 (commanded zero = achieved) |
| track_angular_velocity | 2.0 | → 0 (commanded zero = achieved) |
| upright | 0.7 | Active, rewards flat base |
| pose | 1.0 | Active, rewards default pose |
| body_ang_vel | -0.05 | Active |
| angular_momentum | -0.02 | Active |
| dof_pos_limits | -1.0 | Active |
| action_rate_l2 | -0.3 | Active |
| foot_clearance | -2.0 | **GATED OFF** |
| foot_swing_height | -0.25 | **GATED OFF** |
| swing_phase | 3.0 | **GATED OFF** |
| stance_phase | 2.0 | **GATED OFF** |
| foot_slip | -0.1 | **GATED OFF** |
| soft_landing | -1e-5 | **GATED OFF** |
| self_collisions | -1.0 | Active |

**Standing reward baseline: ~1.7** (pose + upright only).

There is no foot-timing feedback. The policy can shuffle its feet arbitrarily when standing
without incurring any penalty — the gated rewards that would penalize this are all off.

---

## 3. Side-by-Side Comparison

### Standing Episode Signal

| | Holosoma | Colosseum |
|---|---|---|
| Phase value | Frozen at π | Stops advancing (correct) |
| Phase observation | `[cos(π), cos(π), sin(π), sin(π)]` = `[-1,-1,0,0]` | `[0, 0, 0, 0]` (blanked) |
| Feet phase reward | Active (weight=5.0), expects height=0 | **Absent** |
| Swing/stance rewards | Not used (different formulation) | **Gated off** (was 5.0 combined) |
| Foot clearance | Not gated | **Gated off** |
| Alive bonus | 1.0 | **Absent** |
| Standing reward baseline | ~5.5 | ~1.7 |
| Learning signal about feet | Strong | **None** |

### Full Reward Comparison

| Reward | Holosoma weight | Colosseum weight | Notes |
|--------|-----------------|------------------|-------|
| tracking_lin_vel | 2.0 | 2.0 | Similar |
| tracking_ang_vel | 1.5 | 2.0 | Similar |
| **feet_phase** | **5.0** | — | **We have nothing equivalent** |
| penalty_orientation | -10.0 | upright 0.7 | Holosoma penalizes harder |
| penalty_ang_vel_xy | -1.0 | body_ang_vel -0.05 | Holosoma penalizes harder |
| penalty_action_rate | -2.0 | -0.3 | Holosoma penalizes harder |
| **penalty_close_feet_xy** | **-10.0** | — | **We don't have this** |
| penalty_feet_ori | -5.0 | — | We don't have this |
| **alive** | **1.0** | — | **We don't have this** |
| pose | -0.5 | +1.0 | Different sign/formulation |
| swing_phase | — | 3.0 (gated) | Different formulation |
| stance_phase | — | 2.0 (gated) | Different formulation |
| foot_clearance | — | -2.0 (gated) | Holosoma embeds in feet_phase |
| self_collisions | — | -1.0 | We have this |

### Observations

| Term | Holosoma | Colosseum | Notes |
|------|----------|-----------|-------|
| base_ang_vel | ✓ | ✓ | |
| projected_gravity | ✓ | ✓ | |
| joint_pos | ✓ (29D) | ✓ (23D) | |
| joint_vel | ✓ (29D) | ✓ (23D) | |
| actions | ✓ | ✓ | |
| command | ✓ (3D) | ✓ (4D twist) | |
| sin_phase / cos_phase | ✓ **never zeroed** | ✓ **zeroed when standing** | Critical difference |
| base_lin_vel | critic only | critic only | Same |
| foot_height | ✓ | critic only | Holosoma puts in actor |

### Command / Curriculum

| | Holosoma | Colosseum |
|---|---|---|
| Standing fraction | `stand_prob=0.2` (20%) | `rel_standing_envs=0.1` (10%) |
| Phase during standing | Frozen at π | Stops advancing, then blanked |
| Curriculum | Penalty scaling 0.1→1.0 | Velocity range expansion |

---

## 4. Proposed Fixes (Ordered by Expected Impact)

### Fix 1 — Stop Zeroing the Phase Observation (High Impact)

**Problem**: `GaitPhaseCommand.command` returns `[0,0,0,0]` when standing. The policy
can't distinguish "standing with feet on ground" from "standing with feet in the air."

**Fix**: Instead of zeroing the output, freeze the phase at `(0, π)` for standing environments.
The internal phase already stops advancing; we just stop blanking the observation:

```python
@property
def command(self) -> torch.Tensor:
    phase_cmd = torch.cat([torch.cos(self._phase), torch.sin(self._phase)], dim=-1)
    return phase_cmd  # no gating — internal phase already frozen when standing
```

And in `_resample_command`, always reset to `(0, π)` so phases start at a known stance-ready value.

This gives the policy a consistent standing signal: `[cos(0), cos(π), sin(0), sin(π)] = [1, -1, 0, 0]`.

> **Why not π for both feet?** Holosoma snaps both to π. At π: `κ = (1+cos(π))/2 = 0` — both
> feet in "swing" of the phase cycle, but the Bézier height is 0. Our κ-based rewards use
> `(1-κ)` for swing penalty: if κ=0 (phase=π), swing penalty is maximized unless feet are
> airborne. This would perversely reward lifting feet when standing. We should keep `(0, π)`
> offset — at phase=0: `κ=1` (full stance), at phase=π: `κ=0` but this encodes a natural
> alternating-stance signal. See Fix 2 for how to handle this in the reward.

### Fix 2 — Ungate Phase Rewards (High Impact)

**Problem**: Gating `swing_phase` and `stance_phase` off removes all foot-timing feedback.

**Fix**: Remove the velocity gate from `swing_phase_schedule`. The reward formulation
handles standing naturally when the phase is frozen:

- When frozen at `(0, π)`: left foot κ=1 (stance), right foot κ=0 (swing)
- `swing_phase` penalizes: `(1-κ_L)*contact_L + (1-κ_R)*contact_R`
  → left foot contact is free (κ_L=1), right foot contact is penalized? This is wrong for standing.

The issue is that `(0, π)` encodes an asymmetric stance during standing, which will
drive the robot to constantly alternate. Holosoma avoids this by snapping **both** to π,
but their Bézier curve maps π → height=0 (not a swing phase).

**Better fix**: Add a dedicated standing reward instead of modifying the existing phase rewards:

```python
def feet_on_ground_when_standing(
    env,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward both feet on ground when velocity command is zero."""
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :2], dim=-1) < command_threshold).float()
    contact_sensor = env.scene[sensor_name]
    # reward = fraction of feet in contact
    in_contact = (contact_sensor.data.force.norm(dim=-1) > 1.0).float()
    both_feet = in_contact.mean(dim=-1)  # 0.0, 0.5, or 1.0
    return both_feet * standing
```

Keep the existing phase rewards gated as-is, and add this new term with weight ~2.0.

### Fix 3 — Increase `rel_standing_envs` (Medium Impact)

**Problem**: Only 10% of episodes are standing. Low exposure → slow learning.

**Fix**: Raise to 0.2 or even 0.3 early in training. The existing `commands_standing`
curriculum in `src/colosseum/mdp/curriculums.py` supports this — it just needs to be
wired into `cact_cfg.py`. Example staging:

```
step 0:       rel_standing_envs = 0.3
step 50M:     rel_standing_envs = 0.2
step 150M:    rel_standing_envs = 0.1
```

### Fix 4 — Add Alive Bonus (Low-Medium Impact)

**Problem**: During standing episodes, reward baseline drops to ~1.7. This provides weak
signal and can lead to degenerate behavior (robot may "prefer" falling since penalties end).

**Fix**: Add a small constant survival bonus:

```python
"alive": RewardTermCfg(
    func=alive_reward,  # returns torch.ones(N, device=env.device)
    weight=0.5,
),
```

This is independent of all gating and provides a stable baseline throughout training.

### Fix 5 — Add Foot Orientation Penalty (High Impact for the "Elvis Pose")

**This is the specific fix for "foot on edge" / tilted ankle posture.**

Holosoma's `penalty_feet_ori` directly measures foot flatness in world frame by projecting
the gravity vector through each foot's quaternion:

```python
def penalty_feet_ori(env) -> torch.Tensor:
    left_quat  = env.simulator._rigid_body_rot[:, env.feet_indices[0]]
    right_quat = env.simulator._rigid_body_rot[:, env.feet_indices[1]]
    gravity = gravity_vector(env)  # [0, 0, -1] in world frame
    left_gravity  = quat_rotate_inverse(left_quat,  gravity)
    right_gravity = quat_rotate_inverse(right_quat, gravity)
    # xy components are nonzero only when foot is tilted
    return (
        left_gravity[:, :2].square().sum(dim=1).sqrt()
        + right_gravity[:, :2].square().sum(dim=1).sqrt()
    )
```

Weight: **-5.0**. Not gated on velocity — active always.

**Why this catches what we miss**: our `pose` reward penalizes `ankle_roll` deviation from
default in joint space. But foot flatness is a world-frame property. If the hip is rolled
outward, the ankle can be at its default angle and the foot is still tilted. The gravity
projection captures the combined effect of all joints above the ankle — hip roll, hip yaw,
knee — not just the ankle itself.

This is likely the primary cause of the "Elvis" posture:
- Hip roll drifts outward → foot tilts inward → contacts only on inner edge
- Our pose reward penalizes hip_roll deviaton (std=0.15) but weakly
- Nothing tells the policy "the foot itself must be flat on the ground"

**Implementation**: We need to access foot body quaternions via the robot entity. In mjlab:

```python
def foot_orientation_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize feet orientation deviation from flat (world frame)."""
    robot = env.scene[asset_cfg.name]
    # asset_cfg should be resolved with body_names=FOOT_BODY_NAMES (not sites)
    foot_quats = robot.data.body_quat_w[:, asset_cfg.body_ids]  # (N, 2, 4)
    # gravity in world frame = [0, 0, -1]
    gravity_w = torch.zeros(env.num_envs, 3, device=env.device)
    gravity_w[:, 2] = -1.0
    penalty = torch.zeros(env.num_envs, device=env.device)
    for i in range(foot_quats.shape[1]):
        g_local = quat_rotate_inverse(foot_quats[:, i], gravity_w)
        penalty += g_local[:, :2].square().sum(dim=-1).sqrt()
    return penalty
```

We need to identify the foot **body** names (not sites) for `asset_cfg`. Check `JOINT_NAMES` /
`t1_constants.py` for the foot link names — likely `Left_Ankle` / `Right_Ankle` bodies or
the foot plate bodies.

### Fix 6 — Penalize Feet Too Close (Medium Impact for Stance Width)

Holosoma `penalty_close_feet_xy: -10.0` penalizes the lateral distance between feet
falling below 0.15 m (measured perpendicular to the base forward direction):

```python
feet_distance = |cos(yaw)*(y_L - y_R) - sin(yaw)*(x_L - x_R)|
penalty = (feet_distance < 0.15).float()
```

Note: this penalizes feet too **close**, not too far apart. It is relevant if the robot
collapses into a narrow stance (feet crossing or scissoring). For the "legs divaricated"
(too wide) problem, this does not help directly — but the `pose` reward's hip_roll penalty
should handle that.

---

## 5. Contact Sensing: Do We Capture "Foot on Edge" Differently?

Short answer: **no, and that is the gap**.

| Aspect | Holosoma | Colosseum |
|--------|----------|-----------|
| Foot flatness measurement | World-frame gravity projection through foot quat | Nothing (only ankle joint angle in pose reward) |
| Contact sensing | Force sensor (is foot touching?) | Force sensor (same) |
| Ankle roll proxy | Joint angle in pose reward | Joint angle in pose reward |

The contact force sensor tells you **whether** the foot is touching the ground. It does not
tell you **how** — whether the foot is flat, on its heel, on its inner edge, etc. Holosoma
adds `penalty_feet_ori` to close this gap. We rely on the ankle_roll component of the pose
reward, which is an imperfect proxy:

- `ankle_roll` joint angle → only captures ankle contribution to tilt
- `penalty_feet_ori` → captures the aggregate tilt from **all** joints: hip roll, hip yaw, knee
  varus/valgus, ankle roll, ankle pitch combined

If the "Elvis pose" is mostly driven by hip roll outward + ankle adapting (staying near
default angle), our pose reward misses it entirely and only `penalty_feet_ori` catches it.

---

## 6. Implementation Order

| Priority | Fix | Expected impact |
|----------|-----|-----------------|
| 1 | **Fix 5** — `foot_orientation_penalty` (-5.0) | "Elvis pose" / foot on edge — **directly addresses the described problem** |
| 2 | **Fix 2** — `feet_on_ground_when_standing` reward | Walking-in-place at zero velocity |
| 3 | **Fix 3** — Increase `rel_standing_envs` to 0.3 | More standing exposure |
| 4 | **Fix 1** — Ungate phase observation | Phase information during standing |
| 5 | **Fix 4** — Alive bonus | Reward baseline stability |
| 6 | **Fix 6** — Feet proximity penalty | Narrow/crossing stance |

The Elvis pose fix (Fix 5) should be attempted first because it directly targets the
described symptom. Fixes 2–3 target the walking-in-place problem. Fixes 1, 4, 6 are
secondary improvements.
