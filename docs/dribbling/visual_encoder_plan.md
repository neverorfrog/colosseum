# Visual Encoder Integration Plan

## Context

We want to add two RMA-style encoders to the dribbling task so the policy can
eventually run without ground-truth (GT) privileged information:

1. **Visual depth encoder** — replaces GT ball position with a latent extracted
   from head-mounted depth camera frames.
2. **Proprioceptive adaptation module** — replaces GT physics parameters
   (ball mass, ball friction) with a latent extracted from observation history
   (same idea as TUM ADLR's 1D-CNN adaptation module).

We will try **two training approaches** and compare:

- **Approach A (Classical RMA)**: Sequential — train policy with privileged
  encoders first, then train adaptation modules to match.
- **Approach B (Joint Training)**: Train encoders jointly with the policy via
  auxiliary losses (as described in the symposium doc).

Both share the same encoder architectures and observation layout. They differ
only in *when* the encoders are trained and *how* their gradients flow.

---

## Current State (What Exists)

**Already implemented:**

| Component | Location | Notes |
|---|---|---|
| `ball_projection(env, camera_name)` | `tasks/dribbling/mdp/rewards.py:272` | Computes (nx, ny) normalized camera coords |
| `_project_ball_to_camera(...)` | `tasks/dribbling/mdp/rewards.py:225` | Full projection math, returns (nx, ny, in_front) |
| `head_ball_tracking` reward | `tasks/dribbling/mdp/rewards.py:295` | Cosine similarity cam_fwd vs ball direction |
| Head camera in scene | `t1_dribbling_cfg.py:49` | `with_head_camera=True` already set |
| Ball obs (GT) | `observation_cfg.py:56-74` | ball_pos(2D), ball_vel(3D), ball_mass(1D), ball_friction(1D) |
| PPO implementation | `algorithm/ppo.py` | Custom, extensible — single joint optimizer |
| TUM ADLR reference | `external/tum-adlr-ws26-04/utils/model.py` | 1D-CNN adaptation module + privileged encoder |

**Not yet implemented:**

- Depth camera *sensor* (rendering actual pixels) — the camera body exists in
  the XML but no `CameraSensorCfg` is attached
- Observation history buffer (for proprioceptive adaptation module)
- Any encoder networks
- Auxiliary loss integration in PPO

---

## When Is Depth Rendering Needed?

Depth rendering means MuJoCo Warp's GPU ray-traced pipeline produces actual
64×64 pixel images every step for every env. This is the main throughput cost.

The visual encoder's **input** is depth pixels — there is no shortcut. The
**supervision signal** (nx, ny ball projection) can be computed analytically
from GT, but the encoder must learn to extract information from raw depth.

The proprioceptive adaptation module does NOT need depth — it operates on
observation history only.

| Phase | Approach A (Sequential) | Approach B (Joint) |
|---|---|---|
| Phase 1: Policy training | **NO DEPTH** — privileged encoders use GT | **DEPTH ON** — visual encoder trains jointly |
| Phase 2: Encoder training | **DEPTH ON** — visual encoder needs frames | N/A (single phase) |
| Phase 3: Swap + fine-tune | **DEPTH ON** — visual encoder is active | **DEPTH ON** — swap and fine-tune |

**Bottom line for Approach A:** Phase 1 is the longest phase and runs at full
speed without any camera overhead. Depth is only needed from Phase 2 onward,
where the policy is frozen and fewer iterations are needed.

---

## Observation Space Design

### Current Actor Obs (flat vector)

```
base_ang_vel          3D   IMU (noisy)
projected_gravity     3D   IMU
joint_pos             23D  encoders (noisy)
joint_vel             23D  encoders (noisy)
actions               23D  internal
gait_phase            4D   internal clock (cos_L, cos_R, sin_L, sin_R)
command               2D   ball velocity command
ball_pos              2D   GT body-frame XY  <-- PRIVILEGED
ball_vel_obs          3D   GT world-frame    <-- PRIVILEGED
base_height           1D   GT                <-- PRIVILEGED
ball_mass             1D   GT                <-- PRIVILEGED
ball_friction         1D   GT                <-- PRIVILEGED
foot_ball_contact     1D   sensor (ok)
```

Total: ~87D (approximate, depends on exact sensor dims)

### Target Actor Obs (with encoders)

Replace the 5 privileged terms (ball_pos 2D, ball_vel 3D, base_height 1D,
ball_mass 1D, ball_friction 1D = **8D**) with two encoder latents:

```
base_ang_vel          3D   IMU (noisy)
projected_gravity     3D   IMU
joint_pos             23D  encoders (noisy)
joint_vel             23D  encoders (noisy)
actions               23D  internal
gait_phase            4D   internal clock
command               2D   ball velocity command
ball_projection       2D   (nx, ny) camera coords  <-- NEW (replaces ball_pos)
z_visual              8D   visual encoder latent    <-- NEW
z_adapt               8D   adaptation module latent <-- NEW
foot_ball_contact     1D   sensor (ok)
```

Total: ~100D

### Key Decisions

- **Latent dimension = 8D** for both encoders (not 64D). Ball state is ~3D of
  information; physics params are ~2D. 8D is plenty. TUM ADLR uses 8D for
  their full privileged state. A wider bottleneck risks overfitting.
  Exception: with moving obstacles, z_visual grows to 16D (see Obstacles
  section below).

- **(nx, ny) stays as a direct observation** in addition to z_visual. This
  gives the policy both a raw signal (camera projection, available via YOLO at
  deployment) and a learned representation (depth encoder, robust to occlusion).

- **ball_vel and base_height move to critic-only** for both approaches. The
  actor should not see raw GT values that are unavailable at deployment.

- **ball_mass, ball_friction move to privileged encoder input** (Approach A) or
  are removed from actor obs entirely (Approach B, where the adaptation module
  implicitly estimates them).

### Critic Obs (unchanged)

The critic keeps seeing everything it currently sees (including GT values).
Asymmetric actor-critic is standard practice.

---

## Encoder Architectures

### 1. Privileged Ball Encoder (Phase 1 only, both approaches)

Maps GT ball state to a compact latent that the actor conditions on.
**Does NOT need depth rendering.**

```
Input:  ball_pos_body_xy(2D) + ball_vel_xy(2D) = 4D
        (optionally: ball_projection nx,ny = 2D more → 6D input)

Architecture:
  Linear(4 → 32) → ELU → Linear(32 → 8) → LayerNorm(8)

Output: z_ball_priv ∈ R^8
```

This is discarded after the visual encoder is trained to replace it.

### 2. Privileged Physics Encoder (Phase 1 only, both approaches)

Maps GT physics params to a compact latent.
**Does NOT need depth rendering.**

```
Input:  ball_mass(1D) + ball_friction(1D) + base_height(1D) = 3D
        (optionally: ground_friction if domain-randomized)

Architecture:
  Linear(3 → 16) → ELU → Linear(16 → 8)

Output: z_phys_priv ∈ R^8
```

Discarded after the adaptation module replaces it.

### 3. Visual Depth Encoder (Phase 2/3, replaces privileged ball encoder)

Maps 5 depth frames from the head camera to a latent matching z_ball_priv.
**REQUIRES depth rendering.**

```
Input: 5 depth frames [N, 5, 64, 64] float32, clipped to 6m, normalized [0,1]

Per-frame CNN (shared weights):
  Conv2d(1→32, k=3, s=2) → BN → ReLU     # 32×32
  Conv2d(32→64, k=3, s=2, p=1) → BN → ReLU   # 16×16
  Conv2d(64→128, k=3, s=2, p=1) → BN → ReLU  # 8×8
  AdaptiveAvgPool2d(1) → Flatten           # 128D per frame

Temporal:
  LSTM(input=128, hidden=64, num_layers=1, batch_first=True)
  Linear(64 → 8)                           # match z_ball_priv dim

Auxiliary projection head (training only, discarded at deployment):
  Linear(8 → 16) → ReLU → Linear(16 → 2)  # predicts (nx, ny)

Output: z_visual ∈ R^8
```

### 4. Proprioceptive Adaptation Module (Phase 2/3, replaces privileged physics encoder)

1D-CNN over observation history, following TUM ADLR architecture.
**Does NOT need depth rendering.**

```
Input: [N, T, obs_dim] stacked proprioceptive obs
       T = 50 timesteps (1 second at 50Hz policy rate)
       obs_dim = proprioceptive subset (ang_vel 3 + gravity 3 + joint_pos 23
                 + joint_vel 23 + actions 23 = 75D)

Architecture (transposed to [N, obs_dim, T] for Conv1d):
  Conv1d(75 → 32, k=3, padding=1) → ELU
  Conv1d(32 → 16, k=3, padding=1) → ELU
  AdaptiveAvgPool1d(1) → Flatten   # 16D
  Linear(16 → 8)                   # match z_phys_priv dim

Output: z_adapt ∈ R^8
```

---

## Approach A: Classical RMA (Sequential)

This is the standard Kumar et al. (RSS 2021) approach. Clean separation of
concerns: the policy never changes between phases.

### Phase 1 — Train Policy with Privileged Encoders

> **Depth rendering: OFF**

**What trains:** actor, critic, privileged_ball_encoder, privileged_phys_encoder

**Actor input:**
```
[proprio(75D), gait_phase(4D), command(2D), ball_proj_nxny(2D),
 z_ball_priv(8D), z_phys_priv(8D), foot_ball_contact(1D)]  = 100D
```

where:
- `z_ball_priv = privileged_ball_encoder(ball_pos_xy, ball_vel_xy)`
- `z_phys_priv = privileged_phys_encoder(ball_mass, ball_friction, base_height)`

**Critic input:** current critic obs (everything including raw GT).

**Training:** Standard PPO. The privileged encoders are part of the actor's
computation graph — their parameters are in the actor optimizer. No auxiliary
losses needed. No camera sensor in the scene.

**Exit condition:** Policy achieves target dribbling performance (ball velocity
tracking, fall rate, etc.).

**What to save:** actor, critic, privileged_ball_encoder, privileged_phys_encoder,
normalizers.

### Phase 2 — Train Adaptation Modules (Policy Frozen)

> **Depth rendering: ON** (visual encoder needs depth frames)

**What trains:** visual_depth_encoder, adaptation_module (only)

**What's frozen:** actor, critic, privileged encoders (used as regression targets)

**Procedure:**
1. Enable `HEAD_DEPTH_CAMERA` sensor in the scene config.
2. Run the frozen policy in simulation, collecting rollouts.
3. At each step, record:
   - Depth frames from head camera → visual encoder input **(needs depth)**
   - Stacked proprioceptive obs → adaptation module input (no depth needed)
   - z_ball_priv (from frozen privileged encoder) → visual encoder target
   - z_phys_priv (from frozen privileged encoder) → adaptation module target
   - (nx, ny) from GT → auxiliary projection head target
4. Train encoders with:

```
L_visual = MSE(z_visual, z_ball_priv.detach())
         + λ_proj * MSE(projection_head(z_visual), nxny_gt) * in_front_mask

L_adapt  = MSE(z_adapt, z_phys_priv.detach())

L_total  = L_visual + L_adapt
```

**Training variant — online vs offline:**
- *Online*: Run frozen policy, train encoders every step with live data. Covers
  full state distribution naturally.
- *Offline*: Collect a dataset of (depth, obs_history, targets) first, then
  train encoders on the dataset. Simpler but covers a fixed distribution.

Recommend **online** — the policy visits diverse states during rollouts, and
you don't need to manage a dataset.

**Optimizer:** Separate Adam for encoder parameters. LR ~1e-3 to 1e-4.

**Exit condition:** `MSE(z_visual, z_ball_priv) < threshold` and
`projection_head accuracy < 5px equivalent` (i.e., MSE on nx,ny < 0.05).

### Phase 3 — Swap and Optionally Fine-tune

> **Depth rendering: ON** (visual encoder is now the live input source)

**Swap:** Replace `z_ball_priv = privileged_ball_encoder(GT)` with
`z_visual = visual_encoder(depth_buffer)` in the actor obs. Same for
`z_phys_priv → z_adapt`. Same 8D dimension, similar distribution.

**Fine-tune (optional):** Unfreeze actor, run PPO for a few hundred iterations
with the swapped encoders. The encoder parameters can be frozen or trained with
a reduced LR (`η_enc = η_policy / 100`).

**Pros of Approach A:**
- Clean, well-understood pipeline
- Phase 1 is fast (no depth rendering overhead)
- Debugging is easy: each phase has a clear success metric
- The actor's input distribution doesn't change between phases

**Cons of Approach A:**
- Total training time is longer (3 sequential phases)
- The visual encoder only sees the frozen policy's behavior — if the policy
  would benefit from different behavior given visual input, it can't adapt
- Requires careful Phase 2 exit criteria

---

## Approach B: Joint Training (Symposium Doc)

Train encoders alongside the policy from the start, using auxiliary losses.

### Single Phase — Joint Policy + Encoder Training

> **Depth rendering: ON from the start** (visual encoder needs depth at every step)

**What trains:** actor, critic, privileged_ball_encoder, privileged_phys_encoder,
visual_depth_encoder, adaptation_module — all simultaneously.

**Actor input:** Same 100D as Approach A Phase 1. The actor always sees
z_ball_priv and z_phys_priv from the privileged encoders.

**Auxiliary losses (applied every learning step):**
```
L_PPO     = surrogate_loss + value_coef * value_loss - entropy_coef * entropy

L_visual  = MSE(z_visual, z_ball_priv.detach())
          + λ_proj * MSE(projection_head(z_visual), nxny_gt) * in_front_mask

L_adapt   = MSE(z_adapt, z_phys_priv.detach())

L_total   = L_PPO + λ_v * L_visual + λ_a * L_adapt
```

**Key hyperparameters (from the doc):**
- `λ_v ≈ 0.1–0.5` for visual loss (1.0 during warm-up)
- `λ_a ≈ 0.1–0.5` for adaptation loss
- `η_enc = η_policy / 100` (separate, slower LR for encoder params)
- Separate gradient clipping on encoder parameters
- **Warm-up:** `λ_v = 1.0` for first 50 iterations, then ramp PPO. This lets
  the visual encoder get a head start before the policy's distribution shifts.

**After joint training converges:** Swap z_ball_priv → z_visual and
z_phys_priv → z_adapt. Fine-tune briefly. **(Depth still ON.)**

**Pros of Approach B:**
- Single training run (no phase management)
- Encoders see a continuously evolving policy distribution
- The doc's warm-up schedule is well-tuned

**Cons of Approach B:**
- **Depth rendering throughout training** — significantly slower per step
- More hyperparameters to tune (λ_v, λ_a, η_enc, warm-up schedule)
- Harder to debug: if training fails, is it the policy, the encoder, or the
  auxiliary loss?
- Requires modifications to the PPO training loop

---

## Obstacles

The depth encoder sees everything in the scene. Obstacles affect what
information the latent must carry and how the supervision works.

### No Obstacles (Current Baseline)

The depth image contains: ground plane + robot self-occlusion + ball.

- z_visual (8D) encodes ball position/velocity only
- Auxiliary supervision: (nx, ny) ball projection
- Simple, start here

### Fixed Obstacles (Goalposts, Walls, Static Objects)

Fixed obstacles appear as static structures in every depth frame.

**Impact on the encoder:**
- The CNN must learn to distinguish ball (small, round, moves) from obstacles
  (large, static). This is learnable — fixed geometry becomes background that
  the CNN filters out over training.
- The LSTM helps: ball moves across frames, obstacles don't.
- **No architecture change needed.** The 8D latent still encodes ball state
  only. Fixed obstacles are clutter, not information the policy needs from the
  visual encoder (the policy already knows obstacle positions from the scene).

**Impact on training:**
- Training distribution is richer (more visual variety), which may slow
  convergence slightly but improves generalization.
- The auxiliary (nx, ny) loss is unchanged — it still supervises ball tracking.
- **Depth rendering cost unchanged** — obstacles are just more geometry for
  the ray tracer, negligible overhead.

**What to do:** Add obstacles to the scene during encoder training (Phase 2+
for Approach A, always for Approach B). No encoder or loss changes.

### Moving Obstacles (Opponent Robots)

Moving obstacles create **ambiguity**: the depth image now contains multiple
moving entities, and the encoder must figure out which blob is the ball vs.
an opponent.

**Impact on the encoder:**

1. **Ball-opponent discrimination in depth:** Possible by shape (ball is small
   and round, opponent is tall and humanoid) and motion pattern (different
   dynamics over the 5-frame window). Depth-only is harder than RGB for this,
   but the LSTM temporal processing helps.

2. **Occlusion by opponents:** The ball can be hidden behind an opponent. This
   is the primary motivation for the depth encoder over YOLO — the encoder can
   maintain a latent estimate through short occlusions because the LSTM carries
   temporal state.

3. **The latent must encode more:** With opponents, the policy benefits from
   knowing not just "where is the ball" but also "where is the nearest
   opponent" and "is the ball currently occluded." This argues for a wider
   latent:

```
z_visual (ball-only):     8D  — ball position + velocity
z_visual (with opponents): 16D — ball state + opponent state + occlusion
```

**Architecture changes for opponents:**

```
# Only the output layer and auxiliary head change:

Temporal (updated):
  LSTM(input=128, hidden=128, num_layers=1, batch_first=True)  # wider hidden
  Linear(128 → 16)                                              # 16D latent

Auxiliary projection head (updated):
  Linear(16 → 32) → ReLU → Linear(32 → 6)
  # predicts: (nx_ball, ny_ball, nx_opp, ny_opp, ball_visible, opp_dist)
```

The CNN backbone stays identical — only the LSTM hidden size and output linear
change.

**Supervision changes:**

```
L_visual = MSE(z_visual, z_ball_opp_priv.detach())
         + λ_proj * MSE(proj_head(z_visual)[:, :2], nxny_ball_gt) * in_front_mask
         + λ_opp  * MSE(proj_head(z_visual)[:, 2:4], nxny_opp_gt) * opp_visible_mask
         + λ_vis  * BCE(proj_head(z_visual)[:, 4], ball_visible_gt)
```

The privileged ball encoder also grows to accept opponent state:

```
Privileged encoder input (with opponents):
  ball_pos_xy(2D) + ball_vel_xy(2D) + opp_pos_xy(2D) + opp_vel_xy(2D) = 8D
  → Linear(8 → 32) → ELU → Linear(32 → 16) → LayerNorm(16)
  → z_ball_opp_priv ∈ R^16
```

**Impact on training:**
- **Depth rendering still only needed when training the visual encoder**
  (Phase 2+ for Approach A). The opponent's position is GT in Phase 1.
- Training is harder — the encoder must solve a harder perception problem.
  Expect to need more Phase 2 iterations.
- Consider a **curriculum**: start with ball-only scenes, then introduce
  stationary opponents, then moving opponents.

**Recommended approach:**
1. Get ball-only working first (8D latent, current plan).
2. Add fixed obstacles — no encoder changes, just richer scenes.
3. Add opponents — widen latent to 16D, add opponent auxiliary supervision.

Each step is a clean extension of the previous one. The CNN backbone never
changes.

---

## Depth Rendering Cost and Mitigations

Rendering 4096 envs × 64×64 depth at 50Hz is the biggest throughput concern.
mjlab uses MuJoCo Warp's GPU ray-traced pipeline (batch rendering across all
envs is native), so there is no CPU bottleneck, but GPU compute is still
significant.

**Mitigations (ranked by impact):**

1. **Approach A Phase 1 without camera** — the single biggest win. The longest
   training phase runs at full speed. Depth only needed in Phase 2 where the
   policy is frozen and fewer total iterations are needed.

2. **Subsample envs:** Render depth for only a subset of envs (e.g., 512 of
   4096). The visual encoder trains on the subset; the rest use the privileged
   encoder. Both produce the same 8D latent so the actor doesn't care.

3. **Render every N steps:** Depth at 50Hz may be overkill for a ball that
   moves relatively slowly. Render every 2–4 policy steps (12.5–25Hz) and
   repeat the last frame in the buffer. The LSTM can handle the lower temporal
   resolution.

4. **Reduce resolution:** 32×32 instead of 64×64. Ball at 1m is ~3–5px at
   32×32 — borderline but may work. Test both. Going from 64×64 to 32×32 is a
   4× reduction in pixel count.

5. **Depth-only, no textures/shadows:** Already planned. The sensor config
   uses `data_types=("depth",)` with `use_textures=False, use_shadows=False`.
   This is the cheapest rendering mode.

---

## Implementation Plan

### Step 0: Observation Refactoring (shared by both approaches)

Before either approach, restructure observations:

**Move privileged terms out of actor obs and into separate groups:**

```python
# observation_cfg.py

# Proprioceptive (available at deployment)
proprio_terms = {
  "base_ang_vel": ...,
  "projected_gravity": ...,
  "joint_pos": ...,
  "joint_vel": ...,
  "actions": ...,
  "gait_phase": ...,
  "command": ...,
  "ball_projection": ObservationTermCfg(    # NEW: (nx,ny) from camera
    func=ball_projection,
    params={"camera_name": "robot/d455_color"},
  ),
  "foot_ball_contact_force": ...,
}

# Privileged ball state (GT, for encoder supervision)
privileged_ball_terms = {
  "ball_pos": ...,       # 2D body-frame XY
  "ball_vel": ...,       # 2D XY (reduce from 3D)
}

# Privileged physics params (GT, for encoder supervision)
privileged_phys_terms = {
  "base_height": ...,    # 1D
  "ball_mass": ...,      # 1D
  "ball_friction": ...,  # 1D
}

observations = {
  "actor": ObservationGroupCfg(terms=proprio_terms, ...),
  "critic": ObservationGroupCfg(terms={**proprio_terms, **critic_extra}, ...),
  "privileged_ball": ObservationGroupCfg(terms=privileged_ball_terms, ...),
  "privileged_phys": ObservationGroupCfg(terms=privileged_phys_terms, ...),
}
```

The actor no longer directly sees GT ball_pos, ball_vel, ball_mass, etc.
Instead, the PPO algorithm passes them through encoders and concatenates the
latents before feeding to the actor network.

**Files to modify:**
- `tasks/dribbling/config/t1_23dof/observation_cfg.py` — restructure groups
- `tasks/dribbling/mdp/observations.py` — add `ball_projection` as an obs term
  (move from rewards.py or import)

### Step 1: Encoder Networks

**New file:** `src/colosseum/algorithm/encoders.py`

Contains:
- `PrivilegedEncoder(nn.Module)` — generic small MLP, used for both ball and
  physics privileged encoders
- `VisualDepthEncoder(nn.Module)` — CNN-LSTM for depth frames
- `ProprioceptiveAdaptationModule(nn.Module)` — 1D-CNN over obs history

These are algorithm-level modules (not task-specific) because they plug into
the PPO training loop.

### Step 2: Depth Camera Sensor

**New sensor config** in `robots/t1_23dof/sensors.py`:
```python
HEAD_DEPTH_CAMERA = CameraSensorCfg(
  name="head_depth_camera",
  camera_name="robot/d455_color",
  width=64, height=64,
  data_types=("depth",),
  use_textures=False,
  use_shadows=False,
)
```

**Add to scene** in `t1_dribbling_cfg.py` sensors tuple — but only when depth
rendering is needed:
- Approach A: OFF in Phase 1 config, ON in Phase 2/3 config
- Approach B: ON always

Could be controlled by a boolean flag in the env config or by having two scene
config variants.

### Step 3: Depth Frame Buffer + Obs History Buffer

**Modify** `tasks/dribbling/env.py` — add:

1. **Depth frame buffer** `[N, 5, 64, 64]` — rolling buffer of last 5 depth
   frames. Updated after each env step. Reset on episode done.
   **Only allocated when depth camera sensor is present in the scene.**

2. **Proprioceptive obs history buffer** `[N, 50, proprio_dim]` — rolling
   buffer of last 50 proprioceptive observations. Updated after each env step.
   Reset on episode done.

These buffers are *not* part of the observation manager — they're env-level
state that the PPO algorithm reads directly (similar to how the TUM ADLR
`ObservationsWrapper` works).

### Step 4: RMA-PPO Algorithm Variant

**New file:** `src/colosseum/algorithm/rma_ppo.py`

Subclass of `PPO` that adds encoder handling. This is where the two approaches
diverge:

```python
class RmaPPO(PPO):
  """PPO with RMA-style privileged encoders and adaptation modules."""

  def _build_networks(self):
    super()._build_networks()
    # Build privileged encoders
    self.ball_encoder = PrivilegedEncoder(input_dim=4, latent_dim=8)
    self.phys_encoder = PrivilegedEncoder(input_dim=3, latent_dim=8)
    # Build adaptation modules (for Phase 2 / joint training)
    self.visual_encoder = VisualDepthEncoder(latent_dim=8)
    self.adapt_module = ProprioAdaptationModule(obs_dim=75, latent_dim=8)

  def _build_optimizers(self):
    # Separate optimizer groups for policy vs encoders
    ...

  def _compose_actor_obs(self, proprio_obs, priv_ball_obs, priv_phys_obs):
    """Concatenate proprio + encoder latents → actor input."""
    z_ball = self.ball_encoder(priv_ball_obs)
    z_phys = self.phys_encoder(priv_phys_obs)
    return torch.cat([proprio_obs, z_ball, z_phys], dim=-1)

  def _learning_step(self):
    """Override to add auxiliary encoder losses (Approach B only)."""
    ...
```

**For Approach A:** RmaPPO Phase 1 just uses `_compose_actor_obs` — no
auxiliary losses. Phase 2 is a separate script that loads the checkpoint,
freezes everything, and trains only the adaptation modules.

**For Approach B:** RmaPPO adds auxiliary losses in `_learning_step`:
```python
loss = ppo_loss + λ_v * visual_loss + λ_a * adapt_loss
```

### Step 5 (Approach A only): Phase 2 Training Script

**New file:** `src/colosseum/scripts/train_encoders.py`

Loads Phase 1 checkpoint, freezes policy + privileged encoders, runs rollouts
to collect (depth, obs_history, z_priv_targets), trains visual_encoder and
adapt_module online. **Depth rendering is ON in this script.**

### Step 6: Phase 3 / Deployment Swap

After encoders are trained, modify `_compose_actor_obs` to use:
- `z_visual = visual_encoder(depth_buffer)` instead of `ball_encoder(GT)`
  **(needs depth rendering)**
- `z_adapt = adapt_module(obs_history)` instead of `phys_encoder(GT)`
  (no depth needed)

---

## What NOT to Build Yet

- **Contact encoder** (the doc's 36D force/motion → 16D latent). We're using
  the simpler proprioceptive adaptation module instead. Contact encoder can be
  added later if needed.

- **Adversarial opponent / shielding** (Week 3-4 of the doc). Out of scope —
  this plan covers the encoder integration only.

- **Teacher-student distillation** (DAgger + N-P3O). That's a separate effort
  after encoders work.

- **Ball position noise curriculum.** The doc ramps σ from 0.02→0.08m. This
  is relevant for Approach B but not critical for a first implementation. Add
  after basic joint training works.

---

## Recommended Order

1. **Start with Approach A** — it's simpler, faster to iterate, and easier to
   debug. Phase 1 is the current training with minor obs restructuring.
   **No depth rendering needed.**

2. **Validate Phase 1** — confirm the policy works well with privileged
   encoder latents (8D) instead of raw GT values. This is the critical test:
   if the policy can't learn with the bottleneck, neither approach will work.

3. **Implement Phase 2** — train visual encoder + adaptation module. **This is
   where depth rendering turns on.** Discover if 64×64 depth is sufficient,
   if the LSTM helps, if 8D is enough, etc.

4. **Add fixed obstacles to the scene** — no encoder changes, just richer
   training distribution. Validates robustness.

5. **If Approach A works well**, try Approach B for comparison. The encoder
   architectures and obs layout are identical — only the training loop differs.

6. **Add opponent support (later)** — widen z_visual to 16D, add opponent
   auxiliary supervision. This is a clean extension.

---

## Open Questions

1. **obs_dim mismatch:** The actor network input dim changes from ~87D to
   ~100D when adding encoder latents. This means Phase 1 checkpoints are not
   directly loadable into a non-RMA PPO. The RmaPPO subclass must handle this.

2. **Observation normalization:** The privileged encoder outputs (z_ball,
   z_phys) should probably NOT be normalized by the obs normalizer (they're
   already learned representations). Need to handle this in the normalizer or
   in `_compose_actor_obs`.

3. **50-step obs history memory:** At 4096 envs × 50 steps × 75D float32 =
   ~60MB. Manageable, but worth noting.

4. **Depth rendering throughput:** mjlab uses MuJoCo Warp GPU ray-tracing
   (batch rendering is native). No CPU bottleneck, but GPU compute for
   4096 × 64×64 per step needs benchmarking. The mitigations above (env
   subsampling, lower freq, lower res) are available if needed.
