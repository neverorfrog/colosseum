# RMA Visual Encoder — Architecture and Implementation

Two-phase training for the dribbling task. The policy learns using ground-truth
privileged information (Phase 1), then a visual encoder is trained to replace
it with depth camera observations (Phase 2). At inference the two encoders are
interchangeable — the actor input dimension never changes.

---

## Observation Space

### Actor Obs (proprioceptive only)

The actor never sees raw privileged state directly. Encoder latents are
concatenated by `RmaPPO`, not by the observation manager.

```
base_ang_vel          3D   IMU (noisy)
projected_gravity     3D   IMU
joint_pos             23D  encoders (noisy)
joint_vel             23D  encoders (noisy)
actions               23D  last action
gait_phase            4D   (cos_L, cos_R, sin_L, sin_R)
command               2D   ball velocity command
foot_ball_contact     1D   sensor
─────────────────────────────────────────────
+ z_ball              8D   encoder latent  ← appended by RmaPPO
```

### Privileged Ball Obs (`privileged_ball` group)

```
ball_pos_xy           2D   body-frame GT position
ball_vel_xy           2D   GT velocity
─────────────────────────────────────────────
Total:                4D   → input to PrivilegedEncoder (Phase 1)
                           → regression target for DepthEncoder (Phase 2)
```

### Critic Obs (asymmetric, GT everywhere)

Actor terms + `privileged_ball` + base height, ball mass, ball friction, base
linear velocity, foot height/air-time/contact/forces. Critic is unchanged
across phases.

---

## Encoder Networks (`src/colosseum/algorithm/encoders.py`)

### PrivilegedEncoder

Small MLP. Maps GT privileged obs → latent. Used in Phase 1 and as the
frozen regression target in Phase 2.

```
Linear(input_dim → 32) → ELU → Linear(32 → latent_dim) → LayerNorm
```

`input_dim` is inferred from the obs manager's `group_obs_dim` at construction
time. `latent_dim = 8` (default).

### DepthEncoder

Per-frame CNN + LSTM. Maps a rolling window of depth frames → latent. Designed
to produce the same shape as `PrivilegedEncoder` so they are drop-in
replacements.

```
Input:  (B, T, 1, H, W) — T=5 frames, normalized to [0, 1]

Per-frame CNN (shared weights):
  Conv2d(1→32, k=5, s=2) → BN → ReLU
  Conv2d(32→64, k=3, s=2) → BN → ReLU
  Conv2d(64→128, k=3, s=2) → BN → ReLU
  AdaptiveAvgPool2d(1) → Flatten         # 128D per frame

Temporal:
  LSTM(input=128, hidden=lstm_hidden=64, batch_first=True)
  Linear(64 → latent_dim)

Output: (B, latent_dim)
```

`AdaptiveAvgPool2d` means any spatial resolution ≥ 16×16 works without
changing the architecture.

### ProjectionHead

Auxiliary supervision head used during Phase 2 training only. Discarded at
deployment.

```
Linear(latent_dim → 32) → ReLU → Linear(32 → 2)
Output: (nx, ny) camera-space ball projection in [-1, 1]²
```

---

## RMA Term and Manager

### `BallRmaTermCfg` / `BallRmaTerm` (`src/colosseum/tasks/dribbling/mdp/rma_terms.py`)

Task-level term that pairs `PrivilegedEncoder` and `DepthEncoder` for the ball
latent slot. Registered in `RmaBasedEnvCfg.encoders` as `{"ball": BallRmaTermCfg(...)}`.

Key config fields:

| Field | Default | Description |
|---|---|---|
| `privileged_obs_group` | `"privileged_ball"` | Obs group fed to the privileged encoder |
| `adaptation_obs_group` | `None` | Obs group fed to the depth encoder (set to `"depth_frames"` in Phase 2) |
| `latent_dim` | `8` | Shared output dimension |
| `sensor_name` | `"head_rgbd"` | Scene sensor to read depth frames from |
| `seq_len` | `5` | Depth frame buffer length |
| `height` / `width` | `72` / `128` | Frame size after bilinear downsampling |
| `depth_clip` | `6.0 m` | Depth clip range; frames normalised to [0, 1] |
| `camera_name` | `"robot/d455_color"` | Camera for GT ball-in-FOV projection |
| `camera_fovy` | `60°` | Vertical FOV |

**`update()` (called every env step):**
1. Reads raw depth from `head_rgbd` sensor.
2. Bilinearly downsamples to `(height, width)`, clips, normalises.
3. Projects ball to camera space using `cam_xpos` / `cam_xmat` (accounts for
   neck joints and head tracking).
4. Computes `ball_in_fov` (in front of camera, within horizontal and vertical
   FOV limits, within `depth_clip`).
5. On FOV re-entry (out→in transition): zeros the depth buffer so the LSTM
   always starts from a clean slate.
6. Rolls the depth buffer: drops oldest frame, appends newest.
7. Tracks `frames_since_entry` per env (capped at `seq_len`).

**`get_adaptation_mask()` → `(N,) bool | None`:**

Returns `ball_in_fov AND frames_since_entry >= seq_len`. An env is "valid" only
when the ball is in view *and* the buffer has been fully populated since the
last entry. Returns `None` during Phase 1 (camera absent, `update()` is a no-op).

**`get_current_adaptation_obs()` → `{"depth_frames": (N, T, 1, H, W)}`:**

Snapshot of the current depth buffer. Called immediately after `env.step()` so
the snapshot is temporally aligned with the privileged obs from the same step.

### `RmaManager` (`src/colosseum/managers/rma_manager.py`)

Holds all `RmaTerm` instances. Mirrors mjlab's manager pattern.

Key methods:

| Method | Description |
|---|---|
| `encode(obs_dict, phase=1)` | Run all terms with privileged (1) or adaptation (2) encoders; returns `(N, total_latent_dim)` |
| `get_adaptation_obs()` | Snapshot current adaptation obs from all terms |
| `get_adaptation_mask()` | AND of per-term masks |
| `encode_phase2_with_fallback(priv_obs, adapt_obs)` | Per-env blend: use depth encoder for valid envs, frozen privileged encoder for invalid envs |
| `compute_adaptation_loss(priv_obs, adapt_obs, mask)` | MSE of frozen privileged target vs adaptation prediction; mask-aware |
| `privileged_parameters()` / `adaptation_parameters()` | Selective parameter iterators for optimizers |
| `state_dict()` / `load_state_dict()` | Nested `{term_name: {privileged: ..., adaptation: ...}}` |

---

## Algorithm (`src/colosseum/algorithm/rma_ppo.py`)

`RmaPPO` subclasses `PPO`. All RMA-specific logic is here; the base `PPO` and
`BaseAlgorithm` are untouched.

### Phase routing

```
_phase = 1  →  train() delegates to standard PPO loop
_phase = 2  →  train() runs _train_phase2()
```

`train_phase2.py` sets `_phase = 2` before calling `algo.train()`.

### Actor input composition (`_compose_actor_input`)

```python
# Training (always phase 1 path inside rollout):
z = rma_manager.encode(privileged_obs)

# Inference / play:
if _inference_phase == 2:
    adapt_obs = rma_manager.get_adaptation_obs()
    z = rma_manager.encode_phase2_with_fallback(privileged_obs, adapt_obs)
else:
    z = rma_manager.encode(privileged_obs)

return cat([actor_obs, z], dim=-1)
```

### Phase 1 — joint PPO training

Single Adam optimizer for actor + critic + all encoder parameters.

`_build_rollout_buffer` pre-allocates per-group privileged obs storage so
`privileged_obs` is available in the mini-batch during the learning step.

### Phase 2 — adaptation encoder regression (`_phase2_learning_step`)

Collect and learn are **coupled in the same step** to guarantee temporal
alignment between privileged obs and adaptation obs snapshots:

1. **Collection** (all networks frozen, `torch.no_grad()`):
   - Run the frozen policy for `num_steps_per_env` steps.
   - After each `env.step()`, snapshot `(privileged_obs, adaptation_obs, adaptation_mask)` from the current physics state.

2. **Learning** (only adaptation encoder params have grad):
   - Stack snapshots → flat batches.
   - Shuffle and split into mini-batches.
   - For each mini-batch: `compute_adaptation_loss(priv_batch, adapt_batch, mask_batch)` → backward → `_adaptation_optimizer.step()`.

### Optimizer setup for Phase 2 (`build_adaptation_optimizer`)

- Freezes actor, critic, and all privileged encoder parameters.
- Builds a separate `Adam` optimizer over `rma_manager.adaptation_parameters()` only.
- Called by `train_phase2.py` before `algo.train()`.

### Checkpoint auto-detection

`load()` checks `checkpoint["metadata"]["phase"]`. If `== 2`, sets
`_inference_phase = 2` automatically. Play scripts need no extra flags.

### `RmaPPOConfig` (`src/colosseum/config/types/algorithm.py`)

Extends `PpoConfig` with:

```python
inference_phase: int = 1
# 1 = privileged encoder (GT, default)
# 2 = adaptation encoder (depth CNN+LSTM)
```

Can be overridden from the CLI (`--task.algo-cfg.inference-phase 2`) to force
a specific encoder independently of the checkpoint's metadata.

---

## Depth Camera and Env Config (`src/colosseum/tasks/dribbling/config/t1_23dof/t1_dribbling_cfg.py`)

`use_depth_camera: bool` is a CLI-accessible field on `T1DribblingTask`. It
controls whether `HEAD_RGBD_SENSOR` is added to the scene and whether
`adaptation_obs_group="depth_frames"` is set on the `BallRmaTermCfg`.

```
Phase 1 (default):  --task.use-depth-camera is absent → camera OFF
Phase 2:            --task.use-depth-camera           → camera ON
Play:               always use_depth_camera=True (single env, negligible cost)
```

`train_env_cfg` property reconstructs the env config with the correct
`use_depth_camera` while preserving `num_envs` from the CLI `env` field.

`HEAD_RGBD_SENSOR` is defined in `src/colosseum/robots/t1_23dof/sensors.py`
(64×64 depth, no textures, no shadows — cheapest MuJoCo Warp render mode).

---

## Training Scripts

### Phase 1

```bash
pixi run -e train train task:t1-dribbling
```

Trains actor, critic, and `PrivilegedEncoder` jointly via PPO.
No depth rendering. Saves checkpoint with `metadata["phase"] = 1`.

### Phase 2

```bash
pixi run -e train train-phase2 \
    --checkpoint ./logs/<run>/checkpoints/latest.pt \
    task:t1-dribbling --task.use-depth-camera
```

`train_phase2.py` (`Phase2Config`) adds:
- `--adaptation-lr` (default `1e-3`) — Adam LR for `DepthEncoder` only.
- `--save-interval` (default `10_000`) — more frequent saves than Phase 1.

Requires `--checkpoint`. Exits early if not provided.
Saves checkpoint with `metadata["phase"] = 2`.

### Play / Evaluation

```bash
pixi run -e train play --checkpoint <ckpt>.pt task:t1-dribbling
```

`play.py` instantiates the algo with the live play env (so
`rma_manager.get_adaptation_obs()` accesses the live depth buffer). After
`load()`, if the checkpoint has `metadata["phase"] == 2`, the policy
automatically uses the `DepthEncoder` at inference time.

---

## File Map

| File | Role |
|---|---|
| `src/colosseum/algorithm/encoders.py` | `PrivilegedEncoder`, `DepthEncoder`, `ProjectionHead` nn.Modules |
| `src/colosseum/managers/rma_manager.py` | `RmaTermCfg`, `RmaTerm` (abstract), `RmaManager` |
| `src/colosseum/tasks/dribbling/mdp/rma_terms.py` | `BallRmaTermCfg`, `BallRmaTerm` (concrete) |
| `src/colosseum/algorithm/rma_ppo.py` | `RmaPPO` — PPO subclass with Phase 1/2 dispatch |
| `src/colosseum/config/types/algorithm.py` | `RmaPPOConfig` with `inference_phase` field |
| `src/colosseum/tasks/dribbling/config/t1_23dof/observation_cfg.py` | Obs groups: `actor`, `critic`, `privileged_ball` |
| `src/colosseum/tasks/dribbling/config/t1_23dof/algo_cfg.py` | Returns `RmaPPOConfig` for the dribbling task |
| `src/colosseum/tasks/dribbling/config/t1_23dof/t1_dribbling_cfg.py` | `use_depth_camera` CLI flag; `train_env_cfg` property |
| `src/colosseum/robots/t1_23dof/sensors.py` | `HEAD_RGBD_SENSOR` definition |
| `src/colosseum/envs/rma_based_env.py` | `RmaBasedEnv` — owns depth/obs buffers, calls `rma_manager.update()` |
| `src/colosseum/scripts/train.py` | Phase 1 training entry point |
| `src/colosseum/scripts/train_phase2.py` | Phase 2 training entry point |
| `src/colosseum/scripts/play.py` | Evaluation / play entry point |
