# z_enc — Depth Encoder Integration Plan

Visual ball state encoder (CNN+LSTM → 64D latent) for the dribbling task, following the paper's architecture.

## Current State

- **mjlab has `CameraSensor`** (`mjlab/sensor/camera_sensor.py`) supporting `"rgb"` and `"depth"` data types, producing `[num_envs, H, W, 1]` float32 tensors via MuJoCo Warp GPU rendering.
- **Head camera exists in the MuJoCo spec** — `get_spec_with_head_camera()` in `src/colosseum/robots/t1_23dof/constants.py:68-83` creates `"d455_color"` on the H2 head body (60° FOV).
- **DONE: RGB-D rendering is now enabled.** The camera spec in `constants.py` uses calibrated D455 intrinsics (fx=646.06, fy=645.20, cx=644.31, cy=357.13) at 1280x720, matching the mjlab `booster_t1_rgbd_camera.py` demo exactly. `HEAD_RGBD_SENSOR` in `sensors.py` wraps it as a `CameraSensorCfg` with `data_types=("rgb", "depth")`, and is included in the dribbling scene's sensor tuple.

## ~~Step A — Enable Depth Rendering in the Scene~~ (DONE)

The sensor is defined in `src/colosseum/robots/t1_23dof/sensors.py`:

```python
HEAD_RGBD_SENSOR = CameraSensorCfg(
    name="head_rgbd",
    camera_name="robot/d455_color",
    width=1280,
    height=720,
    data_types=("rgb", "depth"),
    use_textures=True,
    use_shadows=False,
)
```

Added to `scene_cfg()` in `src/colosseum/tasks/dribbling/config/t1_23dof/t1_dribbling_cfg.py`.

## Step B — Define the Encoder Module

Create `src/colosseum/tasks/dribbling/mdp/depth_encoder.py`:

```python
class DepthEncoder(nn.Module):
    """CNN + LSTM depth encoder → z_enc ∈ ℝ⁶⁴.

    Input:  (B, T, 1, 64, 64) — T=5 depth frames
    Output: (B, 64) — z_enc latent
    """
    def __init__(self, seq_len=5, latent_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 128),
        )
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.head = nn.Linear(256, latent_dim)

    def forward(self, frames: Tensor) -> Tensor:
        B, T, C, H, W = frames.shape
        e = self.cnn(frames.reshape(B * T, C, H, W))  # (B*T, 128)
        e = e.reshape(B, T, -1)                         # (B, T, 128)
        _, (h, _) = self.lstm(e)                         # h: (1, B, 256)
        return self.head(h.squeeze(0))                   # (B, 64)
```

Add a projection head for auxiliary supervision during training (discarded at deployment):

```python
class ProjectionHead(nn.Module):
    """Predicts GT (nx, ny) from z_enc for auxiliary loss."""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, z_enc: Tensor) -> Tensor:
        return self.net(z_enc)  # (B, 2) predicted (nx, ny)
```

## Step C — Integrate into the RL Environment

Extend `DribblingEnv` in `src/colosseum/tasks/dribbling/env.py` to own the encoder, buffer depth frames, and expose `z_enc`:

```python
class DribblingEnv(ViewerCompatibleEnv):
    def __init__(self, cfg, ...):
        super().__init__(cfg, ...)
        self.depth_encoder = DepthEncoder().to(self.device)
        self.projection_head = ProjectionHead().to(self.device)
        self.depth_buffer = None  # initialized on first step → (N, 5, 1, 64, 64)
        self.z_enc = None         # (N, 64)

    def _post_physics_step(self):
        super()._post_physics_step()
        # Get depth from camera sensor
        depth = self.scene["head_rgbd"].data.depth  # (N, 720, 1280, 1)
        depth = depth.permute(0, 3, 1, 2)            # (N, 1, 64, 64)
        depth = depth.clamp(0, 6.0) / 6.0            # normalize to [0, 1]

        # Initialize or roll buffer
        if self.depth_buffer is None:
            self.depth_buffer = depth.unsqueeze(1).repeat(1, 5, 1, 1, 1)
        else:
            self.depth_buffer = torch.cat(
                [self.depth_buffer[:, 1:], depth.unsqueeze(1)], dim=1
            )

        # Compute z_enc
        self.z_enc = self.depth_encoder(self.depth_buffer)  # (N, 64)
```

Add observation terms in `src/colosseum/tasks/dribbling/mdp/observations.py`:

```python
def z_enc(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Depth encoder latent — implicit ball position + occlusion state. Shape (N, 64)."""
    return env.z_enc
```

Then register `z_enc` in `observation_cfg.py` under `actor_terms`.

## Step D — Auxiliary Supervision Loss

The paper trains with: `L_visual = MSE(proj_head(z_enc), gt_nxny) * in_front_mask`.

- GT (nx, ny) already computable via `_project_ball_to_camera()` in `src/colosseum/tasks/dribbling/mdp/rewards.py:225`.
- Wire `projection_head(z_enc)` against GT each step.
- Warm-up schedule: `λ_v = 1.0` for first 50 iters, then PPO ramps up.
- Encoder learning rate: `η_policy / 100` with separate gradient clipping.

This requires extending the RL algorithm (PPO) config to include the auxiliary loss. Two options:
1. **Custom PPO callback/hook** — add the aux loss inside the PPO update step.
2. **Reward-based proxy** — add a reward term `exp(-MSE(predicted_nxny, gt_nxny))` (weaker but no algo changes needed).

Option 1 is correct; option 2 is a quick sanity check.

## Step E — Observation Vector Update

Current actor observation (from `observation_cfg.py`):
- base_ang_vel (3), projected_gravity (3), joint_pos (23), joint_vel (23), actions (23), gait_phase (4), command (2), ball_pos (2), ball_vel (3), base_height (1), ball_mass (1), ball_friction (1), foot_ball_contact_force (1) = ~90D

After adding z_enc:
- Add `z_enc` (64D) to actor_terms
- Move `ball_vel`, `base_height`, `ball_mass`, `ball_friction` to critic-only (privileged) — these are what z_contact and z_enc should implicitly estimate
- Paper's 165D target: base_ang_vel (3) + projected_gravity (3) + joint_pos (23) + joint_vel (23) + actions (23) + gait_phase (2) + command (2) + ball_pos (2) + ball_vel (2) + z_contact (16) + z_enc (64) + ball_projection_nxny (2)

Note: z_contact (16D, contact/terrain RMA latent) is a separate encoder — not covered here.

## Deployment

At deployment:
- `DepthEncoder` runs on real depth frames from the D455 camera
- `ProjectionHead` is discarded — only `z_enc` enters the policy
- YOLO provides explicit (nx, ny) when ball is visible; z_enc covers occluded cases
- Both channels always feed the policy (no mode-switching)

## File Touchpoints

| File | Change |
|------|--------|
| `robots/t1_23dof/constants.py` | **DONE** — Calibrated D455 intrinsics + `get_spec_with_head_camera()` |
| `robots/t1_23dof/sensors.py` | **DONE** — `HEAD_RGBD_SENSOR` (1280x720, rgb+depth) |
| `tasks/dribbling/config/t1_23dof/t1_dribbling_cfg.py` | **DONE** — `HEAD_RGBD_SENSOR` added to scene sensors |
| `tasks/dribbling/mdp/depth_encoder.py` | **New** — `DepthEncoder` + `ProjectionHead` modules |
| `tasks/dribbling/env.py` | Extend `DribblingEnv` with encoder, depth buffer, `z_enc` |
| `tasks/dribbling/mdp/observations.py` | Add `z_enc()` observation function |
| `tasks/dribbling/config/t1_23dof/observation_cfg.py` | Add `z_enc` term, reorganize privileged obs |
| RL algo config | Add auxiliary projection loss to PPO |
