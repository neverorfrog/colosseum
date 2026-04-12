# Phase 2 Fix: LayerNorm Output + Direct Latent MSE Loss

## Symptom

After a full Phase 2 run (512 envs, ~6 h, clean loss curves converging to
~1e-3), the deployed policy failed to dribble at all. Phase 1 with the same
task converged cleanly and works. The adaptation encoder itself — CNN + GRU
+ ball head — had small supervised errors on `[x, y, vx, vy]`, yet the actor
behaved as if it were receiving noise.

## Diagnosis (Check B)

Instrumentation in `rma_manager.encode_phase2_with_fallback` compared the
privileged latent `z_priv` (used by the frozen actor in Phase 1) against the
adaptation latent `z_adapt` (used by the frozen actor in Phase 2):

| quantity         | privileged (Phase 1 frozen) | adaptation (Phase 2 trained) |
|------------------|-----------------------------|------------------------------|
| `\|z\|`          | ~5.3 (stable)               | 3.6 – 15.2 (swinging)         |
| first 4 dims     | smooth, bounded             | chaotic, huge                 |
| `\|dz\|` in FOV  | —                           | **7 – 16** (should be ~0.5–1.5) |

Ball-head predictions separately collapsed toward the training distribution
mean even though per-sample loss was low.

Two independent failures, one root cause each.

## Root Cause 1 — Manifold mismatch

`PrivilegedEncoder` ends in `LayerNorm(latent_dim)`. Its output lives on a
bounded, zero-mean, unit-variance manifold. The frozen actor learned to
consume exactly that.

`DepthEncoder` ended in a plain `Linear(gru_hidden, latent_dim)` — unbounded
magnitude, no distribution constraint. Even if the ball head decoded it
correctly, `z_adapt` was geometrically off the manifold the actor was
trained on. Out-of-distribution latents in → garbage actions out.

## Root Cause 2 — The core RMA loss was never computed

RMA's defining supervision is

```
L_rma = MSE(z_adapt, stop_grad(z_priv))
```

`rma_manager.compute_adaptation_loss` has this as a fallback, but the
fallback only runs when a term returns `None` from `compute_loss`.
`BallRmaTerm.compute_loss` returned a non-None dict with `ball_pos` and
`ball_vel` — so the RMA latent loss **never ran**.

The ball head is `Linear(64 → 32 → 4)`. Supervising only its 4-D output
constrains `z_seq` through a rank-≤4 projection; the entire left-nullspace
(60 of 64 directions) is unsupervised. The optimizer is free to put
anything it wants into those directions, and it did.

Phase 1 worked because the privileged MLP is small and stays well-behaved
under the end-to-end PPO loss. Phase 2 broke because we were asking a 3 M
parameter CNN+GRU to match a latent through a 4-D bottleneck.

## Fixes applied

### 1. `DepthEncoder.output_norm` — [encoders.py](../../src/colosseum/algorithm/encoders.py)

```python
self.output_norm = nn.LayerNorm(latent_dim)
...
z_t = self.head(out[:, 0, :])
z_t = self.output_norm(z_t)
```

Matches `PrivilegedEncoder`'s output manifold. Bounds `|z_adapt|` to the
same scale `z_priv` lives on, so even early in training the actor sees
in-distribution magnitudes.

### 2. Direct latent MSE in `BallRmaTerm.compute_loss` — [rma_terms.py](../../src/colosseum/tasks/dribbling/mdp/rma_terms.py)

```python
with torch.no_grad():
    gt_flat = gt.reshape(-1, gt.shape[-1])
    z_priv_seq = self._priv_encoder(gt_flat).reshape(B, T, -1)
latent_err = (z_seq - z_priv_seq).pow(2).mean(dim=-1)
...
return {
    "latent_mse": latent_loss,
    "ball_pos":   cfg.lambda_pos * pos_loss,
    "ball_vel":   cfg.lambda_vel * vel_loss,
}
```

Supervises every dimension of `z_seq` directly against the frozen
privileged latent — the RMA paper's loss, which had been silently missing.
`ball_pos`/`ball_vel` are kept as auxiliary regularizers (they give the
encoder an interpretable task signal and a useful probe for visualization),
but `latent_mse` is the load-bearing term.

## Expected result after retrain

- `\|dz\|fov` should drop to ~0.5–1.5 within a few thousand steps.
- `\|z_adapt\|` should stabilize around `\|z_priv\|` (~5 for latent_dim=64).
- Dribbling behavior should recover to Phase-1 quality once `latent_mse`
  falls below ~0.05.

## What to watch in future runs

- If `latent_mse` plateaus above ~0.1, the encoder is capacity-limited or
  the depth frames lack information — not a loss formulation issue.
- If `ball_pos` and `ball_vel` diverge from `latent_mse` (one goes down, the
  other does not), the ball head is fighting the latent regression —
  consider lowering `lambda_pos`/`lambda_vel`.
- Never let `BallRmaTerm.compute_loss` return a dict that omits
  `latent_mse` again. The fallback in `rma_manager` is a safety net, not
  the primary path.
