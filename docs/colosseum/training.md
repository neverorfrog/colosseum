# Training & Evaluation

## Setup

```bash
pixi install
```

The default pixi environment includes all training dependencies. Always invoke scripts through pixi:

```bash
# Correct
pixi run train task:t1-dribbling

# Wrong (will fail with ModuleNotFoundError)
python -m colosseum.scripts.train task:t1-dribbling
```

---

## Training

```bash
# Phase 1 — train with privileged observations
pixi run train task:t1-dribbling
```

### Common flags

```bash
# GPU selection
--cuda 0            # single GPU
--cuda 0,1          # multi-GPU

# Scale environments (more = faster, more VRAM)
--task.env.scene.num-envs 4096

# Override checkpoint save interval
--logger.save-interval 1000000

# Override total training steps
--task.algo-cfg.learning-steps 100000000

# Resume from checkpoint
--checkpoint ./logs/<run>/checkpoints/latest.pt

# DAgger-regularized PPO (requires a teacher checkpoint)
--task.use-dagger
--task.teacher-checkpoint ./logs/<stage0-run>/checkpoints/latest.pt
```

Logs and checkpoints are saved under `./logs/`. Metrics go to W&B.

---

## W&B metrics

The most useful metrics to watch during a training run:

| Metric | What it means |
|---|---|
| `train/episode_return` | Mean cumulative reward per episode. Should rise steadily. |
| `train/episode_length` | Mean episode length in steps. Short episodes early on → terminations are triggered frequently. |
| `train/value_loss` | Critic loss. Should decrease and stabilize. Spikes indicate instability. |
| `train/surrogate_loss` | PPO policy loss. Erratic oscillation → learning rate too high. |
| `terminations/<name>` | Per-term termination rate. Useful to see *why* episodes end (fall, timeout, joint limits). |
| `rewards/<name>` | Per-term episode sum. Watch the task reward rise and regularization terms stay bounded. |
| `curriculum/level` | Current terrain/obstacle difficulty level (if curriculum is active). |

For the dribbling task specifically:

| Metric | What to look for |
|---|---|
| `rewards/ball_vel_tracking` | Primary signal — should dominate the return. |
| `rewards/upright` | Should be high (close to max) throughout; large drops signal falling. |
| `terminations/base_contact` | Should fall toward zero as the robot learns not to fall. |
| `train/episode_length` | Should approach `episode_length_s / step_dt` (20 s × 50 Hz = 1000 steps). |

---

## Evaluation (Play)

```bash
# Run a trained checkpoint in the viewer
pixi run play task:t1-dribbling --checkpoint ./logs/<run>/checkpoints/latest.pt
```

Play mode uses the task's default `num_envs`. Override with `--num-envs 1` for a
single environment, which is easier to watch.

### Viser viewer

```bash
pixi run play task:t1-dribbling --checkpoint ... --viewer viser
```

---

## Exporting to ONNX

Once training is complete, export the policy for deployment with arena:

```bash
pixi run export-onnx task:t1-dribbling \
    --checkpoint ./logs/<run>/checkpoints/latest.pt \
    --output-dir ./models/
```

This traces the policy (actor network + encoder in their Phase 1 or Phase 2
configuration) and writes a `.onnx` file with a fixed input shape matching
the training observation layout. See [Deployment (Arena)](deployment.md) for
how to use the exported model.

---

## Multi-GPU Training

```bash
# Phase 1
pixi run train task:t1-dribbling --task.env.scene.num-envs 10240 --cuda 0,1

# Phase 2
pixi run train-phase2 task:t1-dribbling --task.env.scene.num-envs 1024 --cuda 0,1 \
  --checkpoint checkpoints/dribbling_phase_1.pt
```

`--task.env.scene.num-envs` is **per process**, so two ranks × 10240 = 20480
total environments.

Equivalent explicit `torchrun` invocation:

```bash
CUDA_VISIBLE_DEVICES=0,1 pixi run torchrun \
    --standalone --nproc_per_node=2 \
    -m colosseum.scripts.train task:t1-dribbling
```

### W&B in multi-GPU

Only rank 0 creates the W&B run and uploads checkpoints. Use `logger:disabled`
to suppress W&B entirely.

### Checkpoints in multi-GPU

Written to `logs/<run>/checkpoints/` by rank 0 only. `latest.pt` always
points to the newest checkpoint. `interrupted.pt` is written on Ctrl+C. All
ranks load the same checkpoint path on resume.
