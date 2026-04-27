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

# Phase 2 — train visual adaptation encoder (freeze Phase 1 policy)
pixi run train-phase2 task:t1-dribbling

# Force a fixed obstacle stage (-1 = use curriculum)
pixi run train task:t1-dribbling --obstacle-stage-index 0
pixi run train task:t1-dribbling --obstacle-stage-index 3

# Disable W&B logging
pixi run train task:t1-dribbling logger:disabled
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

## Evaluation (Play)

```bash
# Run a trained checkpoint
pixi run play task:t1-dribbling --checkpoint ./logs/<run>/checkpoints/latest.pt

# Fix obstacle stage during play
pixi run play task:t1-dribbling --checkpoint ./logs/<run>/checkpoints/latest.pt --obstacle-stage-index 2
```

Play mode uses the task's default `num_envs`. Override with `--num-envs 1` for a single environment.

### Viser viewer

```bash
pixi run play task:t1-dribbling --checkpoint ... --viewer viser
```

#### Troubleshooting: `MjlabViserScene` cannot be instantiated

```
TypeError: Can't instantiate abstract class MjlabViserScene with abstract method add_rectangle
```

Known bug in the installed mjlab. Fix: open
`.pixi/envs/default/lib/python3.12/site-packages/mjlab/viewer/viser/scene.py`
and add this stub to `MjlabViserScene` just before `clear()` (~line 427):

```python
@override
def add_rectangle(self, *args, **kwargs) -> None:  # type: ignore[override]
    pass
```

---

## Multi-GPU Training (GIN — two RTX 4090)

GIN has two RTX 4090 GPUs (24 GB VRAM each).

| Phase | Max envs | VRAM |
|-------|----------|------|
| Phase 1 | 20480 (10240/GPU) | ~40 GB |
| Phase 2 | 2048 (1024/GPU) | ~44 GB |

```bash
# Phase 1
pixi run train task:t1-dribbling --task.env.scene.num-envs 10240 --cuda 0,1

# Phase 2
pixi run train-phase2 task:t1-dribbling --task.env.scene.num-envs 1024 --cuda 0,1 \
  --checkpoint checkpoints/dribbling_phase_1.pt
```

`--task.env.scene.num-envs` is **per process**, so two ranks × 10240 = 20480 total environments.

Equivalent explicit `torchrun` invocation:

```bash
CUDA_VISIBLE_DEVICES=0,1 pixi run torchrun \
    --standalone --nproc_per_node=2 \
    -m colosseum.scripts.train task:t1-dribbling
```

### W&B in multi-GPU

- Only rank 0 creates the W&B run and uploads checkpoints.
- Use `logger:disabled` to suppress W&B entirely.

### Checkpoints in multi-GPU

- Written to `logs/<run>/checkpoints/` by rank 0 only.
- `latest.pt` always points to the newest checkpoint.
- `interrupted.pt` is written on Ctrl+C.
- All ranks load the same checkpoint path on resume.
