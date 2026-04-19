# Colosseum

Learning playground for humanoid robots related to robot soccer


## Training

All commands use the `train` pixi environment.

```bash
# Phase 1
pixi run -e train train task:t1-dribbling

# Phase 2
pixi run -e train train-phase2 task:t1-dribbling --task.use-depth-camera

# Force a fixed obstacle stage (-1 = use curriculum)
pixi run -e train train task:t1-dribbling --obstacle-stage-index 0
pixi run -e train train task:t1-dribbling --obstacle-stage-index 3
pixi run -e train play task:t1-dribbling --checkpoint ./logs/<run-dir>/checkpoints/latest.pt --obstacle-stage-index 4

# disable logging
pixi run -e train train task:t1-dribbling logger:disabled
```

```bash
# Single-GPU
--cuda <device> # e.g. --cuda 0 to use GPU 0, --cuda 1 to use GPU 1

# Multi-GPU
--cuda <device_1>,<device_2> # e.g. --cuda 0,1, default 0

# Override num_envs (more envs = faster training, more VRAM)
--task.env.scene.num-envs <num_envs> # e.g. --task.env.scene.num-envs 1024, default 1

# Ovveride save_interval for checkpoints
--logger.save-interval <save_interval> # e.g. --logger.save-interval 1000000, default 1000000

# Override training steps
--task.algo-cfg.learning-steps <learning-steps> # e.g. --task.algo-cfg.learning-steps 100000000 default 500000000

# Resume from a checkpoint
--checkpoint ./logs/<run-dir>/checkpoints/latest.pt
```

Training logs and checkpoints are saved under `./logs/`. Metrics are logged to W&B.

### Notes to train on two 4090 (GIN setup)
GIN has two RTX 4090 GPUs, each with 24 GB of VRAM.

* Phase 1 multi-GPU training supports up to 20480 parallel environments (10240 per GPU) (about 40 GB of VRAM).
* Phase 2 multi-GPU training supports up to 2048 parallel environments (1024 per GPU) (about 44 GB of VRAM).

### Multi-GPU Training details

Simplest way (recommended):
```bash
# Phase 1
pixi run -e train train task:t1-dribbling --task.env.scene.num-envs 10240 --cuda 0,1

# Phase 2
pixi run -e train train-phase2 task:t1-dribbling --task.use-depth-camera --task.env.scene.num-envs 1024 --cuda 0,1 --checkpoint checkpoints/dribbling_phase_1.pt
```

Equivalent explicit `torchrun` command to launch one worker per GPU for the custom PPO trainer:

```bash
CUDA_VISIBLE_DEVICES=0,1 pixi run -e train torchrun \
    --standalone --nproc_per_node=2 \
    -m colosseum.scripts.train task:t1-dribbling
```
where `--nproc_per_node=2` should match the number of GPUs you want to use.

**Notes:**

**With the current implmentation `num-envs` is per process. So, for example `--task.env.scene.num-envs 1024` with `--cuda 0,1` (2 ranks), means effective total environment = 2048.** 


#### W&B logging in multi-GPU

- Only rank 0 creates the W&B run and logs metrics.
- Non-main ranks do not initialize W&B, so you get one run per training job (not one run per GPU).
- Checkpoints and config files are uploaded to W&B from rank 0 only.
- If you want no W&B logging, use `logger:disabled` in the launch command.

#### What rank 0 and rank 1 mean

- `torchrun --nproc_per_node=2` starts 2 training processes (workers).
- On one machine with 2 GPUs, each process is assigned one GPU:
    - Rank 0 -> local rank 0 -> `cuda:0` (main process)
    - Rank 1 -> local rank 1 -> `cuda:1` (worker process)
- Both ranks run the same training code and synchronize gradients each update.
- Side effects are handled by rank 0 only (W&B run, main console logging, periodic checkpoints).

#### How checkpoints are saved in multi-GPU

- Periodic checkpoints are configured only on rank 0.
- Checkpoints are written to `logs/<run>/checkpoints/`.
- `latest.pt` is updated to point to the newest checkpoint.
- On Ctrl+C, rank 0 writes `logs/<run>/checkpoints/interrupted.pt`.
- When resuming (`--checkpoint ...`), all ranks load the same checkpoint path so training state stays synchronized.


## Playing (Evaluation)

Evaluate a trained policy in simulation:

```bash
# Play with a trained checkpoint
pixi run -e train play task:t1-dribbling --checkpoint ./logs/<run-dir>/checkpoints/latest.pt

# Play a fixed obstacle stage (-1 = use obstacle curriculum)
pixi run -e train play task:t1-dribbling --checkpoint ./logs/<run-dir>/checkpoints/latest.pt --obstacle-stage-index 2
```
Play mode uses the task's `num_envs` by default. Override with `--num-envs 1` to run a single environment.

### Viser viewer

To use `viser` as viewer:
```bash
--viewer viser
```

#### Troubleshooting: Viser viewer: `MjlabViserScene` cannot be instantiated

When launching with `--viewer viser`, you may see:

```
TypeError: Can't instantiate abstract class MjlabViserScene with abstract method add_rectangle
```

This is a known bug in the installed mjlab: `MjlabViserScene` inherits from `DebugVisualizer` which declares `add_rectangle` as abstract, but the Viser backend does not implement it yet.

**Fix:** add the missing no-op stub to the installed mjlab. Open `.pixi/envs/train/lib/python3.11/site-packages/mjlab/viewer/viser/scene.py` and add the following method to `MjlabViserScene`, just before `clear()` (around line 427):

```python
@override
def add_rectangle(self, *args, **kwargs) -> None:  # type: ignore[override]
    pass
```
