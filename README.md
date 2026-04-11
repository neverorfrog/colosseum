# Colosseum

Learning playground for humanoid robots related to robot soccer


## Training

All commands use the `train` pixi environment.

### Phase 1
```bash
# Train the dribbling task with default settings (phase 1)
pixi run -e train train task:t1-dribbling

# Select a specific GPU on multi-GPU machines (single-GPU run)
pixi run -e train train task:t1-dribbling --cuda 0
pixi run -e train train task:t1-dribbling --cuda 1
# In single-GPU mode, --cuda N pins CUDA_VISIBLE_DEVICES=N,
# so only that physical GPU is visible to the process.

# Use depth camera (phase 2)
pixi run -e train train task:t1-dribbling --task.use-depth-camera

# Override num_envs (more envs = faster training, more VRAM)
pixi run -e train train task:t1-dribbling --task.env.scene.num-envs 64

# Override training steps
pixi run -e train train task:t1-dribbling --task.algo-cfg.learning-steps 100000000


# Resume from a checkpoint
pixi run -e train train task:t1-dribbling --checkpoint ./logs/<run-dir>/checkpoints/latest.pt

# Set a specific seed
pixi run -e train train task:t1-dribbling --seed 0
```

Training logs and checkpoints are saved under `./logs/`. Metrics are logged to W&B.

### Multi-GPU (2x 4090)

Use `torchrun` to launch one worker per GPU for the custom PPO trainer:

```bash
# 2-GPU dribbling training (single synchronized run)
CUDA_VISIBLE_DEVICES=0,1 pixi run -e train torchrun \
    --standalone --nproc_per_node=2 \
    -m colosseum.scripts.train task:t1-dribbling

# Optional: disable W&B logging
CUDA_VISIBLE_DEVICES=0,1 pixi run -e train torchrun \
    --standalone --nproc_per_node=2 \
    -m colosseum.scripts.train task:t1-dribbling logger:disabled
```

Notes:

- `--nproc_per_node=2` should match the number of GPUs you want to use.
- Each rank gets its own CUDA device automatically (`LOCAL_RANK` -> `cuda:LOCAL_RANK`).
- `--cuda` is for single-GPU runs; in distributed mode it is ignored because `torchrun` assigns devices per rank.
- Rank 0 writes the main run logs/checkpoints; rank 1 writes to `logs/<run>/rank_1/train.log`.
- Quick check:
    - `logs/<run>/train.log` should show rank 0 using `cuda:0`.
    - `logs/<run>/rank_1/train.log` should show rank 1 using `cuda:1`.

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
- Checkpoints are written to `logs/<run>/checkpoints/` with names like `model_0000100.pt`.
- `latest.pt` is updated to point to the newest checkpoint.
- On Ctrl+C, rank 0 writes `logs/<run>/checkpoints/interrupted.pt`.
- When resuming (`--checkpoint ...`), all ranks load the same checkpoint path so training state stays synchronized.

### Run training in background (survives terminal loss)

Use `setsid` to detach the process group, then save both PID and PGID so you can stop it later.

Single GPU:

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$HOME/colosseum_bg/$RUN_ID"
mkdir -p "$RUN_DIR"

setsid bash -lc '
    cd /home/phd_student/Spagnoli/colosseum
    export CUDA_VISIBLE_DEVICES=0
    exec pixi run -e train train task:t1-dribbling
' > "$RUN_DIR/train.log" 2>&1 < /dev/null &

LAUNCH_PID=$!
PGID=$(ps -o pgid= "$LAUNCH_PID" | tr -d " ")
echo "$LAUNCH_PID" > "$RUN_DIR/train.pid"
echo "$PGID" > "$RUN_DIR/train.pgid"
echo "PID=$LAUNCH_PID PGID=$PGID LOG=$RUN_DIR/train.log"
```

Multi GPU (2x 4090):

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$HOME/colosseum_bg/$RUN_ID"
mkdir -p "$RUN_DIR"

setsid bash -lc '
    cd /home/phd_student/Spagnoli/colosseum
    export CUDA_VISIBLE_DEVICES=0,1
    exec pixi run -e train torchrun \
        --standalone --nproc_per_node=2 \
        -m colosseum.scripts.train task:t1-dribbling
' > "$RUN_DIR/train.log" 2>&1 < /dev/null &

LAUNCH_PID=$!
PGID=$(ps -o pgid= "$LAUNCH_PID" | tr -d " ")
echo "$LAUNCH_PID" > "$RUN_DIR/train.pid"
echo "$PGID" > "$RUN_DIR/train.pgid"
echo "PID=$LAUNCH_PID PGID=$PGID LOG=$RUN_DIR/train.log"
```

Monitor:

```bash
ps -fp "$(cat "$RUN_DIR/train.pid")"
tail -f "$RUN_DIR/train.log"
```

Stop cleanly (recommended):

```bash
kill -TERM -"$(cat "$RUN_DIR/train.pgid")"
```

Force stop if needed:

```bash
kill -KILL -"$(cat "$RUN_DIR/train.pgid")"
```

Why PGID: `torchrun` spawns child processes. Killing the process group stops the whole job, not just one process.


## Playing (Evaluation)

Evaluate a trained policy in simulation:

```bash
# Play with a trained checkpoint
pixi run -e train play task:t1-dribbling --checkpoint ./logs/<run-dir>/checkpoints/latest.pt

# Play with a zero-action agent (robot stands still)
pixi run -e train play task:t1-dribbling --agent zero

# Play with a random-action agent
pixi run -e train play task:t1-dribbling --agent random

# Use a specific viewer backend
pixi run -e train play task:t1-dribbling --agent zero --viewer native
pixi run -e train play task:t1-dribbling --agent zero --viewer viser
```

Play mode uses the task's `num_envs` by default. Override with `--num-envs 1` to run a single environment.


## Troubleshooting

### Viser viewer: `MjlabViserScene` cannot be instantiated

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