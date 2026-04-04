# Colosseum

Learning playground for humanoid robots related to robot soccer


## Training

All commands use the `train` pixi environment.

```bash
# Train the dribbling task with default settings
pixi run -e train train task:t1-dribbling

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