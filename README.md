# Colosseum

Learning playground for humanoid robots related to robot soccer,
built on top of [mjlab](external/mjlab) (GPU-accelerated RL via MuJoCo Warp).

## Setup

```bash
pixi install
```

Requires Linux x86-64, Python 3.12, and an NVIDIA GPU.
Always run scripts through pixi — `pixi run python` / `pixi run train` — to
ensure the correct environment is active.

## Training & Evaluation

```bash
# Train a task
pixi run train task:t1-dribbling

# Evaluate a checkpoint
pixi run play task:t1-dribbling --checkpoint ./logs/<run>/checkpoints/latest.pt
```

Full CLI reference, multi-GPU setup, and troubleshooting:
**[docs/colosseum/training.md](docs/colosseum/training.md)**

## Research Projects

| Project | Description |
|---------|-------------|
| [Dribbling](src/colosseum/research/dribbling/README.md) | Obstacle-avoidance dribbling with visual RMA |
| [Soccer Maze](src/colosseum/research/soccer-maze/README.md) | Goal-conditioned maze navigation to a ball |
