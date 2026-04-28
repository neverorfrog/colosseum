# Getting Started

After [installing](setup.md), you can train and evaluate any registered task.

## Available tasks

| Task ID | Description |
|---------|-------------|
| `t1-dribbling` | Obstacle-avoidance dribbling with visual RMA |
| `t1-soccer-maze` | Goal-conditioned maze navigation to a ball |
| `cartpole` | CartPole balancing demo |

## Train a policy

```bash
pixi run train task:t1-dribbling
```

Logs and checkpoints are saved under `./logs/`. Metrics are streamed to W&B.

## Evaluate a checkpoint

```bash
pixi run play task:t1-dribbling --checkpoint ./logs/<run>/checkpoints/latest.pt
```

## Override common parameters

```bash
# Number of parallel environments (more = faster, more VRAM)
pixi run train task:t1-dribbling --task.env.scene.num-envs 4096

# Single GPU
pixi run train task:t1-dribbling --cuda 0

# Disable W&B logging
pixi run train task:t1-dribbling logger:disabled
```

Full flag reference and multi-GPU setup: [Training & Evaluation](training.md).

## Research projects

| Project | Docs |
|---------|------|
| Dribbling with obstacle avoidance | [research/dribbling.md](../research/dribbling.md) |
| HumanoidSoccerMaze benchmark | [research/soccer-maze.md](../research/soccer-maze.md) |
