# Soccer Maze

> Humanoid robot navigation through a structured maze to reach and interact
> with a soccer ball.

<!-- TODO: replace assets/thumbnail.png with a screenshot from the video -->
[![Soccer Maze demo](assets/thumbnail.png)](https://drive.google.com/file/d/1CgFFBQNQtTevHFo_cPRcrJF6HEFSO9g2/view?usp=sharing)

---

## What this is

Soccer Maze trains a Booster T1 humanoid to navigate a maze-like environment
and reach a soccer ball target. The task combines goal-conditioned locomotion,
structured obstacle avoidance, and ball-interaction rewards.

---

## Code structure

```
src/colosseum/research/soccer-maze/
└── scripts/        # Evaluation and ablation scripts (to be added)

src/colosseum/tasks/soccer_maze/   # Task definition (scene, MDP, rewards)
```

---

## Quick start

```bash
# Train
pixi run train task:t1-soccer-maze

# Evaluate
pixi run play task:t1-soccer-maze --checkpoint ./logs/<run>/checkpoints/latest.pt
```
