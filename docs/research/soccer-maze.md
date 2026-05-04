# HumanoidSoccerMaze

> A novel benchmark requiring a humanoid robot to navigate a maze while dribbling a
> ball to a distant target in presence of complex obstacle configurations and 
> a baseline solution integrating planning and learning, using the Sokoban problem as an abstraction of ball dribbling with obstacles

---

This is an example of a evaluation after approximately 200 million steps of training on a RTX 4090 in approximately 3 hours. 

<iframe
  src="https://drive.google.com/file/d/1CgFFBQNQtTevHFo_cPRcrJF6HEFSO9g2/preview"
  width="720" height="405"
  allow="autoplay"
  style="border:none; border-radius:4px;"
></iframe>

---

## What this is

The benchmark tests a Booster T1 humanoid on goal-conditioned maze navigation
combined with ball dribbling. The robot must plan a path through a maze and
execute it with a learned locomotion-dribbling policy.

The baseline solution integrates planning and learning, using the Sokoban
problem as an abstraction of ball dribbling with obstacles.

The benchmark is built on [mjlab](https://github.com/neverorfrog/mjlab)
(GPU-accelerated RL via MuJoCo Warp) and uses the
[Booster T1](https://www.boostspace.com/) 23-DoF humanoid — the platform used
in the [RoboCup Humanoid Soccer League](https://www.robocup.org/leagues/35).

---

## Code structure

```
src/colosseum/tasks/soccer_maze/    # Benchmark environment (scene, MDP, rewards)
src/colosseum/research/soccer-maze/ # Paper scripts and experiment configs
└── scripts/                        # Evaluation and ablation scripts
```

---

## Quick start

```bash
# Train
pixi run train task:t1-soccer-maze

# Evaluate
pixi run play task:t1-soccer-maze --checkpoint ./logs/<run>/checkpoints/latest.pt
```

## Paper

The paper describes the technical contribution of the benchmark (environment features, performance metrics) and the baseline solution approach. It is under submission at **RoboCup Symposium 2026**.

> **HumanoidSoccerMaze: a novel benchmark and a hierarchical solution approach**  
> Flavio Maiorana, Daniel Gigliotti, Luca Iocchi  
> *RoboCup Symposium 2026 (under submission)*
