# Colosseum

**Colosseum** is a research playground for humanoid robot soccer, built on top of
[mjlab](mjlab/overview.md) — a GPU-accelerated reinforcement learning framework
for robotics powered by MuJoCo Warp.

The primary platform is the **Booster T1** humanoid (23 DOF full body). 
Training runs thousands of parallel environments on a single GPU,
enabling fast iteration on complex locomotion and ball-manipulation tasks.

---

## Key features

- **Massive parallelization** — 4096+ environments simultaneously on GPU via MuJoCo Warp
- **Manager-based configuration** — rewards, observations, actions, and events declared
  as composable term configs; no boilerplate loops
- **Shared training / deployment code** — pure MDP functions are used unchanged
  in both training and on-robot inference
- **Curriculum support** — curricula built into the
  task layer
- **Research-ready** — each project lives in its own `research/` folder with a
  dedicated training pipeline and evaluation protocol

---

## Where to start

| I want to… | Go to |
|---|---|
| Install and run a first training | [Setup](colosseum/setup.md) → [Getting Started](colosseum/getting_started.md) |
| Understand the framework internals | [mjlab Overview](mjlab/overview.md) |
| Build a new task from scratch | [CartPole tutorial](colosseum/cartpole.md) |
| Read about dribbling with obstacle avoidance | [Dribbling](research/dribbling.md) |
| Read about goal-conditioned maze navigation | [Soccer Maze](research/soccer-maze.md) |
