# Dribbling with Obstacle Avoidance

> Humanoid robot soccer dribbling with curriculum-based dynamic obstacle avoidance
> and visual adaptation via RMA (Rapid Motor Adaptation).

**Project page:** https://lab-rococo-sapienza.github.io/learning-to-dribble/

---

## What this is

The dribbling project trains a Booster T1 humanoid to dribble a soccer ball
toward a persistent world-frame target while avoiding moving obstacles.

Training uses a two-phase RMA pipeline:

- **Phase 1** — policy trained with privileged observations (ball position,
  obstacle states) across a staged obstacle curriculum (0 → 4 obstacle stages).
- **Phase 2** — visual adaptation encoder trained on top of the frozen Phase 1
  policy, replacing privileged inputs with depth-camera features.

### Key design choices

**Target-driven ball command** — a persistent world-frame target is sampled each
episode; the desired ball velocity is recomputed every step from `ball → target`.
Obstacle avoidance detours are still evaluated against this stable long-horizon
objective.

**Closest-obstacle representation** — the policy tracks only the single nearest
active obstacle (position + velocity). Up to three physical obstacles exist in
the scene; curriculum difficulty increases by changing obstacle behavior
(static → slow → fast → attacker), not by increasing the number tracked.

**DAgger teacher** — the Stage-0 PPO checkpoint acts as teacher for later
stages so the student policy keeps the nominal dribbling gait while learning
avoidance.

---

## Code structure

```
src/colosseum/research/dribbling/
├── encoders.py       # DepthEncoder, BallHead, ObstacleHead
├── rma_terms.py      # DribblingRmaTerm, DribblingRmaTermCfg
└── scripts/
    ├── pipeline_dribbling.py   # Full curriculum (Phase 1 + Phase 2 per stage)
    ├── train_phase2.py         # Phase 2 visual encoder training
    └── evaluate_dribbling.py   # Evaluation protocol runner
```

---

## Quick start

```bash
# 1. Train nominal dribbling — Phase 1, no obstacles
pixi run train task:t1-dribbling --task.obstacle-stage-index 0

# 2. Train obstacle stage i with DAgger (Stage-0 checkpoint as teacher)
pixi run train task:t1-dribbling \
  --task.use-dagger \
  --task.teacher-checkpoint ./logs/<stage0-run>/checkpoints/latest.pt \
  --task.obstacle-stage-index i 

# 3. Train visual adaptation encoder — Phase 2
pixi run train-phase2 task:t1-dribbling \
  --checkpoint ./logs/<phase1-run>/checkpoints/latest.pt

# 4. Full curriculum pipeline (Phase 1 + Phase 2 per stage)
pixi run pipeline-dribbling

# 5. Evaluate
pixi run eval-dribbling task:t1-dribbling \
  --checkpoint ./logs/<run>/checkpoints/latest.pt \
  --episodes-per-condition 1000 --num-envs 128
```
