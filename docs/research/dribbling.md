# Dribbling with Obstacle Avoidance

> Humanoid robot soccer dribbling with curriculum-based dynamic obstacle avoidance
> and visual adaptation via RMA (Rapid Motor Adaptation).

**Project page:** [lab-rococo-sapienza.github.io/learning-to-dribble](https://lab-rococo-sapienza.github.io/learning-to-dribble/)

---

## What this is

The dribbling project trains a Booster T1 humanoid to dribble a soccer ball
toward a persistent world-frame target while avoiding moving obstacles.

Training uses a two-phase RMA pipeline:

- **Phase 1** — policy trained with privileged observations (ball position,
  obstacle states) across a staged obstacle curriculum (0 → 4 obstacle stages).
- **Phase 2** — visual adaptation encoder trained on top of the frozen Phase 1
  policy, replacing privileged inputs with depth-camera features.

---

## Key design choices

**Target-driven ball command** — a persistent world-frame target is sampled
each episode from the current ball position at a random distance and heading.
The desired ball velocity is recomputed every step from `ball → target`. This
gives the policy a stable long-horizon objective throughout each episode,
rather than a free velocity that can be satisfied trivially in any direction.

**Closest-obstacle representation** — the policy tracks only the single nearest
active obstacle (position + velocity in robot body frame). Up to three physical
obstacles exist in the scene; only the closest one is exposed. This
representation is compact and generalises across different numbers of active
obstacles.

**DAgger teacher** — the Stage-0 (no obstacles) PPO checkpoint acts as a
teacher for later curriculum stages. The student minimises a mix of PPO loss
and imitation loss toward the teacher's action distribution. This keeps the
nominal dribbling gait stable while the student learns avoidance behaviour.

**Relaxed tracking near obstacles** — the ball-velocity tracking reward is
scaled down when an obstacle is blocking the direct path to the target. This
prevents the policy from being penalised for detours that are necessary for
avoidance.

---

## Observation and action space

**Actor observations** (proprioceptive, available at deployment):

| Group | Terms | Dim |
|---|---|---|
| Base state | lin_vel, ang_vel, projected_gravity | 9 |
| Joints | pos_rel, vel_rel | 46 |
| Last action | — | 23 |
| Ball velocity command | — | 3 |
| RMA latent | from encoder | 64 |
| **Total** | | **145** |

**Privileged observations** (Phase 1 only):

| Group | Terms | Dim |
|---|---|---|
| `privileged_ball` | ball pos + vel in robot frame | 4 |
| `privileged_obstacles` | nearest obstacle pos + vel in robot frame | 4 |

**Actions**: 23-DOF joint position targets

---

## Code structure

```
src/colosseum/tasks/dribbling/
├── config/t1_23dof/
│   ├── t1_dribbling_cfg.py     # ColosseumEnvCfg assembly + task registration
│   ├── scene_cfg.py            # robot + ball + obstacles
│   ├── observation_cfg.py      # actor, privileged_ball, privileged_obstacles, depth_frames
│   ├── reward_cfg.py           # ball tracking + locomotion regularization
│   ├── event_cfg.py            # ball + obstacle reset events
│   └── command_cfg.py          # BallVelocityCommand, AdversaryCommand, GaitPhaseCommand
└── mdp/
    ├── rewards.py              # ball_vel_tracking_relaxed, robot_ball_distance, ...
    ├── observations.py         # ball_pos_robot_frame, obstacle_pos_robot_frame, ...
    ├── ball_velocity_command.py # BallVelocityCommand (persistent target)
    └── adversary_command.py    # AdversaryCommand (obstacle position/velocity)

src/colosseum/research/dribbling/
├── encoders.py                 # DepthEncoder, BallHead, ObstacleHead
├── rma_terms.py                # DribblingRmaTerm, DribblingRmaTermCfg
└── scripts/
    ├── pipeline_dribbling.py   # Full curriculum (Phase 1 + Phase 2 per stage)
    ├── train_phase2.py         # Phase 2 visual encoder training
    ├── evaluate_dribbling.py   # Evaluation protocol runner
    └── obstacle_ablation.py    # Ablation of obstacle presence
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

# 4. Full curriculum pipeline (Phase 1 + Phase 2 per stage, automated)
pixi run pipeline-dribbling

# 5. Evaluate across all obstacle conditions
pixi run eval-dribbling task:t1-dribbling \
  --checkpoint ./logs/<run>/checkpoints/latest.pt \
  --episodes-per-condition 1000 --num-envs 128

# 6. Obstacle presence ablation
pixi run obstacle-ablation

# 7. Export for deployment
pixi run export-onnx task:t1-dribbling \
  --checkpoint ./logs/<run>/checkpoints/latest.pt
```
