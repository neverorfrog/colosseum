# Dribbling Training Plan

This document separates two concepts that are easy to mix up:

- `phase 1` / `phase 2`: the RMA training phases
- `stage 1` / `stage 2` / ...: the obstacle curriculum stages inside phase 1

## RMA Phases

- `phase 1`: train the policy with privileged obstacle and ball information
- `phase 2`: freeze the phase-1 policy and train the visual adaptation encoder

The checkpoint metadata now records the RMA phase:

- `train.py` saves checkpoints with `metadata.phase = 1`
- `train_phase2.py` saves checkpoints with `metadata.phase = 2`
- both scripts also save the requested CLI value `metadata.obstacle_stage_index`

This means a checkpoint can tell you whether it belongs to phase 1 or phase 2.
It does **not** by itself mean "no obstacles" or "three obstacles". Obstacle
difficulty is controlled by the curriculum stage, not by the RMA phase.

## Obstacle Curriculum

The dribbling task uses a staged obstacle curriculum during phase 1:

1. `stage 0`: no obstacle
   Behavior: `none`
2. `stage 1`: one static blocker in front of the commanded ball direction
   Behavior: `static_blocker`
3. `stage 2`: one lateral blocker near the ball path
   Behavior: `lateral_blocker`
4. `stage 3`: one ball attacker
   Behavior: `ball_attacker`
5. `stage 4`: three obstacles
   Behavior: `mixed_attackers`
   Layout:
   - obstacle 0: ball attacker
   - obstacle 1: lateral blocker
   - obstacle 2: distractor
6. `phase 2`: train the visual encoder from the final phase-1 checkpoint

The curriculum logs these metrics:

- `obstacle_stage_index`
- `obstacle_behavior_id`
- `num_active_obstacles`
- `obstacle_min_dist_m`
- `obstacle_min_speed`
- `obstacle_max_speed`

Use these metrics in WandB or the training logs to know which obstacle stage
was active when a checkpoint was produced.

## Forcing A Stage From The CLI

Use `--obstacle-stage-index <idx>` in `train`, `train-phase2`, or `play`.

- `-1`: use obstacle curriculum
- `0`: no obstacle
- `1`: static blocker
- `2`: lateral blocker
- `3`: ball attacker
- `4`: mixed attackers

Examples:

```bash
pixi run -e train train task:t1-dribbling --obstacle-stage-index 0
pixi run -e train train task:t1-dribbling --obstacle-stage-index 3
pixi run -e train play task:t1-dribbling --checkpoint ./logs/<run>/checkpoints/latest.pt --obstacle-stage-index 4
```

Behavior:

- in training, `-1` keeps the obstacle curriculum enabled
- in training, `>= 0` disables obstacle curriculum progression and pins that stage
- in play, `-1` enables obstacle curriculum progression during evaluation
- in play, `>= 0` pins the requested stage

## Recommended Training Workflow

### Phase 1

Run phase 1 with the obstacle curriculum enabled:

```bash
pixi run -e train train task:t1-dribbling
```

Recommended checkpoint milestones:

1. Save a checkpoint near the end of `stage 0` if you want a pure no-obstacle
   dribbling checkpoint.
2. Save another checkpoint after the policy is stable in `stage 3` or `stage 4`.
3. Use the strongest stable phase-1 checkpoint as the starting point for phase 2.

Practical rule:

- if you want "phase 1 with 0 obstacles", that is **not** "all phase-1 checkpoints"
- it is a **phase-1 checkpoint saved while `obstacle_stage_index == 0`**

### Phase 2

Start phase 2 from the selected phase-1 checkpoint:

```bash
pixi run -e train train-phase2 \
  --checkpoint /path/to/phase1_checkpoint.pt \
  task:t1-dribbling
```

Phase 2 should use the same task distribution as the final phase-1 curriculum,
or a slightly easier subset at the beginning if adaptation is unstable.

## Why This Curriculum

The obstacle generator is designed to increase difficulty without destroying gait:

- early stages teach dribbling and path selection
- intermediate stages teach dynamic avoidance near the ball path
- late stages teach shielding the ball from an attacker
- final multi-obstacle stage forces the policy to identify the most relevant
  nearby obstacle while preserving the closest-obstacle representation used by
  observations and rewards

## Notes On Checkpoint Naming

Because curriculum stage is dynamic, the simplest way to keep checkpoints clear
is to name them explicitly when you export or copy them, for example:

- `phase1_stage0_no_obstacle.pt`
- `phase1_stage3_ball_attacker.pt`
- `phase1_stage4_mixed_attackers.pt`
- `phase2_visual_adaptation.pt`

That naming convention is usually easier to reason about later than relying only
on timestamps.
