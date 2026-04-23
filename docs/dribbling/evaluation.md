# Dribbling Evaluation Protocol

This document defines the evaluation protocol for the target-driven dribbling
policy with visual adaptation and obstacle avoidance.

The main goal is to evaluate a Phase 2 checkpoint trained up to obstacle stage
3, where stage 3 is a single `ball_attacker`, in settings that test both
retention and out-of-training-domain generalisation.

## Checkpoint Under Evaluation

Use the checkpoint produced after:

- Phase 2 visual adaptation training
- obstacle stage 3: one active `ball_attacker`

At inference, the policy should use the Phase 2 depth encoder unless explicitly
running an ablation with privileged observations.

## Evaluation Runner

The protocol can be run with:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./logs/<run>/checkpoints/latest.pt \
    --episodes-per-condition 1000 \
    --velocity-episodes-per-condition 300 \
    --num-envs 128 \
    --seeds 0 1 2 \
    --output-dir logs/dribbling_eval
```

The script prints the formatted results to the terminal and writes a Markdown
report under the output directory. When plot generation is enabled, it also
saves tracking-error and XY trajectory plots for the velocity diagnostic
conditions.

Use `--num-envs` to choose how many parallel environments are used for each
condition. Larger values speed up collection but require more GPU memory,
especially because Phase 2 evaluation uses depth rendering.

The script shows a progress bar for each condition and seed. The bar counts
completed trials, not simulator steps. Disable it with `--no-progress-bar` if
you are redirecting logs to a file or running in a non-interactive job.

### Useful Evaluation Commands

Minimal smoke test, useful to verify that the checkpoint loads and the report is
written:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --episodes-per-condition 1 \
    --velocity-episodes-per-condition 1 \
    --num-envs 16 \
    --seeds 0 \
    --output-dir logs/dribbling_eval_smoke
```

Full single-checkpoint evaluation:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --episodes-per-condition 1000 \
    --velocity-episodes-per-condition 300 \
    --num-envs 128 \
    --seeds 0 1 2 \
    --output-dir logs/dribbling_eval
```

Large-machine run with more parallel environments:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --episodes-per-condition 1000 \
    --velocity-episodes-per-condition 300 \
    --num-envs 512 \
    --seeds 0 1 2 \
    --output-dir logs/dribbling_eval
```

Run only the main task metrics, skipping the velocity diagnostic:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --no-run-velocity-diagnostic \
    --episodes-per-condition 1000 \
    --num-envs 128 \
    --seeds 0 1 2
```

Run only the velocity diagnostic:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --no-run-main \
    --velocity-episodes-per-condition 300 \
    --num-envs 128 \
    --seeds 0 1 2
```

Run a single-environment evaluation while watching it in realtime. The report
and plots are still saved as usual:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --episodes-per-condition 1 \
    --velocity-episodes-per-condition 1 \
    --num-envs 1 \
    --seeds 0 \
    --view-during-eval \
    --output-dir logs/dribbling_eval
```

Disable plot generation:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --no-save-plots
```

Disable progress bars:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --no-progress-bar
```

Run on CPU:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --no-use-cuda \
    --num-envs 16
```

Run on a specific GPU:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --cuda 1 \
    --num-envs 128
```

## Direct Target And Obstacle Generation

The evaluator generates the target and obstacles directly inside
`evaluate_dribbling.py`, rather than relying on the training curriculum sampler.
The default fixed setup uses a `5.0 m` target ahead of the robot and places
obstacles along the ball-to-target corridor using:

- forward fractions: `(0.3, 0.5, 0.7)`
- lateral offsets: `(-0.35, 0.0, 0.35) m`
- moving-obstacle speed: `0.15 m/s`

These can be changed from the CLI with `--eval-target-distance`,
`--eval-target-heading-offset`, `--eval-obstacle-forward-fractions`,
`--eval-obstacle-lateral-offsets`, and `--eval-obstacle-speed`.

Each fixed value also has a plausible default range variant. The evaluator uses
the fixed values by default. Add `--randomize-target-and-obstacle` to sample
fresh values uniformly per environment whenever a new target trial is generated.
Without that flag, the range values are ignored.

Within one evaluation trial, the sampled target and obstacle layout stay fixed.
The evaluator disables the training command behavior that immediately samples a
new target after target reach. A new target/obstacle layout is generated only
when the evaluator starts the next requested trial. Ball-obstacle and
robot-obstacle contacts are reported as safety metrics, but they do not end the
trial. The evaluator also disables constraint-as-termination and environment
auto-reset during evaluation, so obstacle proximity cannot silently end a trial.
Trials end only on target reach, maximum trial duration, fall, or ball lost.

| Fixed parameter | Randomized range parameter used with `--randomize-target-and-obstacle` |
|---|---|
| `--eval-target-distance` | `--eval-target-distance-range LO HI` |
| `--eval-target-heading-offset` | `--eval-target-heading-offset-range LO HI` |
| `--eval-obstacle-forward-fractions A B C` | `--eval-obstacle-forward-fraction-ranges A_LO A_HI B_LO B_HI C_LO C_HI` |
| `--eval-obstacle-lateral-offsets A B C` | `--eval-obstacle-lateral-offset-ranges A_LO A_HI B_LO B_HI C_LO C_HI` |
| `--eval-obstacle-speed` | `--eval-obstacle-speed-range LO HI` |
| `--eval-lateral-limit` | `--eval-lateral-limit-range LO HI` |

The direct target is generated from the current ball position:

```text
target = ball_xy + eval_target_distance * heading_direction
heading_direction = robot_yaw + eval_target_heading_offset
```

The direct obstacles are generated on the ball-to-target segment:

```text
obstacle_i =
  ball_xy
  + forward_fraction_i * (target_xy - ball_xy)
  + lateral_offset_i * segment_side_direction
```

The condition-specific obstacle roles are:

| Condition | Active obstacles | Scripted roles |
|---|---:|---|
| `no_obstacles` | 0 | none |
| `static_3` | 3 | static, static, static |
| `moving_3` | 3 | ball attacker, lateral blocker, distractor |
| `velocity_no_obstacles` | 0 | none |
| `velocity_single_obstacle` | 1 | ball attacker |

Default target and obstacle generation command:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --eval-target-distance 5.0 \
    --eval-target-heading-offset 0.0 \
    --eval-obstacle-forward-fractions 0.3 0.5 0.7 \
    --eval-obstacle-lateral-offsets -0.35 0.0 0.35 \
    --eval-obstacle-speed 0.15 \
    --eval-lateral-limit 0.45
```

Example with a longer target and wider obstacles:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --eval-target-distance 4.0 \
    --eval-obstacle-forward-fractions 0.35 0.55 0.75 \
    --eval-obstacle-lateral-offsets -0.5 0.0 0.5 \
    --eval-obstacle-speed 0.2
```

Example with a target angled 20 degrees left of robot heading:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --eval-target-heading-offset 0.349
```

Example with randomized target and obstacle placement:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --randomize-target-and-obstacle \
    --eval-target-distance-range 3.0 6.0 \
    --eval-target-heading-offset-range -0.35 0.35 \
    --eval-obstacle-forward-fraction-ranges 0.25 0.4 0.45 0.6 0.65 0.8 \
    --eval-obstacle-lateral-offset-ranges -0.6 -0.2 -0.15 0.15 0.2 0.6 \
    --eval-obstacle-speed-range 0.1 0.25 \
    --eval-lateral-limit-range 0.35 0.6
```

Viewer check with randomized placement:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --viewer-condition moving_3 \
    --num-envs 1 \
    --randomize-target-and-obstacle \
    --eval-target-distance-range 3.0 6.0 \
    --eval-target-heading-offset-range -0.35 0.35 \
    --eval-obstacle-forward-fraction-ranges 0.25 0.4 0.45 0.6 0.65 0.8 \
    --eval-obstacle-lateral-offset-ranges -0.6 -0.2 -0.15 0.15 0.2 0.6 \
    --eval-obstacle-speed-range 0.1 0.25
```

## Viewer Checks

To visually inspect one setup with the MuJoCo viewer, select a single condition
and run one environment:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./logs/<run>/checkpoints/latest.pt \
    --viewer-condition static_3 \
    --num-envs 1
```

Supported viewer conditions are `no_obstacles`, `static_3`, `moving_3`,
`velocity_no_obstacles`, and `velocity_single_obstacle`.

No obstacles:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --viewer-condition no_obstacles \
    --num-envs 1
```

Three fixed obstacles:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --viewer-condition static_3 \
    --num-envs 1
```

Three moving obstacles:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --viewer-condition moving_3 \
    --num-envs 1
```

Velocity diagnostic without obstacles:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --viewer-condition velocity_no_obstacles \
    --num-envs 1
```

Velocity diagnostic with one obstacle:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --viewer-condition velocity_single_obstacle \
    --num-envs 1
```

Use the Viser viewer instead of the native MuJoCo viewer:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --viewer-condition moving_3 \
    --viewer viser \
    --num-envs 1
```

Viewer check with custom direct generation parameters:

```bash
pixi run -e train eval-dribbling \
    task:t1-dribbling \
    --checkpoint ./checkpoints/dribbling_phase2_stage1_scratch.pt \
    --viewer-condition static_3 \
    --num-envs 1 \
    --eval-target-distance 4.0 \
    --eval-obstacle-forward-fractions 0.35 0.55 0.75 \
    --eval-obstacle-lateral-offsets -0.5 0.0 0.5
```

## Main Evaluation Environments

Evaluate the same checkpoint in three fixed environment configurations.

| Environment | Obstacle setup | Purpose |
|---|---|---|
| No obstacles | 0 active obstacles, equivalent to stage 0 | Nominal dribbling retention and sanity check |
| Static obstacles | 3 active static blockers | Out-of-domain obstacle count and clutter test |
| Moving obstacles | 3 active `mixed_attackers`, equivalent to stage 4 | Main out-of-domain moving-obstacle test |

The no-obstacle case is not strictly out of distribution, because the policy
has seen no-obstacle training earlier in the curriculum. It should be reported
as a retention baseline: the policy should not lose nominal dribbling quality
after learning obstacle avoidance and visual adaptation.

The three-obstacle cases are the main generalisation tests for a stage-3
checkpoint, because the checkpoint was trained with only one active obstacle.

## Main Task Metrics

Report the following metrics for each main evaluation environment.

| Metric | Definition | Notes |
|---|---|---|
| Success rate | Fraction of episodes where the ball reaches the sampled target before timeout | Primary task metric |
| Time to target | Time elapsed until target reached | Report success-only mean/median and a timeout-censored value |
| Fall rate | Fraction of episodes ending due to robot fall | Separates task failure from locomotion failure |
| Robot-obstacle collision rate | Fraction of episodes with robot-obstacle contact | Only meaningful in obstacle environments |
| Ball-obstacle collision rate | Fraction of episodes with ball-obstacle contact | Measures whether success relies on unsafe/undesired contact |
| Minimum ball-obstacle clearance | Minimum distance between ball and nearest active obstacle | Report mean and low percentile, e.g. 5th percentile |

For time to target, failures should not simply be ignored. A useful convention
is to report:

- success-only time to target
- timeout-censored time to target, where failed episodes are assigned the full
  episode length, for example `20 s`

This prevents a method with low success rate from looking artificially fast.

## Ball Velocity Tracking Diagnostic

Ball velocity error should be treated as a diagnostic metric, not as the main
measure of task quality.

In the no-obstacle environment, velocity error directly measures how well the
robot executes the commanded ball velocity. Lower error is unambiguously better
in this setting.

In obstacle environments, velocity error is more subtle. A good policy may
intentionally deviate from the desired ball velocity to avoid an obstacle and
still reach the target safely. For this reason, velocity tracking should be
reported separately from the main task metrics.

Recommended diagnostic setups:

| Setup | Purpose |
|---|---|
| No obstacles | Clean velocity tracking quality |
| Single obstacle | Show how the policy trades off tracking and avoidance |

For the single-obstacle setup, split timesteps into:

- unblocked timesteps: no relevant obstacle lies on the ball-to-target corridor
- blocked timesteps: the nearest obstacle lies between the ball and the target

Then report the ball velocity error separately for the two cases. The expected
behavior is:

- low velocity error when the path is unblocked
- larger velocity error when the path is blocked, if the robot is actively
  avoiding the obstacle
- successful and safe target reaching despite this temporary deviation

Possible velocity tracking metrics:

- mean vector error: `||v_ball - v_desired||`
- mean squared vector error: `||v_ball - v_desired||^2`
- speed error: `| ||v_ball|| - ||v_desired|| |`
- angular error between actual and desired ball velocity

Report mean and variance, or mean and standard deviation, over timesteps and
episodes.

## Perception And Depth Encoder Metrics

Perception quality can be evaluated on the same episodes used for task
evaluation, but should be reported as a separate subsection or table.

This makes it possible to relate task failures to perception errors without
mixing control quality and encoder quality into one metric.

Recommended perception metrics:

| Metric | Environments | Notes |
|---|---|---|
| Ball position error | All environments | Error between depth-head prediction and GT ball position |
| Ball velocity error | All environments | Error between depth-head prediction and GT ball velocity |
| Nearest obstacle position error | Obstacle environments only | Error against GT nearest active obstacle |
| Nearest obstacle velocity error | Obstacle environments only | Error against GT nearest active obstacle |
| FOV coverage | All environments | Fraction of timesteps where the ball is inside the camera field of view |
| Valid depth coverage | All environments | Fraction of timesteps where the adaptation mask is active |

For ball and obstacle errors, report mean and standard deviation. Median and
percentiles are also useful if the error distribution has large outliers.

## No-Obstacle Perception Case

In the no-obstacle environment, ball perception should still be evaluated
normally.

Obstacle perception is different: there is no true nearest active obstacle. The
environment uses a far-away sentinel representation when no obstacle is active,
so a standard nearest-obstacle position or velocity error is not very
informative.

Recommended reporting for no-obstacle episodes:

- report ball position and velocity error
- do not report nearest-obstacle error as a main metric
- optionally report a false-obstacle metric, if useful

A possible false-obstacle metric is:

```text
false obstacle rate =
  fraction of timesteps where the predicted nearest obstacle is closer than a
  chosen distance threshold even though no obstacle is active
```

If this threshold is not clearly defined or the obstacle head is not used
directly at inference, it is better to skip this metric and state that obstacle
perception is evaluated only in obstacle scenes.

## FOV And Valid-Depth Coverage

FOV coverage measures how often the ball is geometrically visible to the head
camera.

```text
FOV coverage =
  number of timesteps where the ball is inside the camera FOV
  / total number of timesteps
```

Valid depth coverage measures how often the visual adaptation path is allowed
to use the depth encoder output.

```text
valid depth coverage =
  number of timesteps where the adaptation mask is active
  / total number of timesteps
```

In the current implementation, the adaptation mask is tied to ball visibility
and depth validity. Reporting this value is useful because perception errors
are only meaningful when the visual encoder has valid input.

For example:

- low task performance with low FOV coverage may indicate poor head/ball
  visibility rather than a poor depth encoder
- low task performance with high FOV coverage and high perception error points
  more directly to encoder quality
- high success with temporary FOV loss may indicate that the recurrent encoder
  or policy is robust to brief visual dropouts

Report FOV and valid-depth coverage as mean percentage of episode timesteps.

## Number Of Evaluation Runs

For final reporting, use at least `1000` episodes per environment condition if
simulation cost allows it. This gives a reasonably tight confidence interval on
success rate.

For faster iteration, `300-500` episodes per condition is acceptable, but should
be considered preliminary.

Use multiple random seeds and keep the same seed set across environment
conditions whenever possible. Recommended reporting:

- `3-5` evaluation seeds for one checkpoint
- aggregate mean and standard deviation across seeds
- if claiming training robustness, evaluate `3-5` independently trained
  checkpoints rather than only one checkpoint

## Suggested Tables

### Main Task Table

| Environment | Success rate | Time to target | Fall rate | Robot collision | Ball collision | Min clearance |
|---|---:|---:|---:|---:|---:|---:|
| No obstacles | | | | n/a | n/a | n/a |
| 3 static obstacles | | | | | | |
| 3 moving obstacles | | | | | | |

### Velocity Diagnostic Table

| Setup | Segment | Vector error | Speed error | Angular error |
|---|---|---:|---:|---:|
| No obstacles | all timesteps | | | |
| Single obstacle | unblocked | | | |
| Single obstacle | blocked | | | |

### Perception Table

| Environment | Ball pos error | Ball vel error | Obstacle pos error | Obstacle vel error | FOV coverage | Valid depth coverage |
|---|---:|---:|---:|---:|---:|---:|
| No obstacles | | | n/a | n/a | | |
| 3 static obstacles | | | | | | |
| 3 moving obstacles | | | | | | |
