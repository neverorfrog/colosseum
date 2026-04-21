# Dribbling Obstacles

This document explains how obstacles are currently handled in the dribbling task.

It covers:

- how the ball command is generated from a persistent target
- how obstacle entities are created in the simulator
- how obstacle positions and velocities are generated at runtime
- how obstacle difficulty is controlled by the curriculum
- how obstacles are exposed to the policy through observations and encoders
- how obstacles affect rewards
- how obstacle stage selection works in training and play

## Overview

The dribbling task now uses a **closest-obstacle policy representation** together
with a **staged physical obstacle curriculum**.

This means:

- the simulator can contain up to three physical obstacle entities
- the policy/encoder side only tracks the **nearest active obstacle**
- reward terms also use the nearest obstacle
- curriculum difficulty is increased by changing the obstacle behavior over time

So there are two distinct layers:

1. physical obstacle scene in simulation
2. reduced obstacle signal used by learning

There is also one important task-design choice shared by all stages:

- the command exposed as `ball_vel` is still a world-frame ball velocity
- but that velocity is now recomputed every step from a persistent world-frame target
- so the task is "bring the ball toward this target" rather than "track a free velocity until resample"

## Files Involved

Main files:

- ball target / command generation:
  [src/colosseum/tasks/dribbling/mdp/ball_velocity_command.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/ball_velocity_command.py:1)
- obstacle scene definition:
  [src/colosseum/tasks/dribbling/obstacle_spec.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/obstacle_spec.py:1)
- obstacle runtime command and motion:
  [src/colosseum/tasks/dribbling/mdp/obstacle_commands.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/obstacle_commands.py:1)
- obstacle curriculum:
  [src/colosseum/tasks/dribbling/mdp/curriculum.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/curriculum.py:117)
- obstacle curriculum config:
  [src/colosseum/tasks/dribbling/config/t1_23dof/cact_cfg.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/config/t1_23dof/cact_cfg.py:100)
- obstacle observations:
  [src/colosseum/tasks/dribbling/mdp/observations.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/observations.py:83)
- obstacle reward logic:
  [src/colosseum/tasks/dribbling/mdp/rewards.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/rewards.py:468)
- task-level environment assembly:
  [src/colosseum/tasks/dribbling/config/t1_23dof/t1_dribbling_cfg.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/config/t1_23dof/t1_dribbling_cfg.py:141)
- obstacle privileged observations in the encoder path:
  [src/colosseum/tasks/dribbling/config/t1_23dof/observation_cfg.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/config/t1_23dof/observation_cfg.py:78)
- RMA obstacle encoding:
  [src/colosseum/tasks/dribbling/mdp/rma_terms.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/rma_terms.py:1)
- phase-2 obstacle head:
  [src/colosseum/algorithm/encoders.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/algorithm/encoders.py:187)

## Ball Command And Target

The dribbling task still uses a `ball_vel` command, but it is no longer sampled
as a free local velocity. Instead, `BallVelocityCommand` now samples a
persistent world-frame target for the ball and recomputes the desired velocity
from the current ball position toward that target.

At resample time:

- a target heading is sampled around the robot forward direction
- a target distance is sampled from `target_distance_range`
- a world-frame target position is created relative to the current robot position
- if enabled, the obstacle command is also forced to resample for the same envs
  so obstacle placement stays synchronized with the new target

At every step:

```python
target_delta = target_position - ball_position
dir = normalize(target_delta)
speed = clip(speed_gain * ||target_delta||, min_speed, max_speed)
ball_vel_cmd = dir * speed
```

If the ball gets within `target_reached_threshold` of the target, a new target
is sampled immediately.

When `resample_obstacles_on_target_reset=True`, target resampling also calls the
adversary command's public `resample_for_env_ids(...)` hook. This keeps the
obstacle scene aligned with the current ball-to-target corridor not only at
episode reset, but also at timed command resamples and target-reached events.

This is implemented in
[ball_velocity_command.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/ball_velocity_command.py:22).

This design is especially important for obstacle avoidance:

- the desired velocity always points toward a meaningful long-horizon target
- temporary detours remain consistent with the task objective
- obstacle spawning, obstacle rewards, and velocity tracking all stay aligned
  with the same ball-to-target path

### Viewer visualization

The viewer now shows:

- green arrow: desired ball velocity command
- red arrow: actual ball velocity
- blue sphere: current global target for the ball
- blue arrow: direction from the ball to the target

So you can directly see whether the robot is making progress toward the target
or getting stuck while the target remains unchanged.

## Obstacle Entities In The Simulator

Obstacle geometry is defined in
[obstacle_spec.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/obstacle_spec.py:1).

Each obstacle is a fixed-base cylinder:

- height: `1.2 m`
- radius: `0.15 m`

The helper:

- `get_obstacle_spec()` builds the MuJoCo XML spec
- `get_obstacle_cfg(index)` creates the scene entity config

The task scene creates one entity per obstacle slot:

- `obstacle_0`
- `obstacle_1`
- `obstacle_2`

This happens in
[t1_dribbling_cfg.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/config/t1_23dof/t1_dribbling_cfg.py:86).

So the simulator always has up to `NUM_OBSTACLES = 3` physical obstacle bodies,
even if only some of them are active in the current curriculum stage.

## Runtime Obstacle Control

Obstacle runtime behavior is handled by `ObstacleCommand` in
[obstacle_commands.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/obstacle_commands.py:37).

It stores:

- `_positions_w`: world-frame XY obstacle positions
- `_velocities_w`: world-frame XY obstacle velocities
- `_speed_targets`: per-obstacle target speeds
- `_lateral_signs`: sign for lateral blocker movement
- `_tangent_mix`: tangential bias for ball attackers
- `_random_dirs`: distractor motion directions
- `_resample_timers`: per-obstacle timers for speed/direction resampling

The command tensor exposed to the rest of the system is:

- flattened world-frame XY positions

but the richer API used by rewards and observations is:

- `obstacle_positions_w`
- `obstacle_velocities_w`

## Obstacle Behaviors

Obstacle behavior is controlled by `ObstacleCommandCfg.behavior` in
[obstacle_commands.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/obstacle_commands.py:323).

Supported behaviors:

- `none`
- `static_blocker`
- `lateral_blocker`
- `ball_attacker`
- `mixed_attackers`

For `mixed_attackers`, obstacle roles are fixed by index:

- obstacle 0 -> `ball_attacker`
- obstacle 1 -> `lateral_blocker`
- obstacle 2 -> `distractor`

This role mapping is implemented in
`_role_for_obstacle()`:
[obstacle_commands.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/obstacle_commands.py:163).

## Obstacle Spawn Generation

Obstacle positions are generated in `_sample_spawn_positions()`:
[obstacle_commands.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/obstacle_commands.py:173).

The main reference quantities are:

- `ball_xy`: current ball position in world XY
- `target_xy`: current persistent ball target in world XY
- `seg_dir`: normalized ball-to-target segment direction
- `seg_side`: perpendicular to `seg_dir`

So obstacle spawn generation is explicitly aligned with the current
ball-to-target path, not just with a locally sampled velocity heading.

### Static Blocker / Lateral Blocker / Ball Attacker

These all spawn with the same corridor-coupled template.

The command term first computes the current ball-to-target segment:

```python
seg = target_xy - ball_xy
seg_len = ||seg||
seg_dir = normalize(seg)
seg_side = perp(seg_dir)
```

Then the obstacle is sampled as:

```python
position = ball_xy + forward_fraction * seg_len * seg_dir + lateral_offset * seg_side
```

where:

- `forward_fraction` is sampled from `forward_fraction_range`
- `lateral_offset` is sampled from `lateral_offset_range`

So blockers and attackers are not spawned at an arbitrary metric distance ahead
of the ball anymore. They are spawned at a controlled fraction of the current
ball-to-target segment, typically inside the middle part of that segment.

Current defaults are:

- `forward_fraction_range = (0.35, 0.75)`
- `lateral_offset_range = (-0.3, 0.3)`
- `lateral_offset_range = (-0.3, 0.3)`

This is intentional:

- the obstacle is guaranteed to lie on a meaningful part of the current path
- the lateral spread is narrow enough to keep the obstacle inside the reward's
  direction tube (`direction_tube_radius = 0.5` in the current reward config)
- target resampling and obstacle resampling stay geometrically consistent

### Distractor

The distractor uses a different rule:

```python
position = ball_xy + radius * random_dir
```

where `random_dir` is sampled uniformly over the circle.

This obstacle exists mainly to create clutter in the final multi-obstacle stage.

## Obstacle Velocity Generation

Obstacle velocities are generated from behavior-specific target directions and
random target speeds, then smoothed over time.

This logic is in:

- `_resample_motion_state()`:
  [obstacle_commands.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/obstacle_commands.py:213)
- `_compute_velocity_target()`:
  [obstacle_commands.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/obstacle_commands.py:240)

### Speed Sampling

Moving obstacles do not keep a fixed speed forever.

Instead:

- a target speed is sampled uniformly from `[min_speed, max_speed]`
- a resample timer is drawn from `velocity_resample_time_range`
- when the timer expires, a new target speed is sampled

This creates time-varying obstacle motion while avoiding hard discontinuities at
every physics step.

### Smoothing

Actual velocity is updated with a first-order smoothing rule:

```python
v_next = (1 - alpha) * v_current + alpha * v_target
```

where `alpha = velocity_smoothing`.

This reduces jerky motion and makes the dynamic obstacles easier to learn from.

### Per-Behavior Motion

#### `static_blocker`

- speed target is zero
- actual velocity is zero

So the obstacle stays fixed after spawn, but it can still respawn once the
robot has passed it or it is otherwise no longer relevant.

#### `lateral_blocker`

Velocity target is:

```python
v_target = sign * side_dir * speed
```

where `sign` is resampled between `-1` and `+1`.

So the obstacle slides laterally across the commanded ball path.

#### `ball_attacker`

Velocity target is computed from:

- direction from obstacle to ball
- a tangential bias

The target direction is:

```python
to_ball_dir + tangent_mix * tangent_dir
```

then normalized.

So the obstacle does **not** simply move straight toward the robot.
It moves toward the ball with a small random angular variation, which matches
the idea of “attacking the ball.”

#### `distractor`

Velocity target is:

```python
v_target = random_dir * speed
```

where `random_dir` is resampled.

This makes the distractor move independently from the ball.

## Reset And Update Lifecycle

Obstacle command lifecycle:

### On reset

Handled by `_resample_command()`:
[obstacle_commands.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/obstacle_commands.py:77)

For each obstacle:

- if active:
  - sample spawn position
  - sample motion state
  - compute initial velocity target
  - write pose to simulation
- if inactive:
  - park it far away

### Each environment step

Handled by `_update_command()`:
[obstacle_commands.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/obstacle_commands.py:97)

For each obstacle:

- if inactive:
  - park it far away
- if parked but should be active:
  - respawn it
- if active but no longer relevant to the current encounter:
  - respawn it in a new random stage-consistent location
- update resample timer
- if timer expired:
  - resample motion state
- compute behavior-dependent velocity target
- smooth current velocity toward target
- integrate position:
  `position += velocity * dt`
- write pose to MuJoCo mocap body

### Respawn On Irrelevance

Obstacles are not meant to appear once and then remain irrelevant until the
episode ends.

This is handled by `_needs_respawn()` in
[obstacle_commands.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/obstacle_commands.py:289).

For each active obstacle, the command term checks whether the current obstacle
encounter is effectively over. If it is, the obstacle is respawned
immediately with the same stage behavior but a new random spawn position.

The relevance checks use:

- the obstacle position in robot body frame
- the robot-obstacle distance
- the ball-obstacle distance

Current respawn criteria:

- `behind`: obstacle body-frame `x < respawn_behind_x_threshold`
- `far_from_ball`: ball-obstacle distance exceeds `respawn_ball_distance`

Role-specific rule:

- `static_blocker`, `lateral_blocker`, `distractor`
  - respawn when `behind`
- `ball_attacker`
  - respawn when `behind OR far_from_ball`

This means:

- a static blocker stays fixed on the sampled corridor until the encounter is
  genuinely over or the target changes
- blockers are no longer removed just because the robot is far away
- dynamic obstacles also keep generating repeated encounters within one episode

Because target resampling now also triggers obstacle resampling, blockers also
get refreshed automatically whenever the target changes.

So obstacle training is denser within an episode and does not depend only on
episode resets.

## Parking Inactive Obstacles

Inactive obstacles are not deleted from the scene.
They are moved far away:

- `_PARK_FAR = 1000.0`

This happens in `_park_obstacle()`:
[obstacle_commands.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/obstacle_commands.py:278).

Why this is done:

- keep the observation tensor shape fixed
- keep the scene layout stable
- make inactive obstacles irrelevant to rewards and nearest-obstacle selection

## Obstacle Curriculum

Obstacle difficulty is controlled by `obstacle_curriculum`:
[curriculum.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/curriculum.py:117)

It sets:

- `num_active`
- `behavior`
- `distance_range`
- `lateral_offset_range`
- `forward_fraction_range`
- `min_speed`
- `max_speed`
- `velocity_resample_time_range`

The current staged curriculum is defined in
[cact_cfg.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/config/t1_23dof/cact_cfg.py:100).

Stages:

1. stage 0
   - `behavior = none`
   - `num_active = 0`
2. stage 1
   - `behavior = static_blocker`
   - `num_active = 1`
   - `lateral_offset_range = (-0.3, 0.3)`
   - `forward_fraction_range = (0.35, 0.75)`
3. stage 2
   - `behavior = lateral_blocker`
   - `num_active = 1`
   - `lateral_offset_range = (-0.3, 0.3)`
   - `forward_fraction_range = (0.35, 0.75)`
4. stage 3
   - `behavior = ball_attacker`
   - `num_active = 1`
   - `lateral_offset_range = (-0.3, 0.3)`
   - `forward_fraction_range = (0.35, 0.75)`
5. stage 4
   - `behavior = mixed_attackers`
   - `num_active = 3`
   - blockers/attackers use `lateral_offset_range = (-0.3, 0.3)`
   - blockers/attackers use `forward_fraction_range = (0.35, 0.75)`
   - distractors still use `distance_range` for random-angle spawn

Curriculum metrics that are logged:

- `obstacle_stage_index`
- `obstacle_behavior_id`
- `num_active_obstacles`
- `obstacle_min_dist_m`
- `obstacle_min_speed`
- `obstacle_max_speed`

These are useful to know which stage was active when a checkpoint was saved.

## Forcing A Stage From The CLI

Obstacle stage can be controlled from the CLI with:

```bash
--obstacle-stage-index <idx>
```

Supported values:

- `-1`: use curriculum
- `0..4`: pin the stage

This is wired through the shared experiment config and the dribbling task config:

- [experiment.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/config/types/experiment.py:18)
- [t1_dribbling_cfg.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/config/t1_23dof/t1_dribbling_cfg.py:141)

Behavior:

- training:
  - `-1` -> use obstacle curriculum
  - `>= 0` -> disable obstacle curriculum progression and pin one stage
- play/evaluation:
  - `-1` -> keep obstacle curriculum active during play
  - `>= 0` -> pin one stage

Pinning a stage does not freeze one single obstacle instance forever. It only
freezes the stage behavior. Obstacles can still respawn within the episode
according to the relevance rules above.

## Obstacle Observations

Obstacle observations are computed in
[observations.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/observations.py:83).

Two observation functions exist:

- `obstacle_position_b(env)` -> nearest obstacle position in robot body frame
- `obstacle_velocity_b(env)` -> nearest obstacle velocity in robot body frame

Important points:

- only the **nearest active obstacle** is used
- positions and velocities are converted to **robot local/body frame**
- if no obstacle is active, a far-away sentinel is returned for position and
  zero for velocity

This means the policy side never receives a list of top-k obstacles anymore.

## How The Nearest Obstacle Is Selected

Nearest-obstacle selection in observations works as follows:

1. get all active obstacle world positions from `ObstacleCommand`
2. compute robot-obstacle Euclidean distances in XY
3. take `argmin`
4. transform that obstacle into body frame

This is implemented in:

- `obstacle_position_b()`:
  [observations.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/observations.py:83)
- `obstacle_velocity_b()`:
  [observations.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/observations.py:114)

## Encoder Usage: Phase 1 And Phase 2

Obstacle observations are added to the privileged observation group in
[observation_cfg.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/config/t1_23dof/observation_cfg.py:84).

That group is called:

- `privileged_obstacles`

It currently has dimension 4:

- `[x, y, vx, vy]` of the nearest obstacle in body frame

### Phase 1

In phase 1, the privileged encoder consumes:

- ball privileged info
- nearest obstacle privileged info

This happens in `DribblingRmaTerm.encode_privileged()`:
[rma_terms.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/rma_terms.py:183)

So phase 1 learns from **ground-truth nearest obstacle state in body frame**.

### Phase 2

In phase 2:

- the actor is frozen
- the depth encoder is trained to match the phase-1 latent
- the obstacle head predicts nearest obstacle `[x, y, vx, vy]`

The obstacle head is defined in
[encoders.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/algorithm/encoders.py:187).

The supervision path is in
[rma_terms.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/rma_terms.py:312).

So phase 2 is trained against the same nearest-obstacle body-frame target used
in phase 1.

## Reward Structure Around Obstacles

Reward logic lives in
[rewards.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/rewards.py:445)
and the active weights/parameters are configured in
[reward_cfg.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/config/t1_23dof/reward_cfg.py:1).

The current obstacle-aware task shaping is built from four complementary terms:

1. `robot_obstacle_collision(...)`
2. `ball_obstacle_collision(...)`
3. `obstacle_direction(...)`
4. `ball_target_progress(...)`

The first three are obstacle-related directly. The fourth is not an obstacle
penalty by itself, but it is explicitly gated by target-obstacle proximity, so
it is part of the obstacle-handling logic.

### `robot_obstacle_collision`

This is the **robot-body safety** term.

It uses the nearest active obstacle and depends only on robot-obstacle
distance. It does not care whether the obstacle is still between the ball and
the target. If the robot body is close to an obstacle, this penalty stays on.

The shaping is quadratic:

```python
progress = clamp((collision_far_distance - dist) / (collision_far_distance - collision_near_distance), 0, 1)
collision_term = progress^2
```

Properties:

- range `[0, 1]`
- `0` when the nearest obstacle is beyond `collision_detection_range`
- `1` when the robot is at or inside `collision_near_distance`
- quadratic growth between `collision_far_distance` and `collision_near_distance`
- unconditional with respect to ball engagement

Current default parameters:

- `collision_detection_range = 1.5`
- `collision_near_distance = 0.5`
- `collision_far_distance = 1.5`
- reward weight `= -5.0`

### `ball_obstacle_collision`

This is the **ball safety** term.

It penalizes the ball getting too close to any active obstacle, again using the
minimum ball-obstacle distance over the active set.

The shaping is the same quadratic form:

```python
progress = clamp((collision_far_distance - dist) / (collision_far_distance - collision_near_distance), 0, 1)
collision_term = progress^2
```

Properties:

- range `[0, 1]`
- active even if the robot body itself is safe
- intended to discourage trapping, scraping, or kicking the ball into the obstacle
- unconditional with respect to ball engagement

Current default parameters:

- `collision_detection_range = 1.0`
- `collision_near_distance = 0.15`
- `collision_far_distance = 0.6`
- reward weight `= -3.0`

### `obstacle_direction`

This is the **path-blocking direction** term.

It is active only when the nearest obstacle lies on the current ball-to-target
corridor. For the nearest obstacle it computes:

- `target_vec = target_xy - ball_xy`
- `target_dist = ||target_vec||`
- `target_dir = normalize(target_vec)`
- `ball_to_obs = obs_xy - ball_xy`
- `obs_forward = dot(ball_to_obs, target_dir)`
- `obs_lateral = ||ball_to_obs - obs_forward * target_dir||`

The obstacle is direction-relevant only if:

- `0 < obs_forward < target_dist`
- `obs_forward <= direction_detection_range`
- `obs_lateral <= direction_tube_radius`

So this term turns off once the obstacle is no longer between the ball and the
target, even if the obstacle is still nearby in a purely local sense.

Its magnitude is based on how much the commanded ball direction points toward
the obstacle:

```python
toward_obstacle = max(0, dot(cmd_dir, normalize(ball_to_obs)))
direction_term = toward_obstacle^p
```

with `p = direction_sharpness / 2`.

Properties:

- range `[0, 1]`
- near `0` if the command is not directed toward the obstacle
- near `1` if the command points directly at the obstacle
- multiplied by a ball-engagement gate, so it matters most when the robot is
  actually controlling the ball

Current default parameters:

- `direction_detection_range = 3.0`
- `direction_tube_radius = 0.5`
- `direction_sharpness = 3.0`
- `ball_engagement_near_distance = 0.3`
- `ball_engagement_far_distance = 0.75`
- reward weight `= -2.5`

### `ball_target_progress`

This is the **anti-stall progress** term.

It rewards positive ball velocity along the current ball-to-target direction:

```python
progress_speed = max(0, dot(ball_vel, target_dir))
progress_reward = clamp(progress_speed / speed_ref, 0, 1)
```

However, it is not always active. It is multiplied by two gates:

1. a **near-target gate**
2. a **target-obstacle gate**

The near-target gate smoothly disables the reward when the ball is already very
close to the target, so the policy is not over-constrained near completion.

The target-obstacle gate smoothly disables the reward when the target itself is
too close to the nearest obstacle, so the policy is not forced to keep pushing
directly into a blocked target neighborhood.

Current default parameters:

- `target_near_distance = 0.15`
- `target_far_distance = 0.5`
- `target_obstacle_near_distance = 0.3`
- `target_obstacle_far_distance = 0.8`
- `speed_ref = 1.0`
- reward weight `= 2.0`

## Active Task Reward Setup

The current task-level shaping around obstacles is:

- `ball_vel_tracking`
- `ball_vel_norm`
- `ball_vel_angle`
- `robot_ball_distance`
- `robot_ball_yaw`
- `robot_ball_approach_vel`
- `ball_target_progress`
- `robot_obstacle_collision`
- `ball_obstacle_collision`
- `obstacle_direction`

In this setup:

- baseline dribbling behavior is still driven by ball tracking, ball control,
  and robot-ball geometry
- the robot body is discouraged from colliding with obstacles
- the ball is discouraged from colliding with obstacles
- the command is discouraged from pointing through an obstacle that actually
  blocks the current ball-to-target corridor
- forward progress toward the target is rewarded, but this progress pressure is
  removed when the target is already effectively reached or when the target is
  too close to an obstacle

This is the current intended behavior:

- if the obstacle is irrelevant, the robot should keep dribbling toward the target
- if the obstacle is locally dangerous, local collision terms should protect both robot and ball
- if the obstacle blocks the ball-target corridor, the direction term should
  encourage a detour
- if the target itself is effectively blocked by an obstacle, the pure progress
  term should back off instead of forcing a bad push straight into the blocker

## Frame Conventions

Obstacle math uses a mix of world frame and body frame, but it is internally
consistent.

### Body frame

Used for:

- policy obstacle observations
- phase-1 privileged obstacle input
- phase-2 obstacle supervision target

### World frame

Used for:

- obstacle storage and motion in `ObstacleCommand`
- robot-obstacle distance computations
- ball-to-obstacle direction in the reward
- command-direction alignment in the reward
- forward/lateral projection of the obstacle onto the commanded path
- target generation and ball-to-target velocity command computation

This is mathematically consistent because:

- distances are invariant to rotation
- dot products are only taken between vectors expressed in the same frame
- path relevance is evaluated in world frame using the same commanded
  ball-to-target direction used by the task itself

The ball command follows the same rule:

- command generation happens in world frame from `ball -> target`
- actor observations use `ball_vel_command_body()` to rotate that command into body frame
- rewards use the same underlying command in whichever frame is appropriate for the term

So the target-based command changes the task semantics, but it does not
introduce a frame mismatch.

## Training / Evaluation Semantics

There are two different notions that should not be confused:

- `phase`
  - phase 1: policy learning with privileged info
  - phase 2: adaptation encoder learning
- `obstacle_stage_index`
  - obstacle curriculum stage

Checkpoint metadata now records both:

- `metadata.phase`
- `metadata.obstacle_stage_index`

Meaning:

- `phase = 1, obstacle_stage_index = -1`
  -> phase-1 run using obstacle curriculum
- `phase = 1, obstacle_stage_index = 0`
  -> phase-1 run with fixed no-obstacle stage
- `phase = 2, obstacle_stage_index = 4`
  -> phase-2 run pinned to stage 4

## Practical Summary

The current obstacle system is designed around these principles:

- physical scene can contain multiple obstacles
- learning sees only the nearest obstacle
- early curriculum stages are simple and gait-friendly
- later stages create ball-centric pressure, not just robot-body pressure
- obstacle reward is bounded and only active when the nearest obstacle is near
  or when it still lies on the commanded ball-target tube, depending on the term
- the standard ball-carrying rewards remain active even in obstacle scenes
- the desired ball velocity is always induced by a persistent world-frame target
- obstacle stages can now be pinned from the CLI for debugging and checkpointing

In short:

- command side: persistent target, target-driven world-frame ball velocity
- simulator side: multi-obstacle staged behaviors
- policy side: nearest-obstacle compact representation
- reward side: nearest obstacle with split collision-safety and path-blocking direction penalties
