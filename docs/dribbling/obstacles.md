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

At every step:

```python
target_delta = target_position - ball_position
dir = normalize(target_delta)
speed = clip(speed_gain * ||target_delta||, min_speed, max_speed)
ball_vel_cmd = dir * speed
```

If the ball gets within `target_reached_threshold` of the target, a new target
is sampled immediately.

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
- `cmd_dir`: normalized commanded ball direction
- `side_dir`: perpendicular to `cmd_dir`

These are computed by `_cmd_and_side_dirs()`:
[obstacle_commands.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/obstacle_commands.py:150).

Because `cmd_dir` now comes from the target-driven `ball_vel` command, obstacle
spawn generation is effectively aligned with the current ball-to-target path.

### Static Blocker / Lateral Blocker / Ball Attacker

These all spawn with the same geometric template:

```python
position = ball_xy + forward_dist * cmd_dir + lateral_offset * side_dir
```

where:

- `forward_dist` is sampled from `distance_range`
- `lateral_offset` is sampled from `lateral_offset_range`

This means the obstacle is placed **ahead of the ball-to-target path**, not just at a
random angle around the robot.

That design is intentional:

- it creates a meaningful blocker for dribbling
- it avoids teaching only generic body collision avoidance
- it aligns obstacle pressure with the ball-control task

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
- `far_from_robot`: robot-obstacle distance exceeds `respawn_robot_distance`
- `far_from_ball`: ball-obstacle distance exceeds `respawn_ball_distance`

Role-specific rule:

- `static_blocker`, `lateral_blocker`, `distractor`
  - respawn when `behind OR far_from_robot`
- `ball_attacker`
  - respawn when `behind OR far_from_ball`

This means:

- a static blocker stays static while it is still a meaningful blocker
- once the robot has passed it, or it drifts too far away, it respawns
- dynamic obstacles also keep generating repeated encounters within one episode

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
3. stage 2
   - `behavior = lateral_blocker`
   - `num_active = 1`
4. stage 3
   - `behavior = ball_attacker`
   - `num_active = 1`
5. stage 4
   - `behavior = mixed_attackers`
   - `num_active = 3`

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

## Obstacle Reward

Obstacle reward logic lives in
[rewards.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/mdp/rewards.py:501).

The active obstacle penalty is `obstacle_avoidance(...)`.

It uses the nearest obstacle, but it is no longer gated by the coarse rule
"obstacle in front of the robot."

Instead, the obstacle is considered relevant only if it is still close to the
**commanded ball path toward the current target**.

For the nearest obstacle the reward computes:

- `ball_to_obs = obs_xy - ball_xy`
- `cmd_dir = normalize(ball_vel_cmd)`
- `obs_forward = dot(ball_to_obs, cmd_dir)`
- `obs_lateral = ||ball_to_obs - obs_forward * cmd_dir||`

Interpretation:

- `obs_forward > 0` means the obstacle is still ahead of the ball along the
  commanded target direction
- `obs_lateral` measures how far the obstacle is from that commanded path

So the obstacle contributes only if it is:

- within the relevant collision or direction detection range
- ahead of the ball along the current command
- inside a lateral path tube

This is important because once the robot already moved around the obstacle,
the obstacle should stop influencing the reward even if it is still close in
Euclidean distance.

### Reward Terms

The penalty is the sum of two bounded terms:

1. collision term

```python
progress = clamp((collision_far_distance - dist) / (collision_far_distance - collision_near_distance), 0, 1)
collision_term = progress^2
```

Properties:

- range `[0, 1]`
- `0` when the obstacle is farther than `collision_far_distance`
- `1` when the obstacle is at or closer than `collision_near_distance`
- quadratic shaping in the middle
- much less aggressive than the previous exponential close-range penalty
- active only when the obstacle remains inside the **collision path tube**

2. direction term

This measures whether the commanded ball direction points toward the obstacle.

It computes:

- command direction in world frame
- direction from ball to obstacle in world frame
- cosine alignment
- keep only positive alignment

Then:

```python
direction_term = toward_obstacle^p
```

where `p = direction_sharpness / 2` in the current implementation.

Properties:

- range `[0, 1]`
- near 0 if the commanded ball direction is not toward the obstacle
- near 1 if the command points directly at the obstacle
- only active when the nearest obstacle is already near and still inside the
  **direction path tube**

### Final Penalty

```python
penalty = engagement * (
    collision_weight * collision_term +
    direction_weight * direction_term
)
```

where:

```python
collision_relevant = 1[
    min_dist <= collision_detection_range and
    obs_forward > 0 and
    obs_lateral <= collision_tube_radius
]

direction_relevant = 1[
    min_dist <= direction_detection_range and
    obs_forward > 0 and
    obs_lateral <= direction_tube_radius
]
```

and `engagement` decreases obstacle pressure when the robot is not really in
control of the ball anymore.

Current defaults give a maximum raw penalty of:

```python
collision_weight + direction_weight = 0.5 + 1.5 = 2.0
```

With reward weight `-3.0` in
[reward_cfg.py](/home/valeriospagnoli/SPQR/colosseum/src/colosseum/tasks/dribbling/config/t1_23dof/reward_cfg.py:87),
the worst-case obstacle contribution is approximately `-6.0`, but only when:

- the obstacle is still close to the robot
- the obstacle is still ahead on the current target path
- the obstacle is inside the relevant path tube
- and the ball is still closely engaged

So the large weight is much more localized than before.

Current default obstacle-reward parameters are:

- `collision_near_distance = 0.5`
- `collision_far_distance = 1.5`
- `collision_detection_range = 1.5`
- `direction_detection_range = 3.0`
- `collision_tube_radius = 0.75`
- `direction_tube_radius = 1.0`
- `direction_sharpness = 3.0`
- `collision_weight = 0.5`
- `direction_weight = 1.5`
- `ball_engagement_near_distance = 0.3`
- `ball_engagement_far_distance = 0.75`

## Active Minimal Reward Setup

The active task reward setup is intentionally simple:

- `ball_vel_tracking`
- `ball_vel_norm`
- `ball_vel_angle`
- `robot_ball_distance`
- `robot_ball_approach_vel`
- `robot_ball_yaw`
- `obstacle_avoidance`

In this setup:

- the robot always keeps the standard no-obstacle ball-carrying incentives
- obstacles affect behavior only through the nearest-obstacle local penalty
- there is no active danger-based tightening, progress reward, or protection reward

This is meant to preserve the baseline dribbling behavior and only relax the
desired motion locally when the closest obstacle is still on the current
ball-to-target path corridor.

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
  and still lies inside the commanded path tube
- the standard ball-carrying rewards remain active even in obstacle scenes
- the desired ball velocity is always induced by a persistent world-frame target
- obstacle stages can now be pinned from the CLI for debugging and checkpointing

In short:

- command side: persistent target, target-driven world-frame ball velocity
- simulator side: multi-obstacle staged behaviors
- policy side: nearest-obstacle compact representation
- reward side: nearest path-relevant obstacle with bounded collision + direction penalty
