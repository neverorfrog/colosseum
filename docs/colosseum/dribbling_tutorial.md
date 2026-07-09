# Tutorial: Creating a Research Task

This tutorial walks through building a full colosseum research task using
the dribbling task as a reference. By the end you'll understand how to
structure a new task, including the scene, MDP config, and an RMA encoder
for privileged observations.

For the simpler CartPole case (no RMA, no custom commands), see
[Cartpole Tutorial](cartpole.md).

---

## What we're building

A task where the T1 robot dribbles a soccer ball toward a target, with
privileged ball-position observations fed through an RMA encoder. This is a
simplified version of the full dribbling task (no obstacles, no curriculum).

**Files we'll create:**

```
src/colosseum/tasks/my_task/
├── __init__.py
└── config/
    └── t1/
        ├── scene_cfg.py          # robot + ball
        ├── observation_cfg.py    # actor and privileged obs groups
        ├── reward_cfg.py         # task + regularization rewards
        ├── event_cfg.py          # reset events
        ├── command_cfg.py        # ball velocity command
        └── t1_my_task_cfg.py     # ColosseumEnvCfg + task registration
```

---

## Step 1 — Auto-discovery

`src/colosseum/tasks/__init__.py` automatically imports every sub-package it
finds with `pkgutil.iter_modules`. That means **no `pyproject.toml` entry
point is needed** — just place your task folder under `src/colosseum/tasks/`
and the `@register_task` decorator takes care of the rest.

The only required file at the top level of the task folder is `__init__.py`.
It can be empty, or import the task class to make it easier to use from other
modules:

```python
# src/colosseum/tasks/my_task/__init__.py
from colosseum.tasks.my_task.config.t1.t1_my_task_cfg import MyTask  # noqa: F401
```

---

## Step 2 — Scene configuration

The scene declares entities (robot, ball), terrain, and sensors. All T1
tasks share the same robot entity config from `t1/constants.py`.

```python
# config/t1/scene_cfg.py
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.sim import MujocoCfg, SimulationCfg
from colosseum.robots.t1.constants import BASE_BODY_NAME, get_robot_cfg
from colosseum.robots.t1.sensors import (
    FEET_GROUND_CONTACT_SENSOR,
    NONFOOT_GROUND_CONTACT_SENSOR,
)
from colosseum.assets.ball.ball_spec import get_ball_cfg


def scene_cfg(play: bool = False) -> SceneCfg:
    return SceneCfg(
        terrain=TerrainEntityCfg(),
        entities={
            "robot": get_robot_cfg(),
            "ball": get_ball_cfg(),
        },
        sensors=(
            FEET_GROUND_CONTACT_SENSOR,
            NONFOOT_GROUND_CONTACT_SENSOR,
        ),
        num_envs=1,
        env_spacing=10.0,
    )


def viewer_cfg() -> ViewerConfig:
    return ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name=BASE_BODY_NAME,
        distance=3.0,
        elevation=-5.0,
        azimuth=90.0,
    )


def sim_cfg() -> SimulationCfg:
    return SimulationCfg(
        nconmax=80,
        njmax=1500,
        contact_sensor_maxmatch=500,
        mujoco=MujocoCfg(
            timestep=0.005,
            iterations=10,
            ls_iterations=20,
            ccd_iterations=50,
        ),
    )
```

---

## Step 3 — Commands

The ball-velocity command computes a desired ball velocity at each step
toward a persistent world-frame target. It is reusable directly from the
dribbling task:

```python
# config/t1/command_cfg.py
from colosseum.tasks.dribbling.mdp.ball_velocity_command import BallVelocityCommandCfg

commands = {
    "ball_vel": BallVelocityCommandCfg(
        asset_name="ball",
        robot_name="robot",
        target_distance_range=(3.0, 8.0),
        min_speed=0.3,
        max_speed=1.5,
        resampling_time_range=(15.0, 20.0),
    ),
}
```

---

## Step 4 — Observations

Observations are split into groups. The `actor` group is what the deployed
policy sees. The `privileged_ball` group is ground-truth consumed by the
RMA privileged encoder during Phase 1 — it must **not** appear in the actor
group.

```python
# config/t1/observation_cfg.py
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp import (
    joint_pos_rel, joint_vel_rel, last_action,
    projected_gravity, base_lin_vel, base_ang_vel,
)
from colosseum.tasks.dribbling.mdp.observations import (
    ball_pos_robot_frame,
    generated_commands,
)

observations = {
    "actor": ObservationGroupCfg(
        enable_corruption=True,
        terms={
            "base_lin_vel": ObservationTermCfg(
                func=base_lin_vel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise={"std": 0.1},
            ),
            "base_ang_vel": ObservationTermCfg(
                func=base_ang_vel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise={"std": 0.2},
            ),
            "projected_gravity": ObservationTermCfg(
                func=projected_gravity,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise={"std": 0.05},
            ),
            "joint_pos": ObservationTermCfg(
                func=joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise={"std": 0.01},
            ),
            "joint_vel": ObservationTermCfg(
                func=joint_vel_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise={"std": 0.5},
            ),
            "last_action": ObservationTermCfg(func=last_action),
            "ball_vel_cmd": ObservationTermCfg(
                func=generated_commands,
                params={"command_name": "ball_vel"},
            ),
        },
    ),
    "privileged_ball": ObservationGroupCfg(
        enable_corruption=False,
        terms={
            "ball_pos": ObservationTermCfg(
                func=ball_pos_robot_frame,
                params={
                    "ball_cfg": SceneEntityCfg("ball"),
                    "robot_cfg": SceneEntityCfg("robot"),
                },
            ),
        },
    ),
}
```

---

## Step 5 — Rewards

Minimal reward set — task tracking rewards plus the standard locomotion
regularization terms shared across all T1 tasks:

```python
# config/t1/reward_cfg.py
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp import action_rate_l2, joint_pos_limits
from mjlab.tasks.velocity.mdp import flat_orientation
from colosseum.tasks.dribbling.mdp.rewards import (
    ball_vel_tracking_relaxed,
    ball_vel_norm_relaxed,
    robot_ball_distance,
    robot_ball_yaw_body,
)
from colosseum.robots.t1.constants import BASE_BODY_NAME

rewards = {
    "ball_vel_tracking": RewardTermCfg(
        func=ball_vel_tracking_relaxed,
        weight=3.0,
        params={"command_name": "ball_vel", "sharpness": 1.5},
    ),
    "ball_vel_norm": RewardTermCfg(
        func=ball_vel_norm_relaxed,
        weight=2.0,
        params={"command_name": "ball_vel", "sharpness": 1.5},
    ),
    "robot_ball_distance": RewardTermCfg(
        func=robot_ball_distance,
        weight=1.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "asset_cfg": SceneEntityCfg("robot"),
            "near_distance": 0.5,
            "far_distance": 2.0,
        },
    ),
    "robot_ball_yaw": RewardTermCfg(
        func=robot_ball_yaw_body,
        weight=0.5,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
        },
    ),
    # Locomotion regularization (always include these)
    "upright": RewardTermCfg(
        func=flat_orientation,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME)},
    ),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.1),
    "dof_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0),
}
```

---

## Step 6 — Assemble the env config and register

Everything is assembled in one file. The `@register_task` decorator from
`colosseum.config.types.task` instantiates the class and adds it to the
registry at import time — no further wiring is needed.

```python
# config/t1/t1_my_task_cfg.py
from dataclasses import dataclass, field
from colosseum.config.types.task import TaskConfig, register_task
from colosseum.envs.colosseum_env import ColosseumEnvCfg
from colosseum.research.dribbling.rma_terms import DribblingRmaTermCfg

from .scene_cfg import scene_cfg, viewer_cfg, sim_cfg
from .observation_cfg import observations
from .reward_cfg import rewards
from .command_cfg import commands
from .event_cfg import events
from .cact_cfg import actions, terminations, curriculum


def my_task_env_cfg(play: bool = False) -> ColosseumEnvCfg:
    cfg = ColosseumEnvCfg(
        scene=scene_cfg(play=play),
        sim=sim_cfg(),
        viewer=viewer_cfg(),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics={},
        decimation=4,
        episode_length_s=20.0,
        encoders={
            "ball": DribblingRmaTermCfg(
                privileged_obs_group="privileged_ball",
                obstacle_privileged_obs_group=None,  # no obstacles
                adaptation_obs_group=None,            # Phase 1 only
            ),
        },
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}

    return cfg


@register_task("my-task")
@dataclass(frozen=True)
class MyTask(TaskConfig):
    name: str = "my-task"

    @property
    def train_env_cfg(self):
        return my_task_env_cfg(play=False)

    @property
    def play_env_cfg(self):
        return my_task_env_cfg(play=True)

    @property
    def algo_cfg(self):
        return my_task_ppo_cfg()  # define in algo_cfg.py
```

The task is discovered automatically because `src/colosseum/tasks/__init__.py`
uses `pkgutil.iter_modules` to import all sub-packages at startup, which
triggers the `@register_task` decorator.

---

## Step 7 — Train and evaluate

```bash
# Train
pixi run train task:my-task

# Evaluate
pixi run play task:my-task --checkpoint ./logs/<run>/checkpoints/latest.pt

# Export for deployment
pixi run export-onnx task:my-task --checkpoint ./logs/<run>/checkpoints/latest.pt
```
