# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Colosseum** is a learning playground for humanoid robot research focused on robot soccer. It builds on top of **mjlab**, a GPU-accelerated reinforcement learning framework for robotics built on MuJoCo Warp.

Key characteristics:
- GPU-accelerated parallel training (4096+ environments)
- Manager-based declarative configuration (rewards, observations, actions)
- MuJoCo-native physics simulation
- Primary robot: Booster T1 humanoid (12 DOF locomotion / 23 DOF full body)

## Development Setup

### Environment Management

This project uses **Pixi** (a conda/pypi workspace manager) for environment and dependency management:

```bash
# Install dependencies and activate environment
pixi install

# Serve documentation locally
pixi run docs

# Build documentation
pixi run build-docs
```

The project uses:
- Python 3.12 (strict version)
- NVIDIA GPU required for training (MuJoCo Warp)
- `MUJOCO_GL=osmesa` environment variable (set automatically by Pixi)

### Code Quality

Ruff is configured with:
- Source directory: `src`
- Indent width: 2 spaces

## Architecture

### Three-Layer Structure

Colosseum follows mjlab's three-layer architecture:

1. **Simulation Layer** (lowest): GPU-accelerated physics via MuJoCo Warp
   - Batched parallel execution across thousands of environments
   - Direct `mjModel`/`mjData` access

2. **Scene Layer** (middle): Physical object organization
   - Entities (robots, terrain, obstacles)
   - Name → index mapping and resolution
   - Entity data access (poses, velocities, forces)

3. **Task Layer** (highest): Reinforcement learning MDP definition
   - Manager-based orchestration (Actions, Observations, Rewards, Terminations, Events)
   - Declarative term configuration
   - `gym.Env` interface for RL training

### Term-Based Pattern

The core design pattern across all layers:

1. **Declaration** (config time): Define terms declaratively using `*TermCfg` classes
2. **Resolution** (initialization): Map entity/component names → MuJoCo indices once
3. **Execution** (runtime): Fast computation using pre-resolved indices

Example term configuration:
```python
rewards = {
    "stand_reward": RewardTermCfg(
        func=mdp.body_height_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso")}
    )
}
```

During initialization, `SceneEntityCfg.resolve()` converts `body_names="torso"` → `body_ids=[3]` (global MuJoCo index). At runtime, term functions use these pre-resolved indices for efficient tensor operations.

## Directory Structure

```
src/colosseum/
├── robots/           # Robot definitions and constants
│   ├── booster_t1/   # Booster T1 humanoid
│   │   ├── xmls/     # MuJoCo XML models
│   │   │   ├── T1_12dof.xml   # 12-DOF locomotion model
│   │   │   └── T1_23dof.xml   # 23-DOF full body model
│   │   ├── t1_actuators.py    # Motor specs and actuator configs
│   │   ├── t1_contacts.py     # Collision and contact sensor configs
│   │   └── t1_constants.py    # Spec loaders and entity configs
│   └── cartpole/     # CartPole balancing demo
├── train/            # Training tasks
│   └── tasks/        # Task implementations
│       ├── cartpole/ # CartPole balancing task
│       │   ├── cartpole_scene.py  # Scene configuration
│       │   ├── cartpole_task.py   # MDP (actions, obs, rewards, terminations, events)
│       │   └── mdp_functions.py   # Custom term functions
│       └── velocity/ # Velocity tracking task (T1)
│           ├── config/  # Robot-specific configs
│           ├── mdp/     # MDP term functions
│           └── rl/      # RL algorithm configs
├── deploy/           # Deployment configurations
│   ├── configs/      # Deployment configs
│   └── envs/         # Deployment environments
├── play/             # Evaluation/playback utilities
└── utils/            # Shared utilities (path helpers)

external/             # Git submodules (not managed by colosseum)
├── mjlab/            # Core RL framework
└── holosoma/         # Motion retargeting (optional)

tests/                # Test files
docs/                 # MkDocs documentation
```

## Creating Tasks

Tasks are registered via the `mjlab.tasks` entry point in `pyproject.toml`:

```toml
[project.entry-points."mjlab.tasks"]
cartpole = "colosseum.tasks.cartpole"
```

Each task module must:
1. Define scene configuration (`SceneCfg`)
2. Define MDP components (actions, observations, rewards, terminations, events)
3. Create `ManagerBasedRlEnvCfg` combining all components
4. Call `register_mjlab_task()` with a unique `task_id`

See `src/colosseum/train/tasks/cartpole/` for a complete minimal example.

## Working with Robots

### Booster T1 Humanoid

Located in `src/colosseum/robots/booster_t1/`:

#### XML Models

- **12 DOF model** ([T1_12dof.xml](src/colosseum/robots/booster_t1/xmls/T1_12dof.xml)): Legs only (6 DOF per leg), used for locomotion training
- **23 DOF model** ([T1_23dof.xml](src/colosseum/robots/booster_t1/xmls/T1_23dof.xml)): Full body including arms, waist, and neck, for deployment

#### Configuration Modules

The T1 configuration is organized into three modules:

**[t1_actuators.py](src/colosseum/robots/booster_t1/t1_actuators.py)**: Motor specifications and actuator configurations
- `MOTOR_SPECS`: Dictionary of motor specifications from manufacturer data (gear ratio, torque, speed, inertia)
- `compute_pd_gains()`: Computes PD controller gains using Unitree G1 method (natural frequency + damping ratio)
- Actuator configs for 12-DOF locomotion:
  - `T1_ACTUATOR_HIP_PITCH`, `T1_ACTUATOR_HIP_ROLL`, `T1_ACTUATOR_HIP_YAW`
  - `T1_ACTUATOR_KNEE`
  - `T1_ACTUATOR_ANKLE_PITCH`, `T1_ACTUATOR_ANKLE_ROLL`
- Actuator configs for 23-DOF full body:
  - `T1_ACTUATOR_NECK`, `T1_ACTUATOR_ARM`, `T1_ACTUATOR_WAIST`

**[t1_contacts.py](src/colosseum/robots/booster_t1/t1_contacts.py)**: Collision and contact sensor configurations
- Collision configs (modify geom properties):
  - `FEET_ONLY_COLLISION`: Only foot geoms collide (recommended for training)
  - `FULL_COLLISION_WITHOUT_SELF`: All parts collide with environment, no self-collision
  - `FULL_COLLISION`: Full collision including self-collision (most realistic)
  - `HANDS_FEET_COLLISION`: Only hands and feet collide (for manipulation)
- Contact sensor configs (for observation/reward):
  - `FEET_GROUND_CONTACT_SENSOR`: Tracks foot-ground contact with air time
  - `SELF_COLLISION_SENSOR`: Detects self-collisions
  - `HAND_CONTACT_SENSOR`: Tracks hand contact for manipulation
- `T1_FOOT_GEOM_NAMES`: Tuple of all foot geometry names for events

**[t1_constants.py](src/colosseum/robots/booster_t1/t1_constants.py)**: Spec loaders and entity configurations
- XML paths: `T1_12DOF_XML`, `T1_23DOF_XML`
- Spec loaders:
  - `get_t1_12dof_spec()`: Returns MjSpec for 12-DOF locomotion model
  - `get_t1_23dof_spec()`: Returns MjSpec for 23-DOF full body model
- Pre-configured entity configs:
  - `T1_12DOF_ENTITY_CFG`: Complete entity config for locomotion training
  - `T1_23DOF_ENTITY_CFG`: Complete entity config for full body deployment

#### PD Gain Computation

T1 uses the Unitree G1 method for computing actuator gains:

```python
# Natural frequency and damping ratio (same as G1)
natural_freq = 10.0 * 2π  # 10Hz in rad/s
damping_ratio = 2.0  # Overdamped (prevents oscillations)

# Compute gains from motor reflected inertia
stiffness = reflected_inertia × ω_n²
damping = 2 × ζ × reflected_inertia × ω_n
```

This ensures stable, overdamped control appropriate for each joint's mechanical properties.

### Entity Configuration

#### Using Pre-Configured Entities

The simplest way to use T1 is with the pre-configured entity configs:

```python
from colosseum.robots.booster_t1.t1_constants import T1_12DOF_ENTITY_CFG

# Use directly in scene config
entities = {"robot": T1_12DOF_ENTITY_CFG}
```

These configs include all actuators, collision settings, and sensors with manufacturer-accurate specifications.

#### Custom Entity Configuration

Entities are configured using `EntityCfg`:
- `spec_fn`: Function returning `mujoco.MjSpec` (e.g., `get_t1_12dof_spec`)
- `init_state`: Initial joint positions/velocities
- `actuators`: Dict of actuator configs applying to joint patterns
- `articulation_info`: Optional collision/armature settings using `EntityArticulationInfoCfg`:
  - `collision`: `CollisionCfg` object (e.g., `FEET_ONLY_COLLISION`)
  - `armature`: Joint armature values (reflected inertia)
- `sensors`: List of sensor configs (e.g., `[FEET_GROUND_CONTACT_SENSOR]`)

## Documentation

Documentation is built with MkDocs Material and includes:
- Colosseum-specific guides (Getting Started, CartPole tutorial)
- mjlab architecture documentation (3-layer system, term-based patterns)

The documentation covers:
- **Overview**: Architecture and design patterns
- **Task Layer**: Manager-based RL environment, term configuration
- **Scene Layer**: Entities, compilation, indexing
- **Simulation Layer**: MuJoCo Warp integration

## Key Concepts

### SceneEntityCfg Resolution

`SceneEntityCfg` is the bridge between managers and entities:
- Stores entity name + component patterns (e.g., `body_names=".*_knee"`)
- During `resolve()`, converts patterns to global MuJoCo indices
- Passed to term functions at runtime with pre-resolved indices

### MuJoCo Compilation Flow

1. Create `Scene` with entity configs
2. Each entity loads its `MjSpec` (XML representation)
3. Scene merges all specs and compiles → `MjModel` (MuJoCo assigns global indices)
4. Initialize entities with compiled model → creates `EntityIndexing`
5. Resolve `SceneEntityCfg` objects to map names → indices

### Custom Term Functions

Term functions follow this signature:
```python
def my_reward(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    data = entity.data.body_pos_w[:, asset_cfg.body_ids]  # Pre-resolved indices
    return compute_reward(data)
```

## Common Patterns

### Path Resolution

Use utilities from `colosseum.utils`:
- `project_root()`: Returns repository root (looks for pyproject.toml)
- `src_dir()`: Returns `src/colosseum/` directory

### Robot XML Paths

Always use Path objects and verify existence:
```python
from colosseum.utils import src_dir
xml_path = src_dir() / "robots" / "my_robot" / "xmls" / "model.xml"
assert xml_path.exists(), f"XML not found: {xml_path}"
```

### Actuator Configuration

Group actuators by joint type with regex patterns:
```python
from colosseum.robots.booster_t1.t1_actuators import (
    T1_ACTUATOR_HIP_PITCH,
    T1_ACTUATOR_KNEE,
)

actuators = {
    "hip_pitch": T1_ACTUATOR_HIP_PITCH,  # Pre-configured with motor specs
    "knee": T1_ACTUATOR_KNEE,
}
```

Or create custom configs:
```python
from mjlab.actuator import BuiltinPositionActuatorCfg

actuators = {
    "hip_pitch": BuiltinPositionActuatorCfg(
        joint_names_expr=(".*Hip_Pitch",),  # Matches all hip pitch joints
        stiffness=50.0,
        damping=3.0,
        effort_limit=45.0,
    ),
}
```

### Collision and Contact Configuration

Use pre-configured collision settings for different scenarios:
```python
from colosseum.robots.booster_t1.t1_contacts import (
    FEET_ONLY_COLLISION,
    FEET_GROUND_CONTACT_SENSOR,
)
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

entity_cfg = EntityCfg(
    spec_fn=get_t1_12dof_spec,
    actuators={...},
    articulation_info=EntityArticulationInfoCfg(
        collision=FEET_ONLY_COLLISION,  # Only feet collide (stable training)
    ),
    sensors=[FEET_GROUND_CONTACT_SENSOR],  # Track foot-ground contact
)
```

Available collision presets:
- `FEET_ONLY_COLLISION`: Only feet collide (recommended for locomotion training)
- `FULL_COLLISION_WITHOUT_SELF`: All parts collide with environment
- `FULL_COLLISION`: Full collision including self-collision (most realistic)
- `HANDS_FEET_COLLISION`: Only hands and feet collide (for manipulation)

Available contact sensors:
- `FEET_GROUND_CONTACT_SENSOR`: Tracks foot-ground contact with air time
- `SELF_COLLISION_SENSOR`: Detects self-collisions
- `HAND_CONTACT_SENSOR`: Tracks hand contact for manipulation

## External Dependencies

- **mjlab** (Git submodule): Core RL framework, installed via `pyproject.toml`
- **holosoma** (Git submodule): Motion retargeting (optional, not used in core training)

These are managed in `external/` and installed as editable packages during `pixi install`.
