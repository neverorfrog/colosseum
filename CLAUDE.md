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

**IMPORTANT: Always use `pixi run python` for Python commands:**
```bash
# Correct
pixi run python script.py

# Incorrect
python script.py  # Will fail with ModuleNotFoundError
```

This ensures the correct environment with all dependencies (mjlab, booster_robotics_sdk, etc.) is active.

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

## Working with Robots

### Booster T1 Humanoid

Located in `src/colosseum/robots/booster_t1/`:

#### XML Models

- **23 DOF model** ([t1.xml](src/colosseum/robots/t1/xmls/t1.xml)): Full body including arms, waist, and neck, used for all tasks

#### Configuration Modules

The T1 configuration is organized into three modules:

**[actuators.py](src/colosseum/robots/t1/actuators.py)**: Motor specifications and actuator configurations
- `MOTOR_SPECS`: Dictionary of motor specifications from manufacturer data (gear ratio, torque, speed, inertia)
- Actuator configs for 12-DOF locomotion:
  - `T1_ACTUATOR_HIP_PITCH`, `T1_ACTUATOR_HIP_ROLL`, `T1_ACTUATOR_HIP_YAW`
  - `T1_ACTUATOR_KNEE`
  - `T1_ACTUATOR_ANKLE_PITCH`, `T1_ACTUATOR_ANKLE_ROLL`
- Actuator configs for 23-DOF full body:
  - `T1_ACTUATOR_NECK`, `T1_ACTUATOR_ARM`, `T1_ACTUATOR_WAIST`

**[contacts.py](src/colosseum/robots/t1/contacts.py)**: Collision and contact sensor configurations
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

**[constants.py](src/colosseum/robots/t1/constants.py)**: Spec loaders and entity configurations
- XML paths: `T1_12DOF_XML`, `t1_XML`
- Spec loaders:
  - `get_t1_12dof_spec()`: Returns MjSpec for 12-DOF locomotion model
  - `get_t1_spec()`: Returns MjSpec for 23-DOF full body model
- Pre-configured entity configs:
  - `T1_12DOF_ENTITY_CFG`: Complete entity config for locomotion training
  - `t1_ENTITY_CFG`: Complete entity config for full body deployment

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


## Guidelines for Claude Code

When working with code in this repository, please keep the following guidelines in mind:

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
