# Colosseum Architecture Overview

This document provides a high-level overview of Colosseum's architecture, focusing on the novel three-layer MDP system and robot-agnostic deployment.

## Table of Contents

1. [Three-Layer MDP Architecture](#three-layer-mdp-architecture)
2. [Robot-Agnostic Deployment](#robot-agnostic-deployment)
3. [Directory Structure](#directory-structure)
4. [Comparison with Other Frameworks](#comparison-with-other-frameworks)

---

## Three-Layer MDP Architecture

Colosseum implements a unique three-layer architecture for MDP functions (observations, rewards, terminations, etc.) that **eliminates code duplication** between training and deployment.

### The Problem

Traditional RL deployment requires duplicating observation/reward logic:

```python
# Training (mjlab/Isaac Gym)
def base_ang_vel(env, asset_cfg):
    robot = env.scene[asset_cfg.name]
    return robot.data.root_ang_vel_b

# Deployment (DUPLICATED!)
def compute_observation(self):
    base_ang_vel = self.robot.data.root_ang_vel_b  # Same logic!
    # ... more duplicated logic
```

**Problems:**
- ❌ Maintenance burden (change once → update twice)
- ❌ Error-prone (easy to forget deployment)
- ❌ No single source of truth
- ❌ Difficult to test consistency

### The Solution: Three Layers

```
┌─────────────────────────────────────────────────┐
│  Layer 1: Universal MDP (colosseum.mdp)         │
│  • Robot-agnostic physics/math functions        │
│  • compute_projected_gravity()                  │
│  • exponential_reward_kernel()                  │
└──────────────┬──────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────┐
│  Layer 2: Robot-Specific (robots/*/mdp)         │
│  • Platform-specific sensor processing          │
│  • compute_foot_contact_state() [T1]            │
└──────────────┬──────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────┐
│  Layer 3: Task-Specific (tasks/*/mdp)           │
│  ├── observations.py (SHARED pure functions)    │
│  │   • compute_base_ang_vel()                   │
│  │   • compute_joint_pos_rel()                  │
│  └── wrappers.py (TRAINING-ONLY adapters)       │
│      • base_ang_vel(env, asset_cfg) → wraps     │
└──────────────┬──────────────────┬───────────────┘
               ↓                  ↓
      ┌────────────────┐   ┌─────────────────┐
      │  Training      │   │  Deployment     │
      │  (Wrappers)    │   │  (Pure Funcs)   │
      └────────────────┘   └─────────────────┘
```

### Layer Details

#### Layer 1: Universal (`src/colosseum/mdp/`)

**Scope**: Robot-agnostic, physics-based functions

**Examples:**
- `compute_projected_gravity(quat)` - quaternion rotation
- `exponential_reward_kernel(error, std)` - reward shaping
- Generic kinematics, tracking errors

**Used by:** All robots and tasks, both training and deployment

**Dependencies:** Only torch and math utilities

#### Layer 2: Robot-Specific (`src/colosseum/robots/*/mdp/`)

**Scope**: Platform-specific functions

**Examples:**
- `compute_foot_contact_state(forces)` - T1 contact detection
- Robot anatomy constants (foot geom names, body indices)

**Used by:** Multiple tasks using the same robot

**Dependencies:** Layer 1 functions, robot constants

#### Layer 3: Task-Specific (`src/colosseum/tasks/*/mdp/`)

**Scope**: Task-specific functions, split into two modules:

**`observations.py`** - Pure functions (SHARED):
```python
def compute_base_ang_vel(robot_data) -> Tensor:
    """Works in training AND deployment."""
    return robot_data.root_ang_vel_b

def compute_joint_pos_rel(robot_data, default_pos, joint_map=None) -> Tensor:
    """Pure function with optional joint mapping for deployment."""
    joint_pos = robot_data.joint_pos
    if joint_map is not None:  # Deployment only
        joint_pos = joint_pos[joint_map]
    return joint_pos - default_pos
```

**`wrappers.py`** - Training adapters (TRAINING-ONLY):
```python
from colosseum.tasks.velocity.mdp.observations import compute_base_ang_vel

def base_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Tensor:
    """Wrapper for mjlab training environment."""
    robot = env.scene[asset_cfg.name]
    return compute_base_ang_vel(robot.data)  # Calls pure function
```

### Usage Examples

#### Training Configuration

```python
# Import wrappers for training
from colosseum.tasks.velocity.mdp import wrappers

cfg.observations = {
    "policy": {
        "base_ang_vel": ObservationTermCfg(
            func=wrappers.base_ang_vel,  # Use wrapper
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    }
}
```

#### Deployment Policy

```python
# Import pure functions for deployment
from colosseum.tasks.velocity.mdp.observations import (
    compute_base_ang_vel,
    concatenate_observations,
)

class VelocityPolicy(Policy):
    def compute_observation(self):
        obs_dict = {
            "base_ang_vel": compute_base_ang_vel(self.robot.data),  # Pure function
            # ... more observations
        }
        return concatenate_observations(obs_dict)
```

### Benefits

✅ **Single Source of Truth**: Logic defined once in pure functions
✅ **Guaranteed Consistency**: Training and deployment use identical code
✅ **Easy Maintenance**: Change once → works everywhere
✅ **Testable**: Pure functions are independently unit-testable
✅ **Clear Separation**: Computation (pure) vs. integration (wrappers)

---

## Robot-Agnostic Deployment

Colosseum's deployment system decouples policies from specific robot platforms using a registry pattern.

### Robot Registry

Robots are configured and registered independently:

```python
# src/colosseum/robots/booster_t1/deploy/robot_cfg.py
T1_23DOF_DEPLOY_CFG = RobotCfg(
    name="Booster_T1_23DOF",
    joint_names=[...],         # Real robot order
    sim_joint_names=[...],     # Simulation order (alphabetical)
    joint_stiffness=[...],
    joint_damping=[...],
    default_joint_pos=[...],
    effort_limit=[...],
    mjcf_path="path/to/T1_23dof.xml",
)
```

### Task Configuration

Tasks combine robots with policies:

```python
# src/colosseum/deploy/tasks/velocity/configs.py
from colosseum.robots.booster_t1.deploy import T1_23DOF_DEPLOY_CFG
from colosseum.deploy.tasks.velocity import VelocityPolicyCfg

cfg = ControllerCfg(
    robot=T1_23DOF_DEPLOY_CFG,  # Plug in any robot!
    policy=VelocityPolicyCfg(checkpoint_path="models/velocity.pt"),
)

register_task("t1_23dof_velocity", cfg)
```

### Automatic Joint Mapping

The system automatically handles joint order differences:

**Training**: Alphabetical order (simulation convention)
**Deployment**: Hardware-specific order

```python
# Automatically computed in RobotData
real2sim_indexes = [joint_names.index(name) for name in sim_joint_names]
sim2real_indexes = [sim_joint_names.index(name) for name in joint_names]

# Observations → simulation order (for model input)
obs = compute_joint_pos(data, joint_map=robot.data.real2sim_indexes)

# Actions → real order (for hardware output)
targets = action[robot.data.sim2real_indexes] * scale + default_pos
```

### Adding a New Robot

1. Create `RobotCfg` in `robots/your_robot/deploy/robot_cfg.py`
2. Create task config combining robot + policy
3. Register task: `register_task("your_robot_velocity", cfg)`
4. Done! Policy automatically adapts.

---

## Directory Structure

```
src/colosseum/
├── mdp/                      # Layer 1: Universal
│   ├── observations.py
│   └── rewards.py
│
├── robots/
│   └── booster_t1/
│       ├── xmls/             # MuJoCo models
│       ├── t1_actuators.py   # Motor specs (training)
│       ├── t1_contacts.py    # Collision configs (training)
│       ├── t1_constants.py   # Entity configs (training)
│       ├── mdp/              # Layer 2: Robot-specific
│       │   └── observations.py
│       └── deploy/           # Deployment configs
│           └── robot_cfg.py
│
├── tasks/
│   └── velocity/
│       ├── config/           # Training configs
│       │   └── t1/env_cfgs.py
│       ├── mdp/              # Layer 3: Task-specific
│       │   ├── observations.py  # SHARED pure functions
│       │   └── wrappers.py      # Training-only adapters
│       └── rl/               # RL algorithm configs
│
└── deploy/
    ├── core/
    │   ├── controllers/      # Base abstractions
    │   └── utils/
    │       ├── registry.py         # Task registry
    │       └── robot_registry.py   # Robot registry
    └── tasks/
        └── velocity/
            ├── policy.py     # Robot-agnostic policy
            └── configs.py    # Example configs
```

---

## Comparison with Other Frameworks

### Legged Gym / Humanoid-Gym

**Their Approach:**
- Separate training and deployment codebases
- Observation logic duplicated between sim and real
- Manual alignment required

**Colosseum Advantages:**
- ✅ Shared pure functions (no duplication)
- ✅ Guaranteed consistency
- ✅ Robot-agnostic deployment

### Isaac Lab

**Their Approach:**
- Manager-based training (similar to Colosseum)
- Deployment left to user
- No shared MDP functions

**Colosseum Advantages:**
- ✅ Three-layer MDP architecture
- ✅ Built-in deployment system
- ✅ Automatic joint mapping

### rl_sar

**Their Approach:**
- Backend abstraction (IsaacGym, MuJoCo, Real)
- Still requires manual observation alignment

**Colosseum Advantages:**
- ✅ Pure functions (cleaner than backend abstraction)
- ✅ Explicit train/deploy separation (wrappers vs pure functions)
- ✅ Layered responsibility (universal → robot → task)

---

## Documentation References

- **Detailed Implementation Guide**: [docs/SHARED_OBSERVATIONS.md](SHARED_OBSERVATIONS.md)
- **Velocity Task MDP**: [tasks/velocity/mdp/README.md](../src/colosseum/tasks/velocity/mdp/README.md)
- **Deployment Guide**: [deploy/tasks/velocity/README.md](../src/colosseum/deploy/tasks/velocity/README.md)
- **Developer Guide**: [CLAUDE.md](../CLAUDE.md)

---

## Summary

Colosseum's architecture provides:

1. **Three-Layer MDP System**: Eliminates training/deployment duplication
2. **Robot-Agnostic Deployment**: Policies work with any robot configuration
3. **Automatic Joint Mapping**: Handles sim↔real joint order differences
4. **Clear Separation**: Pure functions (shared) vs. wrappers (training-only)

This architecture is **novel** in the open-source RL robotics space and provides significant advantages over existing frameworks.
