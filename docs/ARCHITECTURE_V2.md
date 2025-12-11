# Colosseum Architecture V2

**Key Philosophy:** Accept that training and deployment are different operations. Share the **contract** (observation spec), not the implementation.

## Core Insight

**Training** and **deployment** observe the world differently:

| Aspect | Training (mjlab) | Deployment (Real/Sim) |
|--------|------------------|----------------------|
| **Data Source** | Pre-computed `Entity.data` | Raw sensor readings |
| **Operation** | Extract from memory | Process/compute |
| **Batch Size** | 4096+ environments | Single instance |
| **Framework** | mjlab-specific | Standalone |

**Don't force them to share code** - they're fundamentally different!

## What IS Shared: The Contract

Instead of sharing observation **computation**, we share the observation **specification**:

```python
# tasks/velocity/mdp/observation_spec.py
@dataclass
class VelocityObservationSpec:
    """Contract between training and deployment."""

    ORDER = [
        "velocity_commands",   # (3,)
        "base_ang_vel",        # (3,)
        "projected_gravity",   # (3,)
        "joint_pos_rel",       # (num_joints,)
        "joint_vel",           # (num_joints,)
        "last_action",         # (num_joints,)
    ]

    @staticmethod
    def compute_size(num_joints: int) -> int:
        return 3 + 3 + 3 + num_joints * 3

    @staticmethod
    def validate_observation(obs: Tensor, num_joints: int):
        # Ensures both training and deployment produce valid observations
        ...
```

This provides:
- ✅ **Clear contract**: Both sides know exactly what's expected
- ✅ **Validation**: Runtime checks ensure consistency
- ✅ **Documentation**: Self-documenting observation structure
- ✅ **Flexibility**: Each side implements optimally for its context

## Directory Structure

```
src/colosseum/
├── mdp/                          # Layer 1: Universal (physics/math utilities)
│   ├── observations.py           # Generic quaternion rotations, etc.
│   └── rewards.py                # Generic reward kernels
│
├── robots/
│   └── booster_t1/
│       ├── xmls/                 # MuJoCo models
│       ├── t1_actuators.py       # Motor specs (training)
│       ├── t1_contacts.py        # Collision configs (training)
│       ├── t1_constants.py       # Entity configs (training)
│       └── mdp/                  # Layer 2: Robot-specific (if needed)
│           └── observations.py   # T1-specific sensors
│
└── tasks/
    └── velocity/
        ├── config/               # Training configs
        │   └── t1/env_cfgs.py
        ├── mdp/
        │   ├── observation_spec.py   # ★ CONTRACT (shared)
        │   └── observations.py       # Training MDP functions
        ├── rl/                   # RL algorithm configs
        └── deploy/               # ★ Task-specific deployment
            └── t1_23dof/         # Complete deployment bundle
                ├── robot_cfg.py      # Robot config for this task
                ├── policy.py         # Policy implementation
                ├── config.py         # Deployment config
                └── models/           # Model checkpoints
                    └── velocity_v1.pt
```

## Key Design Decisions

### 1. Task-Specific Deployment (tasks/velocity/deploy/)

**Rationale**: A trained policy is inherently coupled to:
- Specific robot configuration (joint order, PD gains)
- Specific observation structure
- Specific action scale

**Solution**: Bundle them together in `tasks/<task>/deploy/<robot>/`

**Benefits**:
- ✅ Makes coupling explicit
- ✅ Easy to version (`v1`, `v2` folders)
- ✅ Self-contained (can distribute as package)
- ✅ Training config nearby (same task folder)

### 2. Observation Spec as Contract

**Rationale**: Training and deployment compute observations differently, but need the same result.

**Solution**: Define specification, not implementation.

**Benefits**:
- ✅ No artificial "wrapper" pattern
- ✅ Clear documentation
- ✅ Runtime validation
- ✅ Debugging utilities (split, describe)

### 3. No Robot Registry

**Rationale**: Can't swap robots with trained policies.

**Solution**: Remove robot registry, keep robot configs bundled with policies.

**Benefits**:
- ✅ Prevents mismatched robot/policy combinations
- ✅ Simpler mental model
- ✅ Explicit dependencies

## Usage Examples

### Training

```python
# Training uses mjlab observation functions
from colosseum.tasks.velocity.mdp import (
    base_ang_vel,
    projected_gravity,
    joint_pos_rel,
)

cfg.observations = {
    "policy": {
        "base_ang_vel": ObservationTermCfg(func=base_ang_vel, ...),
        "projected_gravity": ObservationTermCfg(func=projected_gravity, ...),
        "joint_pos_rel": ObservationTermCfg(func=joint_pos_rel, ...),
    }
}
```

### Deployment

```python
# Deployment uses task-specific bundle
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY_DEPLOY_CFG
from colosseum.deploy.core.controllers import MujocoController

controller = MujocoController(T1_23DOF_VELOCITY_DEPLOY_CFG)
controller.run()
```

### Custom Deployment

```python
from colosseum.tasks.velocity.deploy.t1_23dof import create_deployment_cfg

cfg = create_deployment_cfg(
    checkpoint_path="models/my_model_v2.pt",
    vx_max=1.5,
    vy_max=0.8,
)

controller = MujocoController(cfg)
controller.run()
```

## Adding a New Robot

1. **Train** on robot using training configs in `tasks/velocity/config/<robot>/`

2. **Create deployment bundle** in `tasks/velocity/deploy/<robot>/`:
   ```
   tasks/velocity/deploy/your_robot/
   ├── robot_cfg.py    # Robot config (must match training!)
   ├── policy.py       # Policy (follows VelocityObservationSpec)
   ├── config.py       # Deployment config
   └── models/
       └── model.pt
   ```

3. **Done!** Import and use:
   ```python
   from colosseum.tasks.velocity.deploy.your_robot import YOUR_ROBOT_VELOCITY_DEPLOY_CFG
   ```

## Comparison with V1

| Aspect | V1 (Robot-Agnostic) | V2 (Task-Specific) |
|--------|---------------------|-------------------|
| **Philosophy** | Share observation functions | Share observation spec |
| **Deployment Location** | `deploy/tasks/velocity/` | `tasks/velocity/deploy/<robot>/` |
| **Robot Coupling** | Tried to avoid it | Embrace it |
| **Code Sharing** | Pure functions + wrappers | Specification only |
| **Complexity** | High (forced abstraction) | Low (explicit) |
| **Maintainability** | Harder (hidden coupling) | Easier (explicit coupling) |

## Benefits of V2

1. **Simpler Mental Model**: Training and deployment are separate, connected by contract
2. **No Forced Abstraction**: Each side implements optimally for its context
3. **Explicit Coupling**: Robot + policy bundled together
4. **Easy Versioning**: Whole folder is one deployable unit
5. **Better Organization**: Deployment lives with its training counterpart

## Summary

V2 recognizes that **training and deployment are fundamentally different operations** and shouldn't be forced to share implementation. Instead:

- **Share the contract** (`VelocityObservationSpec`)
- **Bundle robot + policy** (tasks/velocity/deploy/t1_23dof/)
- **Validate at runtime** (`VELOCITY_OBS_SPEC.validate_observation()`)
- **Keep it simple** (no artificial abstractions)

This approach is **more honest** about the reality of RL deployment and results in **cleaner, more maintainable code**.
