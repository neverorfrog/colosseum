# Migration Guide: V1 → V2

This guide helps you migrate from the original V1 architecture to the refined V2 architecture.

## What Changed?

### Core Philosophy

**V1**: Tried to share observation computation between training and deployment
- Used "pure functions + wrappers" pattern
- Robot-agnostic deployment with robot registry
- Forced abstraction between fundamentally different operations

**V2**: Share the observation **contract**, not the implementation
- Training and deployment compute observations differently (and that's OK!)
- Task-specific deployment bundles (robot + policy + config together)
- Explicit coupling, simpler code

## Directory Structure Changes

### V1 Structure (OLD)
```
src/colosseum/
├── mdp/                     # Universal functions
├── robots/booster_t1/
│   ├── mdp/                 # Robot-specific functions
│   └── deploy/
│       └── robot_cfg.py     # Robot config (decoupled)
├── tasks/velocity/
│   └── mdp/
│       ├── observations.py  # Pure functions (shared)
│       └── wrappers.py      # Training wrappers
└── deploy/
    ├── core/utils/
    │   └── robot_registry.py  # Robot registry
    └── tasks/velocity/
        ├── policy.py        # Robot-agnostic policy
        └── configs.py       # Configs using registry
```

### V2 Structure (NEW)
```
src/colosseum/
├── deploy/
│   └── core/
│       ├── controllers/     # Base controllers
│       ├── observation_spec.py  # Abstract base class
│       └── utils/
│           └── registry.py  # Task registry (optional)
├── robots/booster_t1/
│   ├── xmls/
│   ├── t1_actuators.py      # Training configs
│   ├── t1_contacts.py
│   └── t1_constants.py
└── tasks/velocity/
    ├── config/t1/           # Training configs
    ├── mdp/
    │   ├── observation_spec.py  # CONTRACT (spec, not implementation)
    │   └── observations.py      # Training MDP functions (simple)
    └── deploy/              # ★ Task-specific deployment
        └── t1_23dof/        # Complete bundle
            ├── robot_cfg.py     # Robot config for this task
            ├── policy.py        # Policy implementation
            ├── config.py        # Deployment config
            └── models/
                └── velocity_v1.pt
```

## Code Changes

### 1. Observation Specification

#### V1 (OLD) - Shared Functions
```python
# tasks/velocity/mdp/observations.py (V1)
def compute_base_ang_vel(robot_data) -> Tensor:
    """Pure function used in training AND deployment."""
    return robot_data.root_ang_vel_b

# tasks/velocity/mdp/wrappers.py (V1)
def base_ang_vel(env, asset_cfg) -> Tensor:
    """Training wrapper."""
    robot = env.scene[asset_cfg.name]
    return compute_base_ang_vel(robot.data)  # Calls pure function

# deploy/tasks/velocity/policy.py (V1)
obs = compute_base_ang_vel(self.robot.data)  # Uses pure function
```

#### V2 (NEW) - Shared Contract
```python
# tasks/velocity/mdp/observation_spec.py (V2)
class VelocityObservationSpec(ObservationSpec):
    """Contract defining observation structure."""

    @property
    def observation_names(self) -> List[str]:
        return ["velocity_commands", "base_ang_vel", ...]

    def get_component_size(self, name: str, num_joints: int) -> int:
        if name == "base_ang_vel":
            return 3
        # ...

# tasks/velocity/mdp/observations.py (V2)
def base_ang_vel(env, asset_cfg) -> Tensor:
    """Training function - extracts from entity."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b  # Pre-computed by mjlab

# tasks/velocity/deploy/t1_23dof/policy.py (V2)
def compute_observation(self):
    """Deployment - computes from sensors."""
    obs = torch.cat([
        vel_cmd,
        self.robot.data.root_ang_vel_b,  # From sensors
        # ...
    ])
    VELOCITY_OBS_SPEC.validate_observation(obs, self.robot.num_joints)
    return obs
```

### 2. Deployment Configuration

#### V1 (OLD) - Robot Registry
```python
# robots/booster_t1/deploy/robot_cfg.py (V1)
T1_23DOF_DEPLOY_CFG = RobotCfg(...)

# deploy/tasks/velocity/configs.py (V1)
from colosseum.robots.booster_t1.deploy import T1_23DOF_DEPLOY_CFG
from colosseum.deploy.tasks.velocity import VelocityPolicyCfg

cfg = ControllerCfg(
    robot=T1_23DOF_DEPLOY_CFG,  # Decoupled
    policy=VelocityPolicyCfg(...),
)
register_task("t1_23dof_velocity", cfg)

# Usage (V1)
from colosseum.deploy import get_task
cfg = get_task("t1_23dof_velocity")
```

#### V2 (NEW) - Task Bundles
```python
# tasks/velocity/deploy/t1_23dof/robot_cfg.py (V2)
T1_23DOF_VELOCITY_ROBOT_CFG = RobotCfg(...)  # Bundled with task

# tasks/velocity/deploy/t1_23dof/config.py (V2)
from .robot_cfg import T1_23DOF_VELOCITY_ROBOT_CFG
from .policy import T1VelocityPolicy

def create_deployment_cfg(checkpoint_path: str) -> ControllerCfg:
    return ControllerCfg(
        robot=T1_23DOF_VELOCITY_ROBOT_CFG,  # Bundled together!
        policy=...,  # Uses T1VelocityPolicy
    )

T1_23DOF_VELOCITY_DEPLOY_CFG = create_deployment_cfg()

# Usage (V2)
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY_DEPLOY_CFG
controller = MujocoController(T1_23DOF_VELOCITY_DEPLOY_CFG)
```

## Migration Steps

### For Training Code

**No changes needed!** Training MDP functions are simpler in V2:

```python
# Before (V1)
from colosseum.tasks.velocity.mdp import wrappers
cfg.observations = {
    "policy": {
        "base_ang_vel": ObservationTermCfg(func=wrappers.base_ang_vel, ...),
    }
}

# After (V2) - even simpler!
from colosseum.tasks.velocity.mdp import base_ang_vel
cfg.observations = {
    "policy": {
        "base_ang_vel": ObservationTermCfg(func=base_ang_vel, ...),
    }
}
```

### For Deployment Code

**Change import location:**

```python
# Before (V1)
from colosseum.deploy import get_task
cfg = get_task("t1_23dof_velocity")

# After (V2)
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY_DEPLOY_CFG
# Use T1_23DOF_VELOCITY_DEPLOY_CFG directly
```

### Adding New Robot Deployment

#### V1 (OLD)
1. Create `robots/<robot>/deploy/robot_cfg.py`
2. Create `deploy/tasks/<task>/configs.py` using robot registry
3. Register task

#### V2 (NEW)
1. Create `tasks/<task>/deploy/<robot>/` directory
2. Add files:
   - `robot_cfg.py` - Robot config
   - `policy.py` - Policy (follows ObservationSpec)
   - `config.py` - Deployment config
   - `models/` - Model checkpoints
3. Done! Import directly from bundle

## Key Improvements in V2

### 1. Simpler Mental Model
- ✅ Training and deployment are different → accept it
- ✅ Share contract, not implementation
- ✅ No forced abstractions

### 2. Explicit Coupling
- ✅ Robot + policy bundled together
- ✅ Can't accidentally use wrong robot with policy
- ✅ Easy to version (whole folder)

### 3. Better Organization
- ✅ Deployment lives with training (same task folder)
- ✅ Self-contained bundles
- ✅ Clear dependencies

### 4. ObservationSpec Base Class
- ✅ Abstract base class for any task
- ✅ Can derive from training `ObservationGroupCfg`
- ✅ Runtime validation
- ✅ Debugging utilities

## Validation

### Validate Training Config Against Spec

```python
from colosseum.tasks.velocity.config.t1.env_cfgs import booster_t1_flat_env_cfg
from colosseum.tasks.velocity.mdp.observation_spec import VelocityObservationSpec

# Load training config
training_cfg = booster_t1_flat_env_cfg()

# Validate structure
spec = VelocityObservationSpec.from_observation_group_cfg(
    training_cfg.observations["policy"],
    num_joints=23
)
# ✓ Training config validated against VelocityObservationSpec
# Total observation size: 78
# Components: velocity_commands, base_ang_vel, ...
```

### Validate Deployment Observations

```python
# In policy.py
obs = self.compute_observation()
VELOCITY_OBS_SPEC.validate_observation(obs, self.robot.num_joints)
# Raises ValueError if size/structure doesn't match
```

## Removed Components

### Deleted Files (V1 → V2)
- ❌ `deploy/core/utils/robot_registry.py` (not needed)
- ❌ `deploy/tasks/velocity/` (moved to `tasks/velocity/deploy/`)
- ❌ `robots/booster_t1/deploy/` (moved to task-specific folders)
- ❌ `tasks/velocity/mdp/wrappers.py` (not needed)

### Simplified Files
- ✅ `tasks/velocity/mdp/observations.py` - Simpler (just extract from entity)
- ✅ `deploy/__init__.py` - No robot registry imports

## FAQ

**Q: Can I still use the robot registry pattern?**
A: The registry utils still exist if you need them, but V2 encourages direct imports from task bundles.

**Q: How do I share a robot config across multiple tasks?**
A: Each task has its own robot config in `tasks/<task>/deploy/<robot>/robot_cfg.py`. This explicit duplication makes dependencies clear and allows per-task customization.

**Q: What if my robot config is exactly the same for multiple tasks?**
A: You can create a shared config in `robots/<robot>/deploy_base_cfg.py` and import it:
```python
# tasks/velocity/deploy/t1_23dof/robot_cfg.py
from colosseum.robots.booster_t1.deploy_base_cfg import BASE_T1_CFG

T1_23DOF_VELOCITY_ROBOT_CFG = BASE_T1_CFG  # Or customize if needed
```

**Q: Do I need to change my trained models?**
A: No! Models are unchanged. V2 just changes how deployment is organized.

## Summary

V2 is a **simplification** that accepts the reality of RL deployment:
- Training and deployment are fundamentally different operations
- Share the **contract** (observation spec), not the implementation
- Bundle robot + policy together (explicit coupling)
- Keep it simple (no forced abstractions)

The result is **cleaner, more maintainable code** that's easier to understand and extend.
