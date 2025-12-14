# New Deployment Configuration System - Usage Guide

**Status**: Phase 4-5 Complete (Presets + CLI)
**Date**: 2025-12-13

## Overview

The new deployment configuration system replaces the old `@configclass` + argparse approach with:
- ✅ **Pydantic dataclasses** (type-safe, validated, immutable)
- ✅ **Policy registry** (decorator-based, decoupled from configs)
- ✅ **Tyro CLI** (auto-generated help, deep overrides)
- ✅ **Centralized presets** (DRY, override capability)

---

## Architecture

```
deploy/config/
├── types/                    # Pydantic type definitions
│   ├── robot.py              # RobotConfig, PrepareStateConfig
│   ├── policy.py             # PolicyConfig, VelocityCommandConfig
│   ├── backend.py            # MujocoConfig, BoosterConfig
│   └── controller.py         # ControllerConfig (top-level)
│
├── values/                   # Preset configurations
│   ├── robots/               # Robot hardware specs (centralized)
│   │   └── t1.py             # T1_23DOF_ROBOT_CFG
│   └── deployments/          # Complete deployment presets
│       └── t1_velocity.py    # T1_23DOF_VELOCITY
│
└── registry.py               # Policy registry (POLICY_REGISTRY)

tasks/velocity/deploy/t1_23dof/
├── policy.py                 # T1VelocityPolicy + @register_policy("velocity")
└── __init__.py               # Imports to trigger registration
```

---

## Quick Start

### 1. **Using Presets (Recommended)**

```bash
# List available presets
pixi run python src/colosseum/deploy/scripts/deploy_tyro.py --help

# Run with default preset (MuJoCo simulation)
pixi run python src/colosseum/deploy/scripts/deploy_tyro.py deploy:t1-velocity

# Deploy on real robot
pixi run python src/colosseum/deploy/scripts/deploy_tyro.py deploy:t1-velocity --target real
```

### 2. **Override Preset Parameters**

```bash
# Change checkpoint path
pixi run python src/colosseum/deploy/scripts/deploy_tyro.py deploy:t1-velocity \
    --config.policy.checkpoint-path models/my_model.pt

# Override velocity limits
pixi run python src/colosseum/deploy/scripts/deploy_tyro.py deploy:t1-velocity \
    --config.vel-command.vx-max 1.5 \
    --config.vel-command.vyaw-max 0.8

# Change action scale
pixi run python src/colosseum/deploy/scripts/deploy_tyro.py deploy:t1-velocity \
    --config.policy.action-scale 0.3
```

### 3. **Programmatic Usage (Python)**

```python
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY
from colosseum.deploy.core.backends.mujoco import MujocoController

# Use default preset
with MujocoController(T1_23DOF_VELOCITY) as controller:
    controller.run()
```

### 4. **Override in Python**

```python
from dataclasses import replace
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY

# Override checkpoint path
custom_cfg = replace(
    T1_23DOF_VELOCITY,
    policy=replace(
        T1_23DOF_VELOCITY.policy,
        checkpoint_path="models/my_model.pt"
    )
)

# Override robot PD gains (for a specific task)
from colosseum.deploy.config.values.robots import T1_23DOF_ROBOT_CFG

custom_robot = replace(
    T1_23DOF_ROBOT_CFG,
    name="Booster_T1_23DOF_Custom",
    joint_stiffness=(100.0, 100.0, ...)  # Custom gains
)

custom_cfg = replace(
    T1_23DOF_VELOCITY,
    robot=custom_robot
)
```

---

## Creating New Deployment Presets

### **Step 1: Create Robot Config (if needed)**

Only create a new robot config if you have a different robot or need custom hardware specs.

```python
# deploy/config/values/robots/my_robot.py
from colosseum.deploy.config.types import RobotConfig, PrepareStateConfig
from colosseum.utils import src_dir

MY_ROBOT_CFG = RobotConfig(
    name="MyRobot_12DOF",
    joint_names=(...),
    sim_joint_names=(...),
    body_names=(...),
    sim_body_names=(),
    joint_stiffness=(...),
    joint_damping=(...),
    default_joint_pos=(...),
    effort_limit=(...),
    parallel_joint_indices=(),
    mjcf_path=str(src_dir() / "robots" / "my_robot" / "xmls" / "model.xml"),
    prepare_state=PrepareStateConfig(
        stiffness=(...),
        damping=(...),
        joint_pos=(...),
    ),
)
```

### **Step 2: Create Policy Implementation**

```python
# tasks/my_task/deploy/my_robot/policy.py
from colosseum.deploy.config.registry import register_policy
from colosseum.deploy.config.types import PolicyConfig
from colosseum.deploy.core.base_controller import BaseController

class MyTaskPolicy:
    def __init__(self, config: PolicyConfig, controller: BaseController):
        self.config = config
        self.robot = controller.robot
        # ... load model, setup obs computation, etc.

    def reset(self) -> None:
        # Reset policy state
        pass

    def inference(self) -> torch.Tensor:
        # Run inference and return joint targets
        pass

# Register factory
@register_policy("my_task")
def create_my_task_policy(config: PolicyConfig, controller: BaseController):
    return MyTaskPolicy(config, controller)
```

### **Step 3: Create Deployment Preset**

```python
# deploy/config/values/deployments/my_deployment.py
from colosseum.deploy.config.types import (
    ControllerConfig,
    PolicyConfig,
    MujocoConfig,
    BoosterConfig,
)
from colosseum.deploy.config.values.robots import MY_ROBOT_CFG

MY_DEPLOYMENT = ControllerConfig(
    policy_dt=0.02,
    robot=MY_ROBOT_CFG,
    policy=PolicyConfig(
        policy_type="my_task",
        robot_name="MyRobot_12DOF",
        checkpoint_path="models/my_policy.pt",
        action_scale=0.25,
    ),
    mujoco=MujocoConfig(...),
    booster=BoosterConfig(...),
)
```

### **Step 4: Register with CLI**

```python
# deploy/scripts/deploy_tyro.py
from colosseum.deploy.config.values.deployments import MY_DEPLOYMENT
import colosseum.tasks.my_task.deploy.my_robot.policy  # noqa: F401

DEPLOYMENT_PRESETS = {
    "t1-velocity": T1_23DOF_VELOCITY,
    "my-deployment": MY_DEPLOYMENT,  # Add here
}
```

---

## Configuration Reference

### **RobotConfig**

Hardware specifications (shared across tasks):

```python
RobotConfig(
    name: str,                          # Robot identifier
    joint_names: tuple[str, ...],       # Real robot order
    sim_joint_names: tuple[str, ...],   # Simulation order (alphabetical)
    body_names: tuple[str, ...],        # Body names
    sim_body_names: tuple[str, ...],    # Simulation body order
    joint_stiffness: tuple[float, ...], # PD gains (Kp) - MUST match training!
    joint_damping: tuple[float, ...],   # PD gains (Kd) - MUST match training!
    default_joint_pos: tuple[float, ...],  # Default pose (radians)
    effort_limit: tuple[float, ...],    # Torque limits (Nm)
    parallel_joint_indices: tuple[int, ...],  # Coupled joints
    mjcf_path: str,                     # MuJoCo XML path
    prepare_state: PrepareStateConfig,  # Safe init pose
)
```

### **PolicyConfig**

Policy-specific parameters:

```python
PolicyConfig(
    policy_type: Literal["velocity", ...],  # Registered policy type
    robot_name: str,                         # Robot this policy was trained for
    checkpoint_path: str,                    # Model file path
    action_scale: float = 0.25,              # Must match training!
)
```

### **ControllerConfig**

Top-level deployment config:

```python
ControllerConfig(
    policy_dt: float = 0.02,                      # 50Hz
    robot: RobotConfig,                           # Robot hardware
    policy: PolicyConfig,                         # Policy parameters
    vel_command: VelocityCommandConfig | None,    # Velocity limits (optional)
    mujoco: MujocoConfig,                         # MuJoCo sim params
    booster: BoosterConfig,                       # Booster robot params
)
```

---

## Benefits Over Old System

| Feature | Old System | New System |
|---------|-----------|------------|
| **Type Safety** | `@configclass` (custom metaclass) | Pydantic (industry standard) |
| **Immutability** | Mutable (list attributes) | Frozen dataclasses (tuples) |
| **CLI** | argparse (manual parsing) | Tyro (auto-generated) |
| **Deep Overrides** | ❌ Not supported | ✅ `--config.policy.checkpoint-path` |
| **Policy Registration** | `PolicyCfg.constructor` (Callable) | Registry + decorator |
| **IDE Support** | ❌ Poor autocomplete | ✅ Full type hints |
| **Validation** | Runtime errors | Construction-time errors |
| **Presets** | Duplicated configs | Centralized + override |

---

## Migration Notes

### **Old Code (Still Works)**

```python
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY_DEPLOY_CFG
```

### **New Code (Recommended)**

```python
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY
```

The old name `T1_23DOF_VELOCITY_DEPLOY_CFG` is aliased for backward compatibility but will be deprecated.

---

## Next Steps

1. ✅ **Phases 1-5 Complete**: Config types, registry, presets, CLI
2. ⏭️ **Phase 6**: Update MujocoController (verify complete)
3. ⏭️ **Phase 9**: Deprecate old config system
4. ⏭️ **Phase 10**: Update documentation (CLAUDE.md, deployment guides)
5. ⏭️ **Phase 11**: Add tests (validation, immutability, CLI parsing)

---

## Troubleshooting

### **Policy Not Found**

```
ValueError: Unknown policy type: 'velocity'. Available: []
```

**Solution**: Import the policy module to trigger registration:
```python
import colosseum.tasks.velocity.deploy.t1_23dof.policy  # noqa: F401
```

### **Robot Mismatch**

```
ValueError: Policy trained for Booster_T1_23DOF, but deploying on Booster_T1_12DOF
```

**Solution**: Ensure `PolicyConfig.robot_name` matches `RobotConfig.name`.

### **Tuple Conversion**

```
TypeError: can't set attribute
```

**Solution**: Pydantic configs are **frozen**. Use `dataclasses.replace()` to create modified copies.

---

## Examples

See:
- `deploy/config/values/robots/t1.py` - Robot configuration
- `deploy/config/values/deployments/t1_velocity.py` - Deployment preset
- `tasks/velocity/deploy/t1_23dof/policy.py` - Policy implementation + registration
- `deploy/scripts/deploy_tyro.py` - CLI entry point
