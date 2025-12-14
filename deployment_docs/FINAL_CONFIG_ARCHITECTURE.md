# Final Deployment Configuration Architecture

**Status**: Phase 4-5 Complete + Simplified Registry
**Date**: 2025-12-14

## Overview

The deployment configuration system is based on three key principles:

1. **Policy-Robot Coupling**: Policies are trained for specific robots and cannot be mixed
2. **ABC-Enforced Constructor**: All policies must implement `__init__(controller: BaseController)`
3. **No Factory Functions**: Policy classes are stored directly in the registry

---

## Core Components

### **1. Policy ABC** (`deploy/core/policy.py`)

Enforces uniform interface for all policies:

```python
class Policy(ABC):
    """All policies must:
    1. Inherit from this class
    2. Implement __init__(self, controller: BaseController)
    3. Implement reset() and inference()
    """

    @abstractmethod
    def __init__(self, controller: BaseController):
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def inference(self) -> torch.Tensor:
        pass
```

**Key insight**: Since the ABC enforces the constructor signature, we don't need factory functions!

---

### **2. TaskRegistry** (`deploy/core/registry.py`)

Single registry storing task configurations and policy classes:

```python
class TaskRegistry:
    def __init__(self):
        self._tasks: Dict[str, ControllerConfig] = {}  # task_name → blueprint
        self._policies: Dict[str, type[Policy]] = {}   # policy_type → Policy class

    def register(
        self,
        task_name: str,
        config: ControllerConfig,      # Static blueprint
        policy_class: type[Policy]     # Just the class!
    ):
        self._tasks[task_name] = config
        self._policies[config.policy.policy_type] = policy_class

    def get_config(self, task_name: str) -> ControllerConfig:
        """Get blueprint for user/CLI."""
        return self._tasks[task_name]

    def get_policy(self, policy_type: str) -> type[Policy]:
        """Get policy class for BaseController."""
        return self._policies[policy_type]


TASK_REGISTRY = TaskRegistry()


@register_task(task_name: str):
    """Decorator for registering tasks."""
    def decorator(func: Callable[[], Tuple[ControllerConfig, type[Policy]]]):
        config, policy_class = func()
        TASK_REGISTRY.register(task_name, config, policy_class)
        return func
    return decorator
```

**Why store ControllerConfig and not BaseController?**
- **ControllerConfig** = Static blueprint (frozen, reusable)
- **BaseController** = Runtime instance (needs hardware, created per deployment)

---

### **3. Configuration Hierarchy**

```
ControllerConfig (top-level)
    ├─ robot: RobotConfig              # Hardware specs
    │   ├─ joint_names, stiffness, damping
    │   ├─ default_joint_pos, effort_limit
    │   └─ mjcf_path, prepare_state
    │
    ├─ policy: PolicyConfig            # Policy parameters
    │   ├─ policy_type: str            # e.g., "velocity"
    │   ├─ robot_name: str             # Must match RobotConfig.name
    │   ├─ checkpoint_path: str
    │   └─ action_scale: float
    │
    ├─ vel_command: VelocityCommandConfig (optional)
    ├─ mujoco: MujocoConfig            # Sim parameters
    └─ booster: BoosterConfig          # Real robot parameters
```

---

## Complete Flow

### **1. Task Registration** (happens at import time)

```python
# tasks/velocity/deploy/t1_23dof/__init__.py
from colosseum.deploy.core.registry import register_task
from colosseum.tasks.velocity.deploy.t1_23dof.config import T1_23DOF_VELOCITY
from colosseum.tasks.velocity.deploy.t1_23dof.policy import T1VelocityPolicy

@register_task("t1-velocity")
def t1_velocity_task():
    """Simple! Just return config and policy class."""
    return T1_23DOF_VELOCITY, T1VelocityPolicy
```

### **2. User Gets Config** (CLI or programmatic)

```python
# Option A: Via registry (recommended for CLI)
from colosseum.deploy.core.registry import TASK_REGISTRY
config = TASK_REGISTRY.get_config("t1-velocity")

# Option B: Direct import
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY
config = T1_23DOF_VELOCITY

# Option C: Override preset
from dataclasses import replace
config = replace(
    T1_23DOF_VELOCITY,
    policy=replace(T1_23DOF_VELOCITY.policy, checkpoint_path="my_model.pt")
)
```

### **3. Controller Instantiation**

```python
# User creates controller instance
from colosseum.deploy.backends.mujoco import MujocoController
controller = MujocoController(config)  # ← Calls BaseController.__init__

# Inside BaseController.__init__:
class BaseController:
    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.robot = BoosterRobot(cfg.robot)
        self.vel_command = VelocityCommand(cfg.vel_command) if cfg.vel_command else None

        # Get policy class from registry
        policy_class = TASK_REGISTRY.get_policy(cfg.policy.policy_type)

        # Direct instantiation (no factory!)
        self.policy = policy_class(self)  # ← Enforced by ABC

# Inside T1VelocityPolicy.__init__:
class T1VelocityPolicy(Policy):
    def __init__(self, controller: BaseController):
        # Access everything from controller
        self.config = controller.cfg.policy  # PolicyConfig
        self.robot = controller.robot
        self.vel_command = controller.vel_command

        # Load model, setup action scaling, etc.
        model_path = Path(self.config.checkpoint_path)
        self._model = torch.jit.load(str(model_path))
        self.action_scale = self.config.action_scale * ...
```

### **4. Deployment**

```python
with MujocoController(config) as controller:
    controller.run()
```

---

## Directory Structure

```
deploy/
├── config/                    # Type definitions
│   ├── robot.py               # RobotConfig, PrepareStateConfig
│   ├── policy.py              # PolicyConfig, VelocityCommandConfig
│   ├── backend.py             # MujocoConfig, BoosterConfig
│   └── controller.py          # ControllerConfig (top-level)
│
├── core/
│   ├── policy.py              # Policy ABC (enforces constructor)
│   ├── registry.py            # TaskRegistry (stores configs + classes)
│   ├── base_controller.py    # BaseController (creates policy instances)
│   └── robot.py               # RobotData, BoosterRobot
│
├── backends/                  # Controller implementations
│   ├── mujoco.py              # MujocoController
│   └── booster.py             # BoosterRobotPortal
│
└── scripts/
    └── deploy_tyro.py         # CLI entry point

robots/t1_23dof/
└── deploy_config.py           # T1_23DOF_ROBOT_CFG (centralized)

tasks/velocity/deploy/t1_23dof/
├── config.py                  # T1_23DOF_VELOCITY (ControllerConfig preset)
├── policy.py                  # T1VelocityPolicy (Policy implementation)
└── __init__.py                # @register_task("t1-velocity")
```

---

## Key Design Decisions

### **Q: Why not have two-tier registry (Policy + Deployment)?**
**A:** Policies are **coupled to robots**. A T1 velocity policy can ONLY run on T1. No need for that abstraction.

### **Q: Why enforce constructor in ABC instead of using factory functions?**
**A:** If the ABC enforces `__init__(controller)`, the factory is just the constructor! Storing the class directly is simpler.

### **Q: Why store ControllerConfig instead of BaseController in registry?**
**A:** ControllerConfig is the **static blueprint** (frozen dataclass). BaseController is the **runtime instance** (needs hardware, can only be created when deploying).

### **Q: Why does T1VelocityPolicy need BaseController?**
**A:** The controller provides:
- `controller.cfg.policy` → PolicyConfig (checkpoint path, action scale)
- `controller.robot` → Robot state (sensors, commands)
- `controller.vel_command` → Velocity commands (runtime state)

Everything the policy needs is in the controller!

### **Q: Why is `policy_type` separate from `task_name`?**
**A:**
- **task_name**: User-facing identifier (`"t1-velocity"`)
- **policy_type**: Internal policy implementation (`"velocity"`)

Same policy type can be used for multiple tasks (e.g., `"t1-velocity-rough-terrain"` also uses `"velocity"` policy).

---

## Benefits Over Old System

| Feature | Old System | New System |
|---------|-----------|------------|
| **Type Safety** | `@configclass` (custom) | Pydantic (industry standard) |
| **Immutability** | Mutable lists | Frozen dataclasses (tuples) |
| **Policy Registry** | Factory in config | ABC + class registry |
| **Constructor** | Not enforced | ABC enforces signature |
| **Factory Functions** | `PolicyCfg.constructor` | Not needed! |
| **Circular Imports** | Possible | Prevented (core/robot.py) |
| **Robot-Policy Coupling** | Implicit | Explicit (policy.robot_name) |
| **CLI** | argparse (manual) | Tyro (auto-generated) |
| **Deep Overrides** | ❌ | ✅ `--config.policy.checkpoint-path` |
| **IDE Support** | ❌ Poor | ✅ Full type hints |

---

## Usage Examples

### **CLI Usage**

```bash
# List available tasks
pixi run python src/colosseum/deploy/scripts/deploy_tyro.py --help

# Deploy with default preset
pixi run python src/colosseum/deploy/scripts/deploy_tyro.py deploy:t1-velocity

# Override checkpoint path
pixi run python src/colosseum/deploy/scripts/deploy_tyro.py deploy:t1-velocity \
    --config.policy.checkpoint-path models/my_model.pt

# Deploy on real robot
pixi run python src/colosseum/deploy/scripts/deploy_tyro.py deploy:t1-velocity \
    --target real --network-interface eth0
```

### **Programmatic Usage**

```python
# Use preset
from colosseum.deploy.core.registry import TASK_REGISTRY
from colosseum.deploy.backends.mujoco import MujocoController

config = TASK_REGISTRY.get_config("t1-velocity")
with MujocoController(config) as controller:
    controller.run()

# Override preset
from dataclasses import replace
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY

custom_config = replace(
    T1_23DOF_VELOCITY,
    policy=replace(
        T1_23DOF_VELOCITY.policy,
        checkpoint_path="models/custom.pt",
        action_scale=0.3
    )
)

with MujocoController(custom_config) as controller:
    controller.run()
```

### **Creating New Tasks**

```python
# 1. Create robot config (if new robot)
# robots/my_robot/deploy_config.py
MY_ROBOT_CFG = RobotConfig(
    name="MyRobot",
    joint_names=(...),
    # ... hardware specs
)

# 2. Create policy implementation
# tasks/my_task/deploy/my_robot/policy.py
class MyTaskPolicy(Policy):
    def __init__(self, controller: BaseController):
        # Access from controller
        self.config = controller.cfg.policy
        self.robot = controller.robot
        # ... load model, setup, etc.

    def reset(self) -> None:
        pass

    def inference(self) -> torch.Tensor:
        # ... run inference
        pass

# 3. Create deployment config
# tasks/my_task/deploy/my_robot/config.py
MY_TASK_CFG = ControllerConfig(
    robot=MY_ROBOT_CFG,
    policy=PolicyConfig(
        policy_type="my_task",
        robot_name="MyRobot",
        checkpoint_path="models/my_task.pt",
    ),
    # ... other configs
)

# 4. Register task
# tasks/my_task/deploy/my_robot/__init__.py
@register_task("my-task")
def my_task():
    return MY_TASK_CFG, MyTaskPolicy
```

---

## Next Steps

- ✅ Phase 1-5: Complete config system with registry
- ⏭️ Phase 6: Test end-to-end with real deployment
- ⏭️ Phase 9: Deprecate old config system
- ⏭️ Phase 10: Update CLAUDE.md and deployment guides
- ⏭️ Phase 11: Add tests (validation, immutability, CLI parsing)

---

## Summary

The final architecture achieves:
- ✅ **Simplicity**: No factory functions, just classes
- ✅ **Type Safety**: ABC enforces interface, Pydantic validates configs
- ✅ **Decoupling**: Config (blueprint) vs Controller (runtime)
- ✅ **Flexibility**: Centralized presets + override capability
- ✅ **Discoverability**: Registry lists all available tasks
- ✅ **CLI Integration**: Tyro auto-generates from registry

**Registry Stores:**
- `_tasks`: Map task_name → ControllerConfig (blueprint)
- `_policies`: Map policy_type → Policy class (implementation)

**Controller Creates:**
- `policy = TASK_REGISTRY.get_policy(cfg.policy.policy_type)(self)`
