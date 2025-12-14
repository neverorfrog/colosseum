# Colosseum Deployment Configuration Refactoring Plan

**Author**: Claude (based on holosoma_inference architecture analysis)
**Date**: 2025-12-12
**Status**: Ready for Implementation

---

## Objective
Refactor Colosseum's deployment configuration system to adopt Tyro + Pydantic, inspired by holosoma_inference architecture, while keeping current SDK integration and policy implementation unchanged.

## Scope
- ✅ **Replace**: `@configclass` → Pydantic dataclasses
- ✅ **Replace**: argparse CLI → Tyro auto-generated CLI
- ✅ **Replace**: `PolicyCfg.constructor` → Policy registry pattern
- ✅ **Convert**: Mutable lists → Immutable tuples
- ✅ **Create**: Config presets and subcommands
- ❌ **Keep unchanged**: SDK layer (BoosterRobotController, ROS2 integration)
- ❌ **Keep unchanged**: Policy layer (TorchScript, ObservationSpec)
- ❌ **Keep unchanged**: Controller logic (BaseController, MujocoController)

## User Decisions
- **Backward compatibility**: NOT required (clean slate)
- **Migration strategy**: Big bang (replace everything at once)
- **SDK abstraction**: Not implementing (Booster-only is fine)
- **Policy patterns**: Not changing (keep current implementation)

---

## Implementation Plan

### Phase 1: Dependencies and Structure

#### 1.1 Add Dependencies
**File**: `pyproject.toml`

Add to `[tool.pixi.feature.deploy.pypi-dependencies]`:
```toml
tyro = ">=0.8.0"
pydantic = ">=2.0.0"
```

#### 1.2 Create New Config Module Structure
**New directory**: `src/colosseum/deploy/config/`

```
src/colosseum/deploy/config/
├── __init__.py                    # Export main types
├── types/                         # Type definitions (Pydantic dataclasses)
│   ├── __init__.py
│   ├── robot.py                   # RobotConfig
│   ├── policy.py                  # PolicyConfig
│   ├── velocity.py                # VelocityCommandConfig
│   ├── mujoco.py                  # MujocoConfig
│   ├── booster.py                 # BoosterConfig
│   └── controller.py              # ControllerConfig (top-level)
├── values/                        # Preset configurations
│   ├── __init__.py
│   ├── robots/                    # Robot presets
│   │   ├── __init__.py
│   │   └── t1.py                  # T1_23DOF, T1_12DOF presets
│   └── deployments/               # Complete deployment presets
│       ├── __init__.py
│       └── t1_velocity.py         # T1 velocity deployment preset
└── registry.py                    # Policy factory registry
```

---

### Phase 2: Configuration Type Definitions

#### 2.1 RobotConfig
**File**: `src/colosseum/deploy/config/types/robot.py`

**Changes from current RobotCfg:**
- Use Pydantic `dataclass` with `frozen=True`
- Replace `List[float]` → `tuple[float, ...]` (immutable)
- Remove `MISSING` sentinel (all fields required or have defaults)
- Add docstrings (used by Tyro for help text)

```python
from pydantic import Field
from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class PrepareStateConfig:
    """Safe initialization pose for real robot."""
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    joint_pos: tuple[float, ...]

@dataclass(frozen=True)
class RobotConfig:
    """Robot hardware configuration.

    Defines all robot-specific parameters including joint configuration,
    control parameters, and hardware specifications.
    """

    # Robot identity
    name: str = Field(description="Robot identifier (e.g., 'Booster_T1_23DOF')")

    # Joint configuration (immutable tuples!)
    joint_names: tuple[str, ...] = Field(
        description="Joint names in real robot hardware order"
    )
    sim_joint_names: tuple[str, ...] = Field(
        description="Joint names in simulation order (alphabetical)"
    )

    # Body configuration
    body_names: tuple[str, ...]
    sim_body_names: tuple[str, ...]

    # Control parameters (MUST match training!)
    joint_stiffness: tuple[float, ...] = Field(
        description="PD gains (Kp) - must match training values exactly"
    )
    joint_damping: tuple[float, ...] = Field(
        description="PD gains (Kd) - must match training values exactly"
    )
    default_joint_pos: tuple[float, ...] = Field(
        description="Default standing pose (joint positions in radians)"
    )
    effort_limit: tuple[float, ...] = Field(
        description="Maximum torque limits per joint (Nm)"
    )

    # Hardware-specific
    parallel_joint_indices: tuple[int, ...] = Field(
        default=(),
        description="Indices of mechanically coupled joints"
    )
    mjcf_path: str = Field(
        description="Path to MuJoCo MJCF model file"
    )
    prepare_state: PrepareStateConfig = Field(
        description="Safe initialization pose configuration"
    )

    @property
    def num_joints(self) -> int:
        """Number of controlled joints."""
        return len(self.joint_names)

    @property
    def num_bodies(self) -> int:
        """Number of bodies."""
        return len(self.body_names)
```

**Notes:**
- All tuples are immutable (can't be modified after creation)
- Pydantic validates types at instantiation
- Docstrings become CLI help text via Tyro
- No `MISSING` - Pydantic requires all fields (or explicit defaults)

---

#### 2.2 PolicyConfig
**File**: `src/colosseum/deploy/config/types/policy.py`

**Key Change**: Remove `Callable` field, use string identifier instead.

```python
from typing import Literal
from pydantic.dataclasses import dataclass
from pydantic import Field

PolicyType = Literal["velocity", "whole_body_tracking"]

@dataclass(frozen=True)
class PolicyConfig:
    """Policy configuration.

    Specifies which policy to use and its parameters.
    """

    policy_type: PolicyType = Field(
        description="Type of policy to deploy"
    )
    checkpoint_path: str = Field(
        description="Path to policy checkpoint (.pt or .onnx)"
    )
    action_scale: float = Field(
        default=0.25,
        description="Action scaling factor (must match training)"
    )
```

**Usage Pattern**:
```python
# Old (with Callable):
PolicyCfg(constructor=lambda cfg, ctrl: T1VelocityPolicy(...))

# New (with registry):
PolicyConfig(policy_type="velocity", checkpoint_path="models/velocity.pt")
# → Resolved via POLICY_REGISTRY["velocity"]
```

---

#### 2.3 VelocityCommandConfig
**File**: `src/colosseum/deploy/config/types/velocity.py`

```python
from pydantic.dataclasses import dataclass
from pydantic import Field

@dataclass(frozen=True)
class VelocityCommandConfig:
    """Velocity command limits."""

    vx_max: float = Field(
        default=1.0,
        description="Maximum forward velocity (m/s)"
    )
    vy_max: float = Field(
        default=1.0,
        description="Maximum lateral velocity (m/s)"
    )
    vyaw_max: float = Field(
        default=1.0,
        description="Maximum yaw rate (rad/s)"
    )
```

---

#### 2.4 MujocoConfig
**File**: `src/colosseum/deploy/config/types/mujoco.py`

```python
from pydantic.dataclasses import dataclass
from pydantic import Field

@dataclass(frozen=True)
class MujocoConfig:
    """MuJoCo simulation parameters."""

    init_pos: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.6),
        description="Initial base position (x, y, z)"
    )
    init_quat: tuple[float, float, float, float] = Field(
        default=(1.0, 0.0, 0.0, 0.0),
        description="Initial orientation quaternion (w, x, y, z)"
    )
    decimation: int = Field(
        default=10,
        description="Physics steps per policy step"
    )
    save_states: bool = Field(
        default=False,
        description="Enable state logging for debugging"
    )
```

---

#### 2.5 BoosterConfig
**File**: `src/colosseum/deploy/config/types/booster.py`

```python
from pydantic.dataclasses import dataclass
from pydantic import Field

@dataclass(frozen=True)
class BoosterConfig:
    """Booster robot-specific parameters."""

    low_state_dt: float = Field(
        default=0.002,
        description="ROS2 low_state update period (500Hz = 0.002s)"
    )
    metrics_max_events: int = Field(
        default=2000,
        description="Maximum events in metrics buffer"
    )
```

---

#### 2.6 ControllerConfig (Top-Level)
**File**: `src/colosseum/deploy/config/types/controller.py`

```python
from typing import Optional
from pydantic.dataclasses import dataclass
from pydantic import Field, computed_field

from .robot import RobotConfig
from .policy import PolicyConfig
from .velocity import VelocityCommandConfig
from .mujoco import MujocoConfig
from .booster import BoosterConfig

@dataclass(frozen=True)
class ControllerConfig:
    """Top-level deployment configuration.

    Combines all configuration components for a complete deployment.
    """

    # Core parameters
    policy_dt: float = Field(
        default=0.02,
        description="Policy execution frequency (s). Default 0.02s = 50Hz"
    )

    # Required sub-configs
    robot: RobotConfig = Field(
        description="Robot hardware configuration"
    )
    policy: PolicyConfig = Field(
        description="Policy configuration"
    )

    # Optional sub-configs
    vel_command: Optional[VelocityCommandConfig] = Field(
        default=None,
        description="Velocity command configuration (for velocity tasks)"
    )

    # Backend-specific configs
    mujoco: MujocoConfig = Field(
        default_factory=MujocoConfig,
        description="MuJoCo simulation parameters"
    )
    booster: BoosterConfig = Field(
        default_factory=BoosterConfig,
        description="Booster robot parameters"
    )

    @computed_field
    @property
    def physics_dt(self) -> float:
        """Computed physics timestep."""
        return self.policy_dt / self.mujoco.decimation
```

**Key Features:**
- `frozen=True` makes instances immutable
- `computed_field` for derived properties
- `default_factory` for nested configs with defaults
- All configs are validated at construction time

---

### Phase 3: Policy Registry

#### 3.1 Policy Registry Pattern
**File**: `src/colosseum/deploy/config/registry.py`

Replace `PolicyCfg.constructor` pattern with factory registry:

```python
from typing import Protocol, Dict, Callable
from .types.policy import PolicyConfig

class Policy(Protocol):
    """Policy protocol (duck typing interface)."""
    def reset(self) -> None: ...
    def inference(self) -> torch.Tensor: ...

PolicyFactory = Callable[[PolicyConfig, "BaseController"], Policy]

class PolicyRegistry:
    """Registry for policy factories."""

    def __init__(self):
        self._factories: Dict[str, PolicyFactory] = {}

    def register(self, policy_type: str, factory: PolicyFactory):
        """Register a policy factory.

        Args:
            policy_type: Policy type identifier (e.g., "velocity")
            factory: Factory function (config, controller) -> Policy
        """
        self._factories[policy_type] = factory

    def create(
        self,
        config: PolicyConfig,
        controller: "BaseController"
    ) -> Policy:
        """Create policy instance from config."""
        if config.policy_type not in self._factories:
            raise ValueError(
                f"Unknown policy type: {config.policy_type}. "
                f"Available: {list(self._factories.keys())}"
            )
        return self._factories[config.policy_type](config, controller)

# Global registry
POLICY_REGISTRY = PolicyRegistry()

# Decorator for easy registration
def register_policy(policy_type: str):
    """Decorator to register a policy factory."""
    def decorator(factory: PolicyFactory):
        POLICY_REGISTRY.register(policy_type, factory)
        return factory
    return decorator
```

**Usage in Task Modules**:
```python
# tasks/velocity/deploy/t1_23dof/policy.py
from colosseum.deploy.config.registry import register_policy

@register_policy("velocity")
def create_velocity_policy(config: PolicyConfig, controller: BaseController):
    """Factory for T1 velocity policy."""
    return T1VelocityPolicy(
        model_path=config.checkpoint_path,
        action_scale=config.action_scale,
        controller=controller,
    )
```

---

### Phase 4: Configuration Presets

Create complete robot and deployment preset configurations.

**Files to create**:
- `src/colosseum/deploy/config/values/robots/t1.py` - T1 robot configurations
- `src/colosseum/deploy/config/values/deployments/t1_velocity.py` - T1 velocity deployment

See plan file for complete code examples.

---

### Phase 5: Tyro CLI Integration

#### 5.1 New CLI Entry Point
**File**: `src/colosseum/deploy/scripts/deploy_tyro.py`

Create Tyro-based CLI with subcommands:

```python
#!/usr/bin/env python3
"""Colosseum deployment CLI using Tyro."""

import tyro
from typing import Annotated, Literal

from colosseum.deploy.config.types import ControllerConfig
from colosseum.deploy.config.values.deployments import T1_23DOF_VELOCITY
from colosseum.deploy.core.controllers import MujocoController, BoosterRobotController

# Define presets as subcommands
DEPLOYMENT_PRESETS = {
    "t1-velocity": T1_23DOF_VELOCITY,
}

AnnotatedControllerConfig = Annotated[
    ControllerConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"deploy:{k}": v for k, v in DEPLOYMENT_PRESETS.items()}
        )
    )
]

def main(
    config: AnnotatedControllerConfig,
    target: Literal["mujoco", "real"] = "mujoco",
    network_interface: str = "lo",
    domain_id: int = 0,
) -> None:
    """Deploy trained policy on robot or simulation."""

    if target == "mujoco":
        controller = MujocoController(config)
    else:
        controller = BoosterRobotController(config, network_interface, domain_id)

    controller.run()

if __name__ == "__main__":
    tyro.cli(main)
```

**Usage**:
```bash
# List presets
pixi run python src/colosseum/deploy/scripts/deploy_tyro.py --help

# Run with preset
pixi run python src/colosseum/deploy/scripts/deploy_tyro.py deploy:t1-velocity

# Override config values
pixi run python src/colosseum/deploy/scripts/deploy_tyro.py deploy:t1-velocity \
    --config.policy.checkpoint-path models/new.pt \
    --config.vel-command.vx-max 1.5
```

---

### Phase 6: Update Controllers

Update controllers to use new config types:

**Key changes**:
- Replace imports from old config system
- Convert tuples to lists where needed for numpy/torch
- Use `POLICY_REGISTRY.create()` instead of `cfg.policy.constructor()`

**Files to modify**:
- `src/colosseum/deploy/core/controllers/base_controller.py`
- `src/colosseum/deploy/core/controllers/mujoco_controller.py`
- `src/colosseum/deploy/core/controllers/booster_robot_controller.py`

---

### Phase 7: Update Policy Registration

Register existing policies with decorator pattern:

**File**: `src/colosseum/tasks/velocity/deploy/t1_23dof/policy.py`

```python
from colosseum.deploy.config.registry import register_policy

@register_policy("velocity")
def create_velocity_policy(config: PolicyConfig, controller: BaseController):
    """Factory for T1 velocity tracking policy."""
    return T1VelocityPolicy(
        model_path=Path(config.checkpoint_path),
        controller=controller,
    )
```

---

### Phase 8: Update Deployment Bundles

Simplify task deployment bundles to use centralized configs:

**File**: `src/colosseum/tasks/velocity/deploy/t1_23dof/__init__.py`

```python
from colosseum.deploy.config.values.deployments.t1_velocity import T1_23DOF_VELOCITY

# Legacy export
T1_23DOF_VELOCITY_DEPLOY_CFG = T1_23DOF_VELOCITY

__all__ = ["T1_23DOF_VELOCITY", "T1_23DOF_VELOCITY_DEPLOY_CFG"]
```

**Files to delete**:
- `src/colosseum/tasks/velocity/deploy/t1_23dof/robot_cfg.py`
- `src/colosseum/tasks/velocity/deploy/t1_23dof/config.py`

---

### Phase 9: Remove Old Infrastructure

Deprecate old configuration system:

1. Add deprecation warnings to old config files
2. Mark IsaacLab configclass for removal
3. Deprecate old argparse CLI script

---

### Phase 10: Documentation Updates

Update documentation to reflect new system:

**Files to update**:
- `CLAUDE.md` - Add new config system section
- `docs/BOOSTER_T1_DEPLOYMENT_GUIDE.md` - Update CLI examples

---

### Phase 11: Testing Strategy

Add tests for new configuration system:

1. **Config validation tests** - Pydantic validation
2. **Immutability tests** - Frozen dataclasses
3. **CLI parsing tests** - Tyro argument parsing
4. **Integration tests** - End-to-end deployment

---

## Migration Checklist

- [ ] **Phase 1**: Add dependencies (Tyro, Pydantic)
- [ ] **Phase 2**: Create new config type definitions
- [ ] **Phase 3**: Implement policy registry
- [ ] **Phase 4**: Create robot and deployment presets
- [ ] **Phase 5**: Create Tyro CLI entry point
- [ ] **Phase 6**: Update controllers to use new configs
- [ ] **Phase 7**: Register existing policies
- [ ] **Phase 8**: Update deployment bundles
- [ ] **Phase 9**: Deprecate/remove old infrastructure
- [ ] **Phase 10**: Update documentation
- [ ] **Phase 11**: Add tests

---

## Key Files to Modify

### New Files (Create)
- `src/colosseum/deploy/config/types/robot.py`
- `src/colosseum/deploy/config/types/policy.py`
- `src/colosseum/deploy/config/types/velocity.py`
- `src/colosseum/deploy/config/types/mujoco.py`
- `src/colosseum/deploy/config/types/booster.py`
- `src/colosseum/deploy/config/types/controller.py`
- `src/colosseum/deploy/config/values/robots/t1.py`
- `src/colosseum/deploy/config/values/deployments/t1_velocity.py`
- `src/colosseum/deploy/config/registry.py`
- `src/colosseum/deploy/scripts/deploy_tyro.py`

### Existing Files (Modify)
- `pyproject.toml` (add dependencies)
- `src/colosseum/deploy/core/controllers/base_controller.py`
- `src/colosseum/deploy/core/controllers/mujoco_controller.py`
- `src/colosseum/deploy/core/controllers/booster_robot_controller.py`
- `src/colosseum/tasks/velocity/deploy/t1_23dof/policy.py`
- `src/colosseum/tasks/velocity/deploy/t1_23dof/__init__.py`
- `CLAUDE.md`
- `docs/BOOSTER_T1_DEPLOYMENT_GUIDE.md`

### Files to Delete
- `src/colosseum/tasks/velocity/deploy/t1_23dof/robot_cfg.py`
- `src/colosseum/tasks/velocity/deploy/t1_23dof/config.py`
- (Optional) `src/colosseum/deploy/core/utils/isaaclab/configclass.py`
- (Optional) `src/colosseum/deploy/scripts/deploy.py`

---

## Benefits of This Refactoring

1. **Type Safety**: Pydantic validates all configs at construction time
2. **Immutability**: Frozen configs prevent accidental modification bugs
3. **CLI Ergonomics**: Tyro auto-generates help text and allows deep overrides
4. **Consistency**: Same patterns as holosoma (easier knowledge transfer)
5. **IDE Support**: Better autocomplete and type checking
6. **Documentation**: Docstrings become CLI help automatically
7. **Validation**: Config errors caught early, not at runtime
8. **Maintainability**: Centralized config definitions

---

## Risks and Mitigations

### Risk 1: Tuple Conversion Overhead
**Issue**: Converting tuples → lists for numpy/torch may add overhead.
**Mitigation**: Only convert once at initialization, cache as lists.

### Risk 2: Breaking All Existing Code
**Issue**: Big bang migration breaks everything at once.
**Mitigation**: Thorough testing before deployment, keep old files as reference.

### Risk 3: Policy Registry Complexity
**Issue**: Registry pattern adds indirection.
**Mitigation**: Use decorators for easy registration, clear error messages.

---

## Timeline Estimate

- Phase 1-2 (Dependencies + Types): **2-3 hours**
- Phase 3-4 (Registry + Presets): **2-3 hours**
- Phase 5 (CLI): **1-2 hours**
- Phase 6-8 (Update Controllers + Bundles): **3-4 hours**
- Phase 9-10 (Cleanup + Docs): **1-2 hours**
- Phase 11 (Testing): **2-3 hours**

**Total: 11-17 hours** (1.5-2 working days)

---

## Success Criteria

- [ ] All deployment bundles use new config system
- [ ] Tyro CLI works with presets and overrides
- [ ] Policy registry supports existing policies
- [ ] MujocoController runs with new configs
- [ ] BoosterRobotController runs with new configs
- [ ] No references to old `@configclass` system
- [ ] Documentation updated
- [ ] Tests pass

---

## Post-Migration

### Next Steps After Completion:
1. Add more deployment presets (different robots/tasks)
2. Add validation for training/deployment consistency
3. Consider SDK abstraction layer (if multi-robot support needed later)
4. Add more sophisticated CLI features (interactive mode, config validation)
5. Integration with wandb for model loading
