# Colosseum Deployment Architecture

This document provides a comprehensive overview of Colosseum's deployment architecture, focusing on the observation specification system and task-specific deployment pattern.

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [Observation Specification System](#observation-specification-system)
3. [Task-Specific Deployment](#task-specific-deployment)
4. [Directory Structure](#directory-structure)
5. [Deployment Components](#deployment-components)
6. [Comparison with Other Frameworks](#comparison-with-other-frameworks)

---

## Core Philosophy

### Training vs Deployment: Different Operations

**Training** and **deployment** observe the world differently and should be treated as distinct operations:

| Aspect | Training (mjlab) | Deployment (Real/Sim) |
|--------|------------------|----------------------|
| **Data Source** | Pre-computed `Entity.data` | Raw sensor readings |
| **Operation** | Extract from memory | Process/compute |
| **Batch Size** | 4096+ environments | Single instance |
| **Framework** | mjlab-specific | Standalone |

**Key Insight**: Don't force them to share implementation code. Instead, share the **contract** (observation specification).

### What IS Shared: The Contract

Instead of sharing observation **computation**, we share the observation **specification**:

- ✅ **Clear contract**: Both sides know exactly what's expected
- ✅ **Validation**: Runtime checks ensure consistency
- ✅ **Documentation**: Self-documenting observation structure
- ✅ **Flexibility**: Each side implements optimally for its context

---

## Observation Specification System

### ObservationSpec Base Class

The abstract base class defines the interface for observation specifications:

```python
# deploy/core/observation_spec.py
class ObservationSpec(ABC):
    """Abstract base class for observation specifications."""

    @property
    @abstractmethod
    def observation_names(self) -> List[str]:
        """Ordered list of observation component names."""
        pass

    @abstractmethod
    def get_component_size(self, name: str, num_joints: int) -> int:
        """Size of specific observation component."""
        pass

    def compute_total_size(self, num_joints: int) -> int:
        """Total observation size."""
        return sum(
            self.get_component_size(name, num_joints)
            for name in self.observation_names
        )

    def validate_observation(self, obs: Tensor, num_joints: int):
        """Validate observation matches spec."""
        expected_size = self.compute_total_size(num_joints)
        if obs.shape[-1] != expected_size:
            raise ValueError(f"Expected {expected_size}, got {obs.shape[-1]}")

    def split_observation(self, obs: Tensor, num_joints: int) -> Dict[str, Tensor]:
        """Split into components for debugging."""
        components = {}
        idx = 0
        for name in self.observation_names:
            size = self.get_component_size(name, num_joints)
            components[name] = obs[..., idx:idx+size]
            idx += size
        return components

    @classmethod
    def from_observation_group_cfg(cls, cfg, num_joints: int):
        """Derive spec from training ObservationGroupCfg."""
        spec = cls()
        # Validates cfg.terms keys match spec.observation_names
        # Validates concatenate_terms=True
        return spec
```

**Benefits:**
- ✅ Can be used by any task
- ✅ Enforces consistent interface
- ✅ Optional derivation from training config
- ✅ Built-in validation and debugging

### Task-Specific Implementation

Each task defines its own observation specification:

```python
# tasks/velocity/mdp/observation_spec.py
class VelocityObservationSpec(ObservationSpec):
    """Velocity task observation specification."""

    @property
    def observation_names(self) -> List[str]:
        return [
            "velocity_commands",   # (3,)
            "base_ang_vel",        # (3,)
            "projected_gravity",   # (3,)
            "joint_pos_rel",       # (num_joints,)
            "joint_vel",           # (num_joints,)
            "last_action",         # (num_joints,)
        ]

    def get_component_size(self, name: str, num_joints: int) -> int:
        if name in ["velocity_commands", "base_ang_vel", "projected_gravity"]:
            return 3
        if name in ["joint_pos_rel", "joint_vel", "last_action"]:
            return num_joints
        raise ValueError(f"Unknown component: {name}")
```

### Usage in Training

Training uses simple extraction functions:

```python
# tasks/velocity/mdp/observations.py
def base_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Tensor:
    """Extract base angular velocity (training)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b  # Pre-computed by mjlab

def projected_gravity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Tensor:
    """Extract projected gravity (training)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b  # Pre-computed

# Training config
cfg.observations = {
    "policy": ObservationGroupCfg(
        terms={
            "base_ang_vel": ObservationTermCfg(func=base_ang_vel, ...),
            "projected_gravity": ObservationTermCfg(func=projected_gravity, ...),
        },
        concatenate_terms=True,
    )
}
```

### Usage in Deployment

Deployment computes observations and validates against spec:

```python
# tasks/velocity/deploy/t1_23dof/policy.py
class T1VelocityPolicy(Policy):
    def compute_observation(self):
        # Compute from sensor data
        obs = torch.cat([
            vel_cmd,                          # velocity_commands
            self.robot.data.root_ang_vel_b,   # base_ang_vel
            self.robot.data.projected_gravity_b,  # projected_gravity
            joint_pos_rel,                    # joint_pos_rel
            joint_vel,                        # joint_vel
            self.last_action,                 # last_action
        ])

        # Validate against spec!
        VELOCITY_OBS_SPEC.validate_observation(obs, self.robot.num_joints)
        return obs
```

---

## Task-Specific Deployment

### Rationale

A trained policy is inherently coupled to:
- Specific robot configuration (joint order, PD gains)
- Specific observation structure
- Specific action scale

**Solution**: Bundle them together in `tasks/<task>/deploy/<robot>/`

### Deployment Bundle Structure

```
tasks/velocity/deploy/t1_23dof/
├── __init__.py              # Export bundle
├── robot_cfg.py             # Robot config for this task
├── policy.py                # Policy implementation
├── config.py                # Deployment config
└── models/                  # Model checkpoints
    └── velocity_v1.pt
```

### Benefits

- ✅ Makes coupling explicit
- ✅ Easy to version (`v1`, `v2` folders)
- ✅ Self-contained (can distribute as package)
- ✅ Training config nearby (same task folder)
- ✅ Prevents mismatched robot/policy combinations

### Example Bundle Files

**robot_cfg.py** - Robot configuration:
```python
T1_23DOF_VELOCITY_ROBOT_CFG = RobotCfg(
    name="Booster_T1_23DOF_Velocity",
    joint_names=[...],         # Real robot order
    sim_joint_names=[...],     # Simulation order
    joint_stiffness=[...],
    joint_damping=[...],
    default_joint_pos=[...],
    effort_limit=[...],
    mjcf_path="path/to/T1_23dof.xml",
)
```

**policy.py** - Policy implementation:
```python
class T1VelocityPolicy(Policy):
    def __init__(self, cfg: PolicyCfg, robot_data: RobotData):
        super().__init__(cfg, robot_data)
        self.model = torch.jit.load(cfg.checkpoint_path)

    def compute_observation(self):
        obs = torch.cat([...])  # Follows VelocityObservationSpec
        VELOCITY_OBS_SPEC.validate_observation(obs, self.robot.num_joints)
        return obs

    def compute_action(self, obs: Tensor) -> Tensor:
        return self.model(obs)
```

**config.py** - Deployment configuration:
```python
def create_deployment_cfg(checkpoint_path: str, **kwargs) -> ControllerCfg:
    return ControllerCfg(
        robot=T1_23DOF_VELOCITY_ROBOT_CFG,  # Bundled!
        policy=T1VelocityPolicyCfg(checkpoint_path),
        vel_command=VelocityCommandCfg(**kwargs),
    )

T1_23DOF_VELOCITY_DEPLOY_CFG = create_deployment_cfg(
    checkpoint_path="models/velocity_v1.pt"
)
```

---

## Directory Structure

```
src/colosseum/
├── mdp/                          # Universal physics/math utilities
│   ├── observations.py           # Generic functions (quaternion rotations)
│   └── rewards.py                # Generic reward kernels
│
├── robots/
│   └── booster_t1/
│       ├── xmls/                 # MuJoCo models
│       ├── t1_actuators.py       # Motor specs (training)
│       ├── t1_contacts.py        # Collision configs (training)
│       ├── t1_constants.py       # Entity configs (training)
│       └── mdp/                  # Robot-specific (if needed)
│           └── observations.py   # T1-specific sensors
│
├── tasks/
│   └── velocity/
│       ├── config/               # Training configs
│       │   └── t1/env_cfgs.py
│       ├── mdp/
│       │   ├── observation_spec.py   # ★ CONTRACT (shared)
│       │   └── observations.py       # Training MDP functions
│       ├── rl/                   # RL algorithm configs
│       └── deploy/               # ★ Task-specific deployment
│           └── t1_23dof/         # Complete deployment bundle
│               ├── robot_cfg.py      # Robot config
│               ├── policy.py         # Policy implementation
│               ├── config.py         # Deployment config
│               └── models/           # Model checkpoints
│
└── deploy/
    └── core/
        ├── observation_spec.py   # Abstract base class
        ├── controllers/          # Base controller abstractions
        └── utils/
            └── registry.py       # Optional task registry
```

---

## Deployment Components

### Controllers

**Base Controller** ([base_controller.py](../src/colosseum/deploy/core/controllers/base_controller.py)):
- Abstract `BaseController` class
- `Policy` abstract class
- `RobotData` for sensor/actuator interface

**MuJoCo Controller** ([mujoco_controller.py](../src/colosseum/deploy/core/controllers/mujoco_controller.py)):
- Sim2sim deployment using MuJoCo
- For testing policies before real robot deployment
- Simulates sensor readings and PD control

**Robot Controller** (optional):
- Real hardware interface
- ROS 2 integration
- Safety checks and state machine

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

---

## Comparison with Other Frameworks

### Legged Gym / Humanoid-Gym

**Their Approach:**
- Separate training and deployment codebases
- Observation logic duplicated between sim and real
- Manual alignment required

**Colosseum Advantages:**
- ✅ Shared observation contract (no duplication)
- ✅ Guaranteed consistency via validation
- ✅ Task-specific deployment bundles

### Isaac Lab

**Their Approach:**
- Manager-based training (similar to Colosseum)
- Deployment left to user
- No shared MDP functions

**Colosseum Advantages:**
- ✅ ObservationSpec system
- ✅ Built-in deployment infrastructure
- ✅ Automatic joint mapping

### rl_sar

**Their Approach:**
- Backend abstraction (IsaacGym, MuJoCo, Real)
- Still requires manual observation alignment

**Colosseum Advantages:**
- ✅ Observation specification (cleaner than backend abstraction)
- ✅ Explicit train/deploy separation
- ✅ Task-specific deployment bundles

---

## Summary

Colosseum's deployment architecture:

1. **Observation Specification**: Share contract, not implementation
2. **Task-Specific Deployment**: Bundle robot + policy together
3. **Runtime Validation**: Ensure training/deployment consistency
4. **Automatic Joint Mapping**: Handle sim↔real order differences
5. **Simple & Honest**: Accept reality instead of fighting it

This approach is more honest about the reality of RL deployment and results in cleaner, more maintainable code.
