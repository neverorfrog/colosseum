# Deployment Guide

This guide explains how to deploy trained policies from training to real robots, with intermediate testing via sim2sim.

## Table of Contents

1. [Deployment Pipeline](#deployment-pipeline)
2. [Observation Specification](#observation-specification)
3. [Creating a Deployment Bundle](#creating-a-deployment-bundle)
4. [Testing with Sim2Sim](#testing-with-sim2sim)
5. [Deploying to Real Robot](#deploying-to-real-robot)
6. [Advanced Topics](#advanced-topics)

---

## Deployment Pipeline

### Overview

```
┌─────────────────────┐
│  1. Train Policy    │  GPU-accelerated parallel training (mjlab)
│  (train/)           │  → Produces policy checkpoint (.pt)
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  2. Create Bundle   │  Define observation spec, robot config, policy
│  (tasks/*/deploy/)  │  → Task-specific deployment bundle
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  3. Test Sim2Sim    │  MuJoCo simulation
│  (MujocoController) │  → Verify policy works correctly
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  4. Deploy to Robot │  Real hardware
│  (RobotController)  │  → Production deployment
└─────────────────────┘
```

### Key Differences: Training vs Deployment

| Aspect | Training (mjlab) | Deployment |
|--------|------------------|------------|
| **Execution** | Vectorized (4096+ envs) | Single robot instance |
| **Hardware** | GPU (CUDA/Warp) | CPU or robot controller |
| **Observations** | Pre-computed by `Entity.data` | Computed from sensors |
| **Actions** | Normalized targets | Absolute joint positions |
| **Joint Order** | Alphabetical (simulation) | Hardware-specific |
| **Control Loop** | Managed by environment | Manual loop with PD control |

---

## Observation Specification

### The Contract Between Training and Deployment

The **observation specification** defines the contract that both training and deployment must follow:

```python
# tasks/velocity/mdp/observation_spec.py
from colosseum.deploy.core.observation_spec import ObservationSpec

class VelocityObservationSpec(ObservationSpec):
    """Defines the structure of velocity task observations."""

    @property
    def observation_names(self) -> List[str]:
        """Ordered list of observation components."""
        return [
            "velocity_commands",   # (3,) - vx, vy, vyaw
            "base_ang_vel",        # (3,) - angular velocity
            "projected_gravity",   # (3,) - gravity in base frame
            "joint_pos_rel",       # (num_joints,) - relative to default
            "joint_vel",           # (num_joints,)
            "last_action",         # (num_joints,)
        ]

    def get_component_size(self, name: str, num_joints: int) -> int:
        """Size of each component."""
        if name in ["velocity_commands", "base_ang_vel", "projected_gravity"]:
            return 3
        if name in ["joint_pos_rel", "joint_vel", "last_action"]:
            return num_joints
        raise ValueError(f"Unknown component: {name}")
```

### Using the Spec in Training

Training uses simple extraction functions that follow the spec:

```python
# tasks/velocity/mdp/observations.py
def base_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Tensor:
    """Extract base angular velocity (training)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b

def projected_gravity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Tensor:
    """Extract projected gravity (training)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b

# In training config
cfg.observations = {
    "policy": ObservationGroupCfg(
        terms={
            "base_ang_vel": ObservationTermCfg(func=base_ang_vel, ...),
            "projected_gravity": ObservationTermCfg(func=projected_gravity, ...),
            # ... other terms
        },
        concatenate_terms=True,  # Must be True!
    )
}
```

### Using the Spec in Deployment

Deployment computes observations and validates them:

```python
# tasks/velocity/deploy/t1_23dof/policy.py
from colosseum.tasks.velocity.mdp.observation_spec import VELOCITY_OBS_SPEC

class T1VelocityPolicy(Policy):
    def compute_observation(self) -> Tensor:
        # Compute observation components
        obs = torch.cat([
            self.velocity_commands,           # From command generator
            self.robot.data.root_ang_vel_b,   # From IMU
            self.robot.data.projected_gravity_b,  # Computed from quaternion
            joint_pos_rel,                    # Computed from encoders
            joint_vel,                        # From encoders
            self.last_action,                 # Stored from previous step
        ])

        # Validate observation
        VELOCITY_OBS_SPEC.validate_observation(obs, self.robot.num_joints)
        return obs
```

---

## Creating a Deployment Bundle

### Bundle Structure

A deployment bundle lives in `tasks/<task>/deploy/<robot>/` and contains:

```
tasks/velocity/deploy/t1_23dof/
├── __init__.py              # Export public interface
├── robot_cfg.py             # Robot hardware configuration
├── policy.py                # Policy implementation
├── config.py                # Deployment configuration
└── models/                  # Model checkpoints
    └── velocity_v1.pt
```

### Step 1: Define Robot Configuration

```python
# robot_cfg.py
from colosseum.deploy.core.controllers import RobotCfg

T1_23DOF_VELOCITY_ROBOT_CFG = RobotCfg(
    name="Booster_T1_23DOF_Velocity",

    # Real robot joint order (from hardware interface)
    joint_names=[
        "LHipYaw", "LHipRoll", "LHipPitch",
        "LKnee", "LAnklePitch", "LAnkleRoll",
        "RHipYaw", "RHipRoll", "RHipPitch",
        "RKnee", "RAnklePitch", "RAnkleRoll",
        # ... arms, waist, neck (23 total)
    ],

    # Simulation joint order (alphabetical, from mjlab)
    sim_joint_names=[
        "LAnklePitch", "LAnkleRoll", "LHipPitch",
        "LHipRoll", "LHipYaw", "LKnee",
        # ... (sorted alphabetically)
    ],

    # PD gains (must match training!)
    joint_stiffness=[50.0, 50.0, 50.0, ...],  # 23 values
    joint_damping=[2.0, 2.0, 2.0, ...],       # 23 values

    # Default standing pose
    default_joint_pos=[0.0, 0.0, -0.4, ...],  # 23 values

    # Motor effort limits
    effort_limit=[45.0, 45.0, 45.0, ...],     # 23 values

    # MuJoCo model path
    mjcf_path="robots/booster_t1/xmls/T1_23dof.xml",
)
```

### Step 2: Implement Policy

```python
# policy.py
import torch
from colosseum.deploy.core.controllers import Policy, PolicyCfg, RobotData
from colosseum.tasks.velocity.mdp.observation_spec import VELOCITY_OBS_SPEC

class T1VelocityPolicy(Policy):
    """Velocity tracking policy for T1 23-DOF."""

    def __init__(self, cfg: PolicyCfg, robot: RobotData):
        super().__init__(cfg, robot)

        # Load trained model
        self.model = torch.jit.load(cfg.checkpoint_path)
        self.model.eval()

        # Initialize state
        self.last_action = torch.zeros(robot.num_joints)

    def compute_observation(self) -> torch.Tensor:
        """Compute observation following VelocityObservationSpec."""

        # Compute joint positions relative to default (in simulation order!)
        joint_pos_rel = (
            self.robot.data.joint_pos[self.robot.data.real2sim_indexes]
            - self.robot.cfg.default_joint_pos[self.robot.data.real2sim_indexes]
        )

        # Compute joint velocities (in simulation order!)
        joint_vel = self.robot.data.joint_vel[self.robot.data.real2sim_indexes]

        # Concatenate observation
        obs = torch.cat([
            self.velocity_commands,
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            joint_pos_rel,
            joint_vel,
            self.last_action,
        ])

        # Validate
        VELOCITY_OBS_SPEC.validate_observation(obs, self.robot.num_joints)
        return obs

    def compute_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Run inference and return action."""
        with torch.no_grad():
            action = self.model(obs)

        # Convert from simulation order to real order
        action_real_order = action[self.robot.data.sim2real_indexes]

        # Store for next observation
        self.last_action = action.clone()

        return action_real_order
```

### Step 3: Create Deployment Configuration

```python
# config.py
from colosseum.deploy.core.controllers import (
    ControllerCfg,
    VelocityCommandCfg,
    MujocoControllerCfg,
)
from .robot_cfg import T1_23DOF_VELOCITY_ROBOT_CFG
from .policy import T1VelocityPolicy, T1VelocityPolicyCfg

def create_deployment_cfg(
    checkpoint_path: str = "models/velocity_v1.pt",
    vx_max: float = 1.0,
    vy_max: float = 0.5,
    vyaw_max: float = 1.0,
    policy_dt: float = 0.02,
) -> ControllerCfg:
    """Create velocity deployment configuration."""

    return ControllerCfg(
        policy_dt=policy_dt,  # 50Hz

        # Robot configuration
        robot=T1_23DOF_VELOCITY_ROBOT_CFG,

        # Velocity command settings
        vel_command=VelocityCommandCfg(
            vx_max=vx_max,
            vy_max=vy_max,
            vyaw_max=vyaw_max,
        ),

        # Policy
        policy=T1VelocityPolicyCfg(
            constructor=T1VelocityPolicy,
            checkpoint_path=checkpoint_path,
            action_scale_factor=0.25,  # Must match training!
        ),

        # MuJoCo sim settings
        mujoco=MujocoControllerCfg(
            init_pos=[0.0, 0.0, 0.6],
            init_quat=[1.0, 0.0, 0.0, 0.0],
            decimation=10,  # 500Hz physics / 50Hz policy
        ),
    )

# Pre-configured deployment
T1_23DOF_VELOCITY_DEPLOY_CFG = create_deployment_cfg()
```

### Step 4: Export Bundle

```python
# __init__.py
from .config import T1_23DOF_VELOCITY_DEPLOY_CFG, create_deployment_cfg
from .policy import T1VelocityPolicy, T1VelocityPolicyCfg
from .robot_cfg import T1_23DOF_VELOCITY_ROBOT_CFG

__all__ = [
    "T1_23DOF_VELOCITY_DEPLOY_CFG",
    "create_deployment_cfg",
    "T1VelocityPolicy",
    "T1VelocityPolicyCfg",
    "T1_23DOF_VELOCITY_ROBOT_CFG",
]
```

---

## Testing with Sim2Sim

Test your policy in simulation before deploying to real hardware:

```python
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY_DEPLOY_CFG
from colosseum.deploy.core.controllers import MujocoController

# Create sim2sim controller
controller = MujocoController(T1_23DOF_VELOCITY_DEPLOY_CFG)

# Run deployment
controller.run()
```

### Interactive Commands

Once running, you can send velocity commands via stdin:

```
vx: 0.5    # Move forward at 0.5 m/s
vy: 0.2    # Move left at 0.2 m/s
vyaw: 0.3  # Turn counterclockwise at 0.3 rad/s
```

### Debugging

Use the observation spec to inspect observations:

```python
# In policy.py
obs = self.compute_observation()

# Split observation for debugging
components = VELOCITY_OBS_SPEC.split_observation(obs, self.robot.num_joints)
print("Velocity commands:", components["velocity_commands"])
print("Base ang vel:", components["base_ang_vel"])
print("Joint pos rel:", components["joint_pos_rel"])
```

---

## Deploying to Real Robot

### Prerequisites

- ROS 2 installed
- Robot SDK installed
- Network connection to robot

### Running Deployment

```python
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY_DEPLOY_CFG
from colosseum.deploy.core.controllers import RobotController

# Create robot controller
controller = RobotController(T1_23DOF_VELOCITY_DEPLOY_CFG)

# Run deployment
controller.run()
```

### Safety Checks

The robot controller includes safety checks:

1. **Joint limits**: Actions clamped to safe ranges
2. **Velocity limits**: Joint velocities monitored
3. **Emergency stop**: Can be triggered remotely
4. **Prepare state**: Gradual transition to policy control

---

## Advanced Topics

### Custom Observation Components

To add new observation components:

1. **Update observation spec**:
```python
class VelocityObservationSpec(ObservationSpec):
    @property
    def observation_names(self) -> List[str]:
        return [
            # ... existing components
            "foot_contact",  # New component
        ]

    def get_component_size(self, name: str, num_joints: int) -> int:
        if name == "foot_contact":
            return 2  # Left and right foot
        # ... existing logic
```

2. **Add to training**:
```python
def foot_contact(env, asset_cfg):
    # Extract contact forces
    ...
```

3. **Add to deployment**:
```python
def compute_observation(self):
    foot_contact = self.compute_foot_contact()  # From force sensors
    obs = torch.cat([
        # ... existing components
        foot_contact,
    ])
```

### Multiple Robot Support

To deploy the same task on different robots:

```
tasks/velocity/deploy/
├── t1_23dof/          # T1 full body
├── t1_12dof/          # T1 legs only
└── unitree_g1/        # Different robot
```

Each deployment bundle is self-contained with its own robot config and (optionally) policy implementation.

### Versioning

Version your deployment bundles:

```
tasks/velocity/deploy/t1_23dof/
├── models/
│   ├── v1_baseline.pt
│   ├── v2_improved.pt
│   └── v3_latest.pt
└── config.py         # Switch between versions
```

---

## Summary

1. **Define observation spec** - Contract between training and deployment
2. **Create deployment bundle** - Robot config + policy + deployment config
3. **Test in sim2sim** - Verify policy works in MuJoCo
4. **Deploy to robot** - Run on real hardware with safety checks

The observation specification ensures training and deployment remain consistent, while task-specific bundles keep everything organized and versioned.
