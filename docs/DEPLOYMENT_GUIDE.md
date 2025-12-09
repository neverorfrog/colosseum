# Deployment Guide: From mjlab Training to T1 Robot

This guide explains how to deploy policies trained with mjlab on the Booster T1 robot, with intermediate testing via sim2sim (MuJoCo).

> **Important**: This guide uses the **shared observations architecture** to ensure training and deployment observations match exactly. See [SHARED_OBSERVATIONS.md](SHARED_OBSERVATIONS.md) for detailed implementation.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Understanding booster_deploy](#understanding-booster_deploy)
4. [Integration into Colosseum](#integration-into-colosseum)
5. [Shared Observations Pattern](#shared-observations-pattern)
6. [Step-by-Step: Deploy Velocity Task](#step-by-step-deploy-velocity-task)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### The Deployment Pipeline

```
┌─────────────────────┐
│  1. Train in mjlab  │  GPU-accelerated parallel training
│  (train/)           │  → Produces policy checkpoint
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  2. Export Policy   │  Convert to TorchScript (.pt)
│                     │  → Standalone deployable model
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  3. Test Sim2Sim    │  MuJoCo simulation (play/)
│  (play.py)          │  → Verify policy works
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  4. Deploy to Robot │  Real hardware (deploy/)
│  (deploy.py)        │  → Production deployment
└─────────────────────┘
```

### Key Differences: Training vs Deployment

| Aspect | Training (mjlab) | Deployment (booster_deploy) |
|--------|-----------------|----------------------------|
| **Execution** | Vectorized (4096+ envs in parallel) | Single robot instance |
| **Hardware** | GPU (CUDA/Warp) | CPU or robot controller |
| **Observations** | Provided by `ManagerBasedRlEnv` | Computed manually in policy |
| **Actions** | Normalized targets | Absolute joint positions |
| **Joint Order** | `sim_joint_names` (alphabetical) | `joint_names` (real robot order) |
| **Control Loop** | Managed by environment | Manual loop with PD control |

---

## Architecture

### Colosseum Directory Structure

```
colosseum/
├── src/colosseum/
│   ├── tasks/                    # SHARED task definitions
│   │   └── velocity/             # Velocity task (shared)
│   │       ├── observations.py   # Shared observation functions ⭐
│   │       └── constants.py      # Task constants
│   │
│   ├── train/                    # Training infrastructure
│   │   └── tasks/
│   │       ├── cartpole/         # Example task
│   │       └── velocity/         # Velocity tracking task
│   │           ├── config/       # Environment configs
│   │           ├── mdp/          # MDP wrappers (use shared obs)
│   │           │   └── observations.py
│   │           └── rl/           # RL algorithm configs
│   │
│   ├── deploy/                   # Deployment infrastructure
│   │   ├── core/                 # Shared deployment code
│   │   │   ├── controllers/      # Controller implementations
│   │   │   │   ├── base_controller.py
│   │   │   │   ├── mujoco_controller.py
│   │   │   │   ├── booster_robot_controller.py
│   │   │   │   └── controller_cfg.py
│   │   │   └── utils/            # Utilities
│   │   │       ├── registry.py   # Task registration
│   │   │       ├── motion_loader.py
│   │   │       └── isaaclab/     # IsaacLab utilities
│   │   │
│   │   └── tasks/                # Task-specific policies
│   │       └── velocity/         # Velocity policy
│   │           ├── __init__.py
│   │           ├── policy.py     # Uses shared observations ⭐
│   │           └── models/       # Checkpoints
│   │               └── policy.pt
│   │
│   ├── play/                     # Sim2sim testing
│   │   └── scripts/
│   │       └── play.py           # Sim2sim runner
│   │
│   └── robots/                   # Robot definitions
│       └── booster_t1/
│           ├── t1_constants.py   # Robot config (train + deploy)
│           ├── t1_actuators.py
│           └── t1_contacts.py
```

**Key Addition**: The `src/colosseum/tasks/` directory contains **shared task definitions** that work in both training and deployment. See [Shared Observations Pattern](#shared-observations-pattern).

---

## Understanding booster_deploy

### What is booster_deploy?

`booster_deploy` is a lightweight deployment harness that provides:
- **Unified interface** for sim2sim (MuJoCo) and sim2real (robot) deployment
- **Controller abstractions** that handle sensor reading and action application
- **Task registry** for managing multiple deployment configurations

### Core Components

#### 1. Controllers

**BaseController** ([base_controller.py](../src/colosseum/deploy/core/controllers/base_controller.py))
- Abstract base class defining the controller interface
- Execution flow:
  ```python
  start()                    # Initialize policy
  while running:
      update_state()         # Read sensors
      action = policy_step() # Run inference
      ctrl_step(action)      # Apply action
  stop()                     # Cleanup
  ```

**MujocoController** ([mujoco_controller.py](../src/colosseum/deploy/core/controllers/mujoco_controller.py))
- Sim2sim implementation using MuJoCo
- Runs PD control at high frequency (decimation steps)
- Provides interactive viewer
- Handles velocity command input from stdin

**BoosterRobotController** ([booster_robot_controller.py](../src/colosseum/deploy/core/controllers/booster_robot_controller.py))
- Real robot implementation
- Communicates via ROS 2 (`/low_state`, `/low_cmd` topics)
- Requires Booster Robotics SDK

#### 2. Configuration System

All configurations use the `@configclass` decorator from IsaacLab:

**RobotCfg** - Robot hardware specification:
```python
@configclass
class RobotCfg:
    name: str                        # Robot name
    joint_names: list[str]           # Real robot joint order
    sim_joint_names: list[str]       # Simulation joint order
    joint_stiffness: List[float]     # PD gains
    joint_damping: List[float]
    default_joint_pos: List[float]   # Home position
    effort_limit: List[float]        # Torque limits
    mjcf_path: str                   # MuJoCo XML path
```

**PolicyCfg** - Policy specification:
```python
@configclass
class PolicyCfg:
    constructor: Callable  # Policy class constructor
    # Task-specific parameters...
```

**ControllerCfg** - Complete deployment configuration:
```python
@configclass
class ControllerCfg:
    policy_dt: float = 0.02          # Policy frequency (50 Hz)
    robot: RobotCfg                  # Robot config
    policy: PolicyCfg                # Policy config
    vel_command: VelocityCommandCfg  # For velocity tasks
    mujoco: MujocoControllerCfg      # Sim2sim settings
    booster: BoosterRobotControllerCfg  # Real robot settings
```

#### 3. Policy Interface

Every deployable policy must implement:

```python
class Policy:
    def __init__(self, cfg: PolicyCfg, controller: BaseController):
        self.cfg = cfg
        self.controller = controller
        # Load model, initialize state...

    def reset(self) -> None:
        """Called when controller starts."""
        # Reset episode state

    def inference(self) -> torch.Tensor:
        """Called each step to get actions.

        Returns:
            Joint position targets (N,) where N = number of joints
        """
        # Compute observations
        # Run model inference
        # Return joint targets
```

#### 4. Joint Ordering

**Critical**: Joint order differs between simulation and real robot!

**Real Robot Order** (`joint_names`):
```python
# T1 23DOF real robot order
[
    "AAHead_yaw", "Head_pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", ...
    "Left_Hip_Pitch", "Left_Hip_Roll", ...
]
```

**Simulation Order** (`sim_joint_names`):
```python
# IsaacLab/mjlab alphabetical order
[
    "AAHead_yaw",
    "Left_Shoulder_Pitch",
    "Right_Shoulder_Pitch",
    "Waist",
    "Head_pitch",
    ...
]
```

**Mapping**: Use `robot.data.real2sim_joint_indexes` and `sim2real_joint_indexes`:
```python
# When reading observations (real → sim order)
real2sim_map = self.robot.data.real2sim_joint_indexes
joint_pos_sim = self.robot.data.joint_pos[real2sim_map]

# When returning actions (sim → real order)
sim2real_map = self.robot.data.sim2real_joint_indexes
joint_targets_real = action[sim2real_map]
```

---

## Integration into Colosseum

### What We've Done (Step 1)

Copied booster_deploy core infrastructure into colosseum:

```bash
# Controllers
src/colosseum/deploy/core/controllers/
├── base_controller.py
├── mujoco_controller.py
├── booster_robot_controller.py
└── controller_cfg.py

# Utilities
src/colosseum/deploy/core/utils/
├── registry.py
├── motion_loader.py
├── metrics.py
├── synced_array.py
├── remote_control_service.py
└── isaaclab/
    ├── configclass.py
    ├── dict.py
    ├── math.py
    ├── array.py
    └── string.py
```

Import paths updated to use `colosseum.deploy.core.*` instead of relative imports.

---

## Shared Observations Pattern

### Why Shared Observations?

**Problem**: Observations are computed differently in training vs deployment, leading to:
- Code duplication
- Maintenance burden
- Risk of inconsistency

**Solution**: Create **shared observation functions** that work in both contexts.

### Architecture

```
┌─────────────────────────────────────────┐
│    Shared Observation Functions         │
│  (src/colosseum/tasks/velocity/)        │
│                                          │
│  Pure functions, no env dependencies    │
└─────────┬──────────────────┬────────────┘
          │                  │
  ┌───────▼────────┐  ┌─────▼──────────┐
  │ Training MDP   │  │ Deployment     │
  │  (Wrappers)    │  │  Policy        │
  │                │  │                │
  │ Extract entity │  │ Direct calls   │
  │ Call shared fn │  │ Handle mapping │
  └────────────────┘  └────────────────┘
```

### Quick Example

**Shared Function** (`src/colosseum/tasks/velocity/observations.py`):
```python
def compute_projected_gravity(robot_data) -> torch.Tensor:
    """Gravity vector in base frame (works in both contexts)."""
    gravity_w = torch.tensor([0.0, 0.0, -1.0])
    return quat_rotate_inverse(robot_data.root_quat_w, gravity_w)
```

**Training Wrapper** (`src/colosseum/train/tasks/velocity/mdp/observations.py`):
```python
from colosseum.tasks.velocity.observations import compute_projected_gravity

def projected_gravity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg):
    robot: Entity = env.scene[asset_cfg.name]
    return compute_projected_gravity(robot.data)  # Use shared function
```

**Deployment** (`src/colosseum/deploy/tasks/velocity/policy.py`):
```python
from colosseum.tasks.velocity.observations import compute_projected_gravity

def compute_observation(self):
    gravity = compute_projected_gravity(self.robot.data)  # Same function!
    # ...
```

### Full Implementation

For complete step-by-step implementation, see **[SHARED_OBSERVATIONS.md](SHARED_OBSERVATIONS.md)**.

---

### What You'll Do Next

#### Step 2: Implement Shared Observations (RECOMMENDED)

**Follow**: [SHARED_OBSERVATIONS.md](SHARED_OBSERVATIONS.md) for detailed steps.

**Summary**:
1. Create `src/colosseum/tasks/velocity/observations.py` with shared functions
2. Create MDP wrappers in `src/colosseum/train/tasks/velocity/mdp/observations.py`
3. Update deployment policy to use shared functions
4. Write unit tests to verify consistency

**Benefits**:
- ✅ Guaranteed observation consistency
- ✅ Single source of truth
- ✅ Easy to maintain and test

#### Alternative: Create Velocity Task Deployment Adapter (Not Recommended)

Create `src/colosseum/deploy/tasks/velocity/policy.py`:

**Key Requirements:**
1. Load TorchScript model from checkpoint
2. Implement `compute_observation()` matching training observations
3. Handle joint order mapping
4. Scale actions correctly

**Template:**
```python
import torch
from pathlib import Path
from colosseum.deploy.core.controllers import BaseController, Policy, PolicyCfg
from colosseum.deploy.core.utils.isaaclab.configclass import configclass
from colosseum.deploy.core.utils.isaaclab import math as lab_math

class VelocityPolicy(Policy):
    def __init__(self, cfg: 'VelocityPolicyCfg', controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot
        self.vel_command = controller.vel_command

        # Load TorchScript model
        model_path = Path(__file__).parent / cfg.checkpoint_path
        self._model: torch.jit.ScriptModule = torch.jit.load(model_path)
        self._model.eval()

        # Compute action scale (from training config)
        # action_scale = 0.25 * effort_limit / stiffness
        self.action_scale = (
            0.25 * self.robot.effort_limit / self.robot.joint_stiffness
        )

    def reset(self) -> None:
        """Reset policy state."""
        self.last_action = torch.zeros(
            self.robot.num_joints, dtype=torch.float32
        )
        # Initialize any episode-specific state

    def compute_observation(self) -> torch.Tensor:
        """Compute observations matching training setup.

        CRITICAL: This must match EXACTLY what your training env provides!
        Check your velocity task config to see what observations are used.
        """
        # Common velocity task observations:
        # - Velocity commands (vx, vy, vyaw)
        # - Base angular velocity
        # - Projected gravity vector
        # - Joint positions (relative to default)
        # - Joint velocities
        # - Last action

        # Get velocity commands
        cmd = torch.tensor([
            self.vel_command.lin_vel_x,
            self.vel_command.lin_vel_y,
            self.vel_command.ang_vel_yaw,
        ], dtype=torch.float32)

        # Base angular velocity in base frame
        base_ang_vel = self.robot.data.root_ang_vel_b

        # Projected gravity (gravity vector in base frame)
        # Assuming world frame gravity is [0, 0, -9.81]
        base_quat = self.robot.data.root_quat_w
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        projected_gravity = lab_math.quat_rotate_inverse(
            base_quat, gravity_w
        )

        # Joint positions and velocities (in simulation order!)
        real2sim_map = self.robot.data.real2sim_joint_indexes
        joint_pos = (
            self.robot.data.joint_pos[real2sim_map]
            - self.robot.default_joint_pos[real2sim_map]
        )
        joint_vel = self.robot.data.joint_vel[real2sim_map]

        # Concatenate observations
        obs = torch.cat([
            cmd,                    # (3,)
            base_ang_vel,          # (3,)
            projected_gravity,     # (3,)
            joint_pos,             # (num_joints,)
            joint_vel,             # (num_joints,)
            self.last_action,      # (num_joints,)
        ])

        return obs.reshape(1, -1)

    def inference(self) -> torch.Tensor:
        """Run policy inference and return joint targets.

        Returns:
            Joint position targets in REAL robot order (N,)
        """
        with torch.no_grad():
            # Compute observations
            obs = self.compute_observation()

            # Run model (returns normalized actions in sim order)
            action = self._model(obs).flatten()

            # Store for next step
            self.last_action = action

            # Map from sim order to real order
            sim2real_map = self.robot.data.sim2real_joint_indexes

            # Scale and add default positions
            joint_targets = (
                action[sim2real_map] * self.action_scale
                + self.robot.default_joint_pos
            )

            return joint_targets


@configclass
class VelocityPolicyCfg(PolicyCfg):
    constructor = VelocityPolicy
    checkpoint_path: str = "models/policy.pt"
```

**Create ControllerCfg:**

`src/colosseum/deploy/tasks/velocity/__init__.py`:

```python
from colosseum.deploy.core.controllers import (
    ControllerCfg,
    MujocoControllerCfg,
    VelocityCommandCfg,
)
from colosseum.deploy.core.utils import register_task
from colosseum.deploy.core.utils.isaaclab.configclass import configclass

from colosseum.robots.booster_t1.t1_constants import T1_23DOF_DEPLOY_CFG
from .policy import VelocityPolicy, VelocityPolicyCfg


@configclass
class T1VelocityControllerCfg(ControllerCfg):
    """Velocity task deployment configuration for T1."""

    robot = T1_23DOF_DEPLOY_CFG  # Need to create this in t1_constants.py

    vel_command = VelocityCommandCfg(
        vx_max=1.0,
        vy_max=0.5,
        vyaw_max=1.0,
    )

    policy = VelocityPolicyCfg(
        checkpoint_path="models/velocity_policy.pt"
    )

    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.6],  # Spawn height
        decimation=10,              # 10 physics steps per policy step
    )


# Register task
register_task("t1_velocity", T1VelocityControllerCfg())
```

**Add T1 Deploy Config to t1_constants.py:**

```python
from colosseum.deploy.core.controllers import RobotCfg, PrepareStateCfg

T1_23DOF_DEPLOY_CFG = RobotCfg(
    name="Booster_T1_23DOF",
    joint_names=[
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Waist",
        "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
        "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
        "Right_Ankle_Pitch", "Right_Ankle_Roll",
    ],
    sim_joint_names=[  # From your training task (alphabetical order)
        "AAHead_yaw",
        "Left_Shoulder_Pitch",
        "Right_Shoulder_Pitch",
        "Waist",
        "Head_pitch",
        "Left_Shoulder_Roll",
        "Right_Shoulder_Roll",
        "Left_Hip_Pitch",
        "Right_Hip_Pitch",
        "Left_Elbow_Pitch",
        "Right_Elbow_Pitch",
        "Left_Hip_Roll",
        "Right_Hip_Roll",
        "Left_Elbow_Yaw",
        "Right_Elbow_Yaw",
        "Left_Hip_Yaw",
        "Right_Hip_Yaw",
        "Left_Knee_Pitch",
        "Right_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Right_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Ankle_Roll",
    ],
    body_names=[...],  # Copy from T1_23DOF_ENTITY_CFG
    joint_stiffness=[...],  # From your training actuator configs
    joint_damping=[...],
    default_joint_pos=[...],
    effort_limit=[...],
    parallel_joint_indices=[15, 16, 21, 22],  # Ankle joints
    mjcf_path="{BOOSTER_ASSETS_DIR}/robots/T1/T1_23dof.xml",
    prepare_state=PrepareStateCfg(...),  # For real robot initialization
)
```

#### Step 3: Create Play Script for Sim2Sim

`src/colosseum/play/scripts/play.py`:

```python
#!/usr/bin/env python3
"""Sim2sim deployment testing script.

Loads a trained policy and runs it in MuJoCo simulation for testing
before deploying to real hardware.
"""

import argparse
import pkgutil
import sys
from pathlib import Path

# Add src to path to enable imports
colosseum_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(colosseum_root))

from colosseum.deploy.core.controllers import MujocoController
from colosseum.deploy.core.utils import get_task, list_tasks


def main():
    parser = argparse.ArgumentParser(description="Play trained policies in MuJoCo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=str, help="Task name to run")
    group.add_argument("-l", "--list", action="store_true", help="List available tasks")
    args = parser.parse_args()

    # Auto-import all task modules to register them
    import colosseum.deploy.tasks as tasks_pkg
    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="colosseum.deploy.tasks."):
        try:
            __import__(mod_info.name)
        except Exception as e:
            print(f"Warning: Failed to import {mod_info.name}: {e}")

    # List or run task
    if args.list:
        print("Available tasks:")
        for task_name, cfg in list_tasks().items():
            cls = type(cfg)
            print(f"  {task_name:20s} : {cls.__module__}.{cls.__qualname__}")
        return

    # Load and run task
    try:
        task_cfg = get_task(args.task)
    except KeyError:
        print(f"Error: Unknown task '{args.task}'")
        print(f"Available tasks: {list(list_tasks().keys())}")
        sys.exit(1)

    print(f"Running task '{args.task}' in MuJoCo simulation...")
    print(f"Policy: {task_cfg.policy.checkpoint_path}")
    print(f"Robot: {task_cfg.robot.name}")
    print(f"Frequency: {1/task_cfg.policy_dt:.1f} Hz")

    if task_cfg.vel_command is not None:
        print("\nVelocity command mode enabled.")
        print("Type three numbers (vx vy vyaw) to set velocity commands.")

    controller = MujocoController(task_cfg)
    controller.run()


if __name__ == "__main__":
    main()
```

Make it executable:
```bash
chmod +x src/colosseum/play/scripts/play.py
```

---

## Step-by-Step: Deploy Velocity Task

### 1. Train Policy in mjlab

```bash
# From colosseum root
pixi run python -m mjlab.scripts.train --task velocity/t1_rough
```

This produces checkpoints in `logs/rsl_rl/exp_name/`.

### 2. Export Policy as TorchScript

Create `scripts/export_policy.py`:

```python
#!/usr/bin/env python3
"""Export trained policy to TorchScript for deployment."""

import argparse
import torch
from pathlib import Path

def export_policy(checkpoint_path: Path, output_path: Path):
    """Export policy to TorchScript.

    Args:
        checkpoint_path: Path to training checkpoint (.pt)
        output_path: Where to save TorchScript model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract actor network
    # (Adjust based on your RL algorithm structure)
    actor = checkpoint["model"]["actor"]  # RSL-RL structure
    actor.eval()

    # Create example input (match your observation size)
    # For velocity task: typically cmd(3) + ang_vel(3) + gravity(3) + joints(2*N) + action(N)
    num_joints = 23  # T1 full body
    obs_size = 3 + 3 + 3 + 2*num_joints + num_joints
    example_input = torch.zeros(1, obs_size)

    # Trace model
    scripted_model = torch.jit.trace(actor, example_input)

    # Save
    torch.jit.save(scripted_model, str(output_path))
    print(f"Exported policy to {output_path}")
    print(f"  Input shape: {example_input.shape}")
    print(f"  Model size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Training checkpoint (.pt)")
    parser.add_argument("--output", type=Path, required=True, help="Output path")
    args = parser.parse_args()

    export_policy(args.checkpoint, args.output)
```

Export:
```bash
python scripts/export_policy.py \
    logs/rsl_rl/exp_name/model_5000.pt \
    --output src/colosseum/deploy/tasks/velocity/models/velocity_policy.pt
```

### 3. Test Sim2Sim

```bash
# List available tasks
pixi run python src/colosseum/play/scripts/play.py --list

# Run velocity task
pixi run python src/colosseum/play/scripts/play.py --task t1_velocity
```

**In the MuJoCo viewer:**
- Robot should spawn at configured height
- Type velocity commands in terminal: `0.5 0 0` (walk forward)
- Observe if policy behaves as expected

### 4. Deploy to Real Robot

```bash
# Copy colosseum to robot
scp -r colosseum robot@<robot-ip>:~/

# SSH to robot
ssh robot@<robot-ip>

# Source ROS 2 environment
source /opt/booster/BoosterRos2Interface/install/setup.bash

# Run deployment
cd ~/colosseum
python src/colosseum/deploy/scripts/deploy.py --task t1_velocity
```

---

## Troubleshooting

### Common Issues

#### 1. Observation Dimension Mismatch

**Error**: `Expected input size X, got Y`

**Solution**: Your `compute_observation()` doesn't match training setup.
- Check training config for observation terms
- Verify joint counts (12 DOF vs 23 DOF?)
- Ensure all observations are included in same order

#### 2. Robot Falling/Unstable

**Possible causes:**
- PD gains too low/high → Tune `joint_stiffness` and `joint_damping`
- Action scale incorrect → Check training `action_scale` parameter
- Joint order mismatch → Verify `sim2real_joint_indexes` mapping
- Observations not normalized correctly

#### 3. Joint Order Issues

**Symptom**: Robot moves but in weird ways

**Debug**:
```python
# In policy.py, add logging
print("Real joint names:", self.robot.cfg.joint_names)
print("Sim joint names:", self.robot.cfg.sim_joint_names)
print("Real2sim map:", self.robot.data.real2sim_joint_indexes)
print("Sim2real map:", self.robot.data.sim2real_joint_indexes)
```

Ensure mapping is bidirectional and correct.

#### 4. Model Loading Fails

**Error**: `No such file or directory: models/policy.pt`

**Solution**: Check paths are relative to policy file location:
```python
model_path = Path(__file__).parent / cfg.checkpoint_path
```

### Debugging Tips

**1. Print observations:**
```python
def compute_observation(self) -> torch.Tensor:
    obs = ...
    print(f"Obs shape: {obs.shape}, min: {obs.min():.3f}, max: {obs.max():.3f}")
    return obs
```

**2. Visualize actions:**
```python
def inference(self) -> torch.Tensor:
    action = self._model(obs)
    print(f"Action: {action[:5].numpy()}")  # First 5 joints
    return ...
```

**3. Check PD control:**
```python
# In MujocoController.ctrl_step()
print(f"Targets: {dof_targets[:3]}")
print(f"Actual: {dof_pos[:3]}")
print(f"Error: {(dof_targets - dof_pos)[:3]}")
```

---

## Next Steps

After successful sim2sim testing:

1. **Fine-tune PD gains** for real hardware (usually higher than simulation)
2. **Add safety checks** (joint limits, torque limits, emergency stop)
3. **Implement state machine** (prepare → stand → walk → stop)
4. **Add telemetry** (log joint states, commands, actions)
5. **Test incrementally** (first static poses, then slow motions, then full speed)

---

## References

- **[SHARED_OBSERVATIONS.md](SHARED_OBSERVATIONS.md)** - Detailed guide for shared observation architecture
- [booster_deploy GitHub](https://github.com/BoosterRobotics/booster_deploy)
- [Booster Assets](https://github.com/BoosterRobotics/booster_assets)
- [Booster Robotics SDK](https://github.com/BoosterRobotics/booster_robotics_sdk)
- mjlab documentation: `pixi run docs`
