# Shared Observations Architecture

This guide explains how to create observation functions that work in both training (mjlab) and deployment (booster_deploy) contexts, eliminating code duplication and ensuring consistency.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Architecture](#solution-architecture)
3. [Implementation Steps](#implementation-steps)
4. [Action Handling](#action-handling)
5. [Testing Strategy](#testing-strategy)
6. [Migration Guide](#migration-guide)

---

## Problem Statement

### Current Approach: Code Duplication

**Training** (mjlab):
```python
# src/colosseum/train/tasks/velocity/mdp/observations.py
def base_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.root_ang_vel_b

def projected_gravity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    gravity_w = torch.tensor([0.0, 0.0, -1.0])
    return quat_rotate_inverse(robot.data.root_quat_w, gravity_w)
```

**Deployment** (booster_deploy):
```python
# src/colosseum/deploy/tasks/velocity/policy.py
def compute_observation(self) -> torch.Tensor:
    # DUPLICATED LOGIC!
    base_ang_vel = self.robot.data.root_ang_vel_b
    gravity_w = torch.tensor([0.0, 0.0, -1.0])
    projected_gravity = quat_rotate_inverse(self.robot.data.root_quat_w, gravity_w)
    # ...
```

### Problems

❌ **Maintenance Burden**: Change observation → update two places
❌ **Error-Prone**: Easy to forget deployment when modifying training
❌ **No Single Source of Truth**: Which implementation is correct?
❌ **Testing Difficulty**: Must test same logic twice

---

## Solution Architecture

### Shared Observation Layer

```
┌─────────────────────────────────────────────────┐
│         Shared Observation Functions            │
│    (src/colosseum/tasks/velocity/observations/) │
│                                                  │
│  - Pure functions operating on data objects     │
│  - No environment dependencies                  │
│  - Work with both Entity.data and RobotData     │
└──────────────┬──────────────────┬───────────────┘
               │                  │
               │                  │
      ┌────────▼────────┐   ┌────▼─────────────┐
      │  Training MDP   │   │ Deployment Policy│
      │  (Wrappers)     │   │  (Direct Calls)  │
      │                 │   │                  │
      │  - Extract      │   │  - Call shared   │
      │    entity       │   │    functions     │
      │  - Call shared  │   │  - Handle joint  │
      │    functions    │   │    mapping       │
      └─────────────────┘   └──────────────────┘
```

### Key Principles

1. **Pure Functions**: Observation logic has no side effects
2. **Minimal Dependencies**: Only torch and math utilities
3. **Flexible Inputs**: Accept generic data objects
4. **Explicit Order**: Observation concatenation order is documented
5. **Reusable**: Same code for training and deployment

---

## Implementation Steps

### Step 1: Create Shared Observation Module

**File**: `src/colosseum/tasks/velocity/__init__.py`

```python
"""Shared definitions for velocity tracking task.

This module provides observation functions and constants that work in both
training (mjlab) and deployment (booster_deploy) contexts.
"""
```

**File**: `src/colosseum/tasks/velocity/observations.py`

```python
"""Shared observation functions for velocity task.

These pure functions work in both training and deployment contexts by
operating on generic data objects (Entity.data or RobotData).
"""

from typing import Optional
import torch
from colosseum.deploy.core.utils.isaaclab import math as lab_math


# ============================================================================
# Observation Functions
# ============================================================================

def compute_velocity_commands(vel_command) -> torch.Tensor:
    """Extract velocity commands as tensor.

    Args:
        vel_command: Either VelocityCommand object (deployment) or
                    torch.Tensor (training)

    Returns:
        Tensor of shape (3,) containing [vx, vy, vyaw]
    """
    if isinstance(vel_command, torch.Tensor):
        # Training context: already a tensor
        return vel_command
    else:
        # Deployment context: convert from VelocityCommand object
        return torch.tensor(
            [
                vel_command.lin_vel_x,
                vel_command.lin_vel_y,
                vel_command.ang_vel_yaw,
            ],
            dtype=torch.float32,
        )


def compute_base_ang_vel(robot_data) -> torch.Tensor:
    """Base angular velocity in base frame.

    Args:
        robot_data: Object with root_ang_vel_b attribute
                   (Entity.data or RobotData)

    Returns:
        Tensor of shape (3,) or (N, 3) containing angular velocity
    """
    return robot_data.root_ang_vel_b


def compute_base_lin_vel(robot_data) -> torch.Tensor:
    """Base linear velocity in base frame.

    Args:
        robot_data: Object with root_lin_vel_b attribute
                   (Entity.data or RobotData)

    Returns:
        Tensor of shape (3,) or (N, 3) containing linear velocity
    """
    return robot_data.root_lin_vel_b


def compute_projected_gravity(robot_data) -> torch.Tensor:
    """Gravity vector projected into base frame.

    This tells the robot which direction is "down" in its own coordinate frame.

    Args:
        robot_data: Object with root_quat_w attribute
                   (Entity.data or RobotData)

    Returns:
        Tensor of shape (3,) or (N, 3) containing projected gravity
    """
    # World frame gravity (pointing down)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)

    # Handle both batched and unbatched
    if robot_data.root_quat_w.dim() == 1:
        # Single instance (deployment)
        return lab_math.quat_rotate_inverse(robot_data.root_quat_w, gravity_w)
    else:
        # Batched (training)
        # Expand gravity to match batch size
        batch_size = robot_data.root_quat_w.shape[0]
        gravity_w = gravity_w.unsqueeze(0).expand(batch_size, -1)
        return lab_math.quat_rotate_inverse(robot_data.root_quat_w, gravity_w)


def compute_joint_pos_rel(
    robot_data,
    default_pos: torch.Tensor,
    joint_map: Optional[list[int]] = None,
) -> torch.Tensor:
    """Joint positions relative to default pose.

    Args:
        robot_data: Object with joint_pos attribute
                   (Entity.data or RobotData)
        default_pos: Default joint positions
        joint_map: Optional index mapping for joint reordering
                  (needed in deployment for real→sim mapping)

    Returns:
        Tensor of shape (num_joints,) or (N, num_joints)
    """
    joint_pos = robot_data.joint_pos

    # Apply joint mapping if provided (deployment only)
    if joint_map is not None:
        joint_pos = joint_pos[joint_map]
        default_pos = default_pos[joint_map]

    return joint_pos - default_pos


def compute_joint_vel(
    robot_data,
    joint_map: Optional[list[int]] = None,
) -> torch.Tensor:
    """Joint velocities.

    Args:
        robot_data: Object with joint_vel attribute
                   (Entity.data or RobotData)
        joint_map: Optional index mapping for joint reordering
                  (needed in deployment for real→sim mapping)

    Returns:
        Tensor of shape (num_joints,) or (N, num_joints)
    """
    joint_vel = robot_data.joint_vel

    # Apply joint mapping if provided (deployment only)
    if joint_map is not None:
        joint_vel = joint_vel[joint_map]

    return joint_vel


def compute_last_action(last_action: torch.Tensor) -> torch.Tensor:
    """Previous action (provides temporal context).

    Args:
        last_action: Previous action tensor

    Returns:
        Tensor of shape (num_joints,) or (N, num_joints)
    """
    return last_action


# ============================================================================
# Observation Configuration
# ============================================================================

# Define observation order (CRITICAL: must match training!)
VELOCITY_OBS_ORDER = [
    "velocity_commands",   # (3,)
    "base_ang_vel",        # (3,)
    "projected_gravity",   # (3,)
    "joint_pos_rel",       # (num_joints,)
    "joint_vel",           # (num_joints,)
    "last_action",         # (num_joints,)
]


def compute_observation_size(num_joints: int) -> int:
    """Calculate total observation size.

    Args:
        num_joints: Number of actuated joints

    Returns:
        Total observation dimension
    """
    return (
        3  # velocity_commands
        + 3  # base_ang_vel
        + 3  # projected_gravity
        + num_joints  # joint_pos_rel
        + num_joints  # joint_vel
        + num_joints  # last_action
    )


# ============================================================================
# Helper Functions
# ============================================================================

def concatenate_observations(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate observations in defined order.

    Args:
        obs_dict: Dictionary mapping observation names to tensors

    Returns:
        Concatenated observation tensor
    """
    # Verify all required observations are present
    missing = set(VELOCITY_OBS_ORDER) - set(obs_dict.keys())
    if missing:
        raise ValueError(f"Missing observations: {missing}")

    # Concatenate in defined order
    obs_list = [obs_dict[key] for key in VELOCITY_OBS_ORDER]
    return torch.cat(obs_list, dim=-1)
```

### Step 2: Create Training MDP Wrappers

**File**: `src/colosseum/train/tasks/velocity/mdp/observations.py`

```python
"""MDP observation term wrappers for velocity task training.

These functions wrap the shared observation functions to work with
mjlab's ManagerBasedRlEnv interface.
"""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.entity import Entity

# Import shared functions
from colosseum.tasks.velocity.observations import (
    compute_base_ang_vel,
    compute_base_lin_vel,
    compute_projected_gravity,
    compute_joint_pos_rel,
    compute_joint_vel,
)


def base_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Base angular velocity observation term.

    Args:
        env: RL environment
        asset_cfg: Asset configuration (specifies robot entity)

    Returns:
        Batched tensor of shape (num_envs, 3)
    """
    robot: Entity = env.scene[asset_cfg.name]
    return compute_base_ang_vel(robot.data)


def base_lin_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Base linear velocity observation term.

    Args:
        env: RL environment
        asset_cfg: Asset configuration (specifies robot entity)

    Returns:
        Batched tensor of shape (num_envs, 3)
    """
    robot: Entity = env.scene[asset_cfg.name]
    return compute_base_lin_vel(robot.data)


def projected_gravity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Projected gravity observation term.

    Args:
        env: RL environment
        asset_cfg: Asset configuration (specifies robot entity)

    Returns:
        Batched tensor of shape (num_envs, 3)
    """
    robot: Entity = env.scene[asset_cfg.name]
    return compute_projected_gravity(robot.data)


def joint_pos_rel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions relative to default observation term.

    Args:
        env: RL environment
        asset_cfg: Asset configuration (specifies robot entity)

    Returns:
        Batched tensor of shape (num_envs, num_joints)
    """
    robot: Entity = env.scene[asset_cfg.name]
    return compute_joint_pos_rel(robot.data, robot.data.default_joint_pos)


def joint_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint velocities observation term.

    Args:
        env: RL environment
        asset_cfg: Asset configuration (specifies robot entity)

    Returns:
        Batched tensor of shape (num_envs, num_joints)
    """
    robot: Entity = env.scene[asset_cfg.name]
    return compute_joint_vel(robot.data)
```

### Step 3: Update Training Configuration

**File**: `src/colosseum/train/tasks/velocity/config/t1/env_cfgs.py`

Update to use the new MDP wrappers:

```python
"""Booster T1 velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import ObservationTermCfg
from mjlab.managers import SceneEntityCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# Import shared observation order
from colosseum.tasks.velocity.observations import VELOCITY_OBS_ORDER

# Import MDP wrappers
from colosseum.tasks.velocity.mdp import observations as obs_mdp

from colosseum.robots.booster_t1.t1_constants import T1_ROBOT_CFG


def booster_t1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    # Use T1 robot config
    cfg.scene.entities = {"robot": T1_ROBOT_CFG}

    # Configure observations using shared functions
    # Order matches VELOCITY_OBS_ORDER
    cfg.observations = {
        "policy": {
            # Note: velocity_commands is provided by velocity task base config
            "base_ang_vel": ObservationTermCfg(
                func=obs_mdp.base_ang_vel,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "projected_gravity": ObservationTermCfg(
                func=obs_mdp.projected_gravity,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "joint_pos_rel": ObservationTermCfg(
                func=obs_mdp.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "joint_vel": ObservationTermCfg(
                func=obs_mdp.joint_vel,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            # Note: last_action is provided by velocity task base config
        }
    }

    # ... rest of config

    return cfg
```

### Step 4: Create Deployment Policy

**File**: `src/colosseum/deploy/tasks/velocity/policy.py`

```python
"""Velocity tracking policy for deployment."""

from pathlib import Path
import torch

from colosseum.deploy.core.controllers import BaseController, Policy, PolicyCfg
from colosseum.deploy.core.utils.isaaclab.configclass import configclass

# Import shared observation functions
from colosseum.tasks.velocity.observations import (
    compute_base_ang_vel,
    compute_projected_gravity,
    compute_joint_pos_rel,
    compute_joint_vel,
    compute_last_action,
    compute_velocity_commands,
    concatenate_observations,
    VELOCITY_OBS_ORDER,
    compute_observation_size,
)


class VelocityPolicy(Policy):
    """Velocity tracking policy for T1 robot deployment.

    This policy uses shared observation functions to ensure consistency
    with training.
    """

    def __init__(self, cfg: "VelocityPolicyCfg", controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot
        self.vel_command = controller.vel_command

        # Load TorchScript model
        model_path = Path(__file__).parent / cfg.checkpoint_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._model: torch.jit.ScriptModule = torch.jit.load(str(model_path))
        self._model.eval()

        # Compute action scale (from training config)
        # Scale = 0.25 * effort_limit / stiffness
        self.action_scale = (
            0.25 * self.robot.effort_limit / self.robot.joint_stiffness
        )

        # Verify observation size
        expected_obs_size = compute_observation_size(self.robot.num_joints)
        print(f"[VelocityPolicy] Expected observation size: {expected_obs_size}")
        print(f"[VelocityPolicy] Observation order: {VELOCITY_OBS_ORDER}")

    def reset(self) -> None:
        """Reset policy state at start of episode."""
        self.last_action = torch.zeros(self.robot.num_joints, dtype=torch.float32)
        print("[VelocityPolicy] Reset complete")

    def compute_observation(self) -> torch.Tensor:
        """Compute observations using shared functions.

        This ensures observations match exactly what the policy saw during training.

        Returns:
            Observation tensor of shape (1, obs_size)
        """
        # Get joint mapping (real robot order → simulation order)
        real2sim_map = self.robot.data.real2sim_joint_indexes

        # Compute each observation using shared functions
        obs_dict = {
            "velocity_commands": compute_velocity_commands(self.vel_command),
            "base_ang_vel": compute_base_ang_vel(self.robot.data),
            "projected_gravity": compute_projected_gravity(self.robot.data),
            "joint_pos_rel": compute_joint_pos_rel(
                self.robot.data, self.robot.default_joint_pos, joint_map=real2sim_map
            ),
            "joint_vel": compute_joint_vel(self.robot.data, joint_map=real2sim_map),
            "last_action": compute_last_action(self.last_action),
        }

        # Concatenate in defined order
        obs = concatenate_observations(obs_dict)

        return obs.reshape(1, -1)

    def inference(self) -> torch.Tensor:
        """Run policy inference and return joint targets.

        Returns:
            Joint position targets in real robot order (num_joints,)
        """
        with torch.no_grad():
            # Compute observations
            obs = self.compute_observation()

            # Run model (returns normalized actions in simulation order)
            action = self._model(obs).flatten()

            # Store for next step
            self.last_action = action

            # Map from simulation order to real robot order
            sim2real_map = self.robot.data.sim2real_joint_indexes

            # Scale actions and add default positions
            joint_targets = (
                action[sim2real_map] * self.action_scale + self.robot.default_joint_pos
            )

            return joint_targets


@configclass
class VelocityPolicyCfg(PolicyCfg):
    """Configuration for velocity tracking policy."""

    constructor = VelocityPolicy
    checkpoint_path: str = "models/velocity_policy.pt"
```

### Step 5: Update Robot Constants

**File**: `src/colosseum/robots/booster_t1/t1_constants.py`

Add deployment configuration:

```python
from colosseum.deploy.core.controllers import RobotCfg, PrepareStateCfg


# ============================================================================
# Deployment Configuration
# ============================================================================

T1_23DOF_DEPLOY_CFG = RobotCfg(
    name="Booster_T1_23DOF",

    # Real robot joint order
    joint_names=[
        "AAHead_yaw",
        "Head_pitch",
        "Left_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch",
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Yaw",
        "Waist",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
    ],

    # Simulation joint order (alphabetical)
    sim_joint_names=[
        "AAHead_yaw",
        "Head_pitch",
        "Left_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Knee_Pitch",
        "Left_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Yaw",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Knee_Pitch",
        "Right_Shoulder_Pitch",
        "Right_Shoulder_Roll",
        "Waist",
    ],

    # Body names (for sensors and observations)
    body_names=[
        "Trunk",
        "H1",
        "H2",
        "AL1",
        "AL2",
        "AL3",
        "left_hand_link",
        "AR1",
        "AR2",
        "AR3",
        "right_hand_link",
        "Waist",
        "Hip_Pitch_Left",
        "Hip_Roll_Left",
        "Hip_Yaw_Left",
        "Shank_Left",
        "Ankle_Cross_Left",
        "left_foot_link",
        "Hip_Pitch_Right",
        "Hip_Roll_Right",
        "Hip_Yaw_Right",
        "Shank_Right",
        "Ankle_Cross_Right",
        "right_foot_link",
    ],

    # PD gains (match training actuator configs)
    joint_stiffness=[
        4.0, 4.0,  # Head
        4.0, 4.0, 4.0, 4.0,  # Left arm
        4.0, 4.0, 4.0, 4.0,  # Right arm
        80.0,  # Waist
        80.0, 80.0, 80.0, 80.0, 30.0, 30.0,  # Left leg
        80.0, 80.0, 80.0, 80.0, 30.0, 30.0,  # Right leg
    ],
    joint_damping=[
        1.0, 1.0,  # Head
        1.0, 1.0, 1.0, 1.0,  # Left arm
        1.0, 1.0, 1.0, 1.0,  # Right arm
        2.0,  # Waist
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0,  # Left leg
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0,  # Right leg
    ],

    # Default pose (standing)
    default_joint_pos=[
        0.0, 0.0,  # Head
        0.0, -1.4, 0.0, 0.0,  # Left arm
        0.0, 1.4, 0.0, 0.0,  # Right arm
        0.0,  # Waist
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Left leg
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Right leg
    ],

    # Effort limits (from motor specs)
    effort_limit=[
        7.0, 7.0,  # Head
        18.0, 18.0, 18.0, 18.0,  # Left arm
        18.0, 18.0, 18.0, 18.0,  # Right arm
        25.0,  # Waist
        45.0, 25.0, 25.0, 60.0, 24.0, 15.0,  # Left leg
        45.0, 25.0, 25.0, 60.0, 24.0, 15.0,  # Right leg
    ],

    # Parallel joint indices (ankles with mechanical coupling)
    parallel_joint_indices=[15, 16, 21, 22],

    # MuJoCo model path
    mjcf_path="{BOOSTER_ASSETS_DIR}/robots/T1/T1_23dof.xml",

    # Prepare state (for real robot initialization)
    prepare_state=PrepareStateCfg(
        stiffness=[
            5.0, 5.0,
            40.0, 50.0, 20.0, 20.0,
            40.0, 50.0, 20.0, 20.0,
            350.0,
            350.0, 350.0, 180.0, 350.0, 150.0, 150.0,
            350.0, 350.0, 180.0, 350.0, 150.0, 150.0,
        ],
        damping=[
            0.1, 0.1,
            0.5, 1.5, 0.2, 0.2,
            0.5, 1.5, 0.2, 0.2,
            5.0,
            7.5, 7.5, 3.0, 5.5, 2.0, 2.0,
            7.5, 7.5, 3.0, 5.5, 2.0, 2.0,
        ],
        joint_pos=[
            0.0, 0.0,
            0.0, -1.4, 0.0, 0.0,
            0.0, 1.4, 0.0, 0.0,
            0.0,
            -0.15, 0.0, 0.0, 0.2, -0.10, 0.0,
            -0.15, 0.0, 0.0, 0.2, -0.10, 0.0,
        ],
    ),
)
```

### Step 6: Register Deployment Task

**File**: `src/colosseum/deploy/tasks/velocity/__init__.py`

```python
"""Velocity task deployment configuration."""

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
    """T1 velocity tracking deployment configuration."""

    # Robot configuration
    robot = T1_23DOF_DEPLOY_CFG

    # Velocity command configuration
    vel_command = VelocityCommandCfg(
        vx_max=1.0,  # Max forward velocity (m/s)
        vy_max=0.5,  # Max lateral velocity (m/s)
        vyaw_max=1.0,  # Max yaw rate (rad/s)
    )

    # Policy configuration
    policy = VelocityPolicyCfg(checkpoint_path="models/velocity_policy.pt")

    # MuJoCo simulation configuration
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.6],  # Spawn position (x, y, z)
        init_quat=[1.0, 0.0, 0.0, 0.0],  # Spawn orientation (w, x, y, z)
        decimation=10,  # Physics steps per policy step (500Hz / 50Hz)
    )


# Register task in deployment registry
register_task("t1_velocity", T1VelocityControllerCfg())
```

---

## Action Handling

### Overview: Actions in Training vs Deployment

While observations are **computed** differently, actions are **processed** differently:

| Aspect | Training | Deployment |
|--------|----------|------------|
| **Policy Output** | Normalized (-1 to 1) | Normalized (-1 to 1) ✅ Same! |
| **Joint Order** | Simulation order | Simulation order ✅ Same! |
| **Scaling** | Automatic (ActionManager) | Manual (in policy) |
| **Final Output** | Environment handles | Absolute joint positions |

### The Action Pipeline

#### Training (mjlab)

```python
# 1. Policy outputs normalized actions in simulation order
action = policy(obs)  # Shape: (num_envs, num_joints), range: [-1, 1]

# 2. ActionManager automatically scales and offsets
# (defined in your training config)
actions = {
    "joint_pos": JointPositionActionCfg(
        asset_name="robot",
        scale=0.25,  # Scale factor
        offset=default_joint_pos,  # Offset (default pose)
    )
}

# 3. Environment applies:
joint_targets = action * scale + offset

# 4. Sends to MuJoCo actuators (PD control at physics frequency)
```

#### Deployment (booster_deploy)

```python
# 1. Policy outputs normalized actions in simulation order (same as training!)
action = self._model(obs).flatten()  # Shape: (num_joints,), range: [-1, 1]

# 2. MANUALLY map from simulation order to real robot order
sim2real_map = self.robot.data.sim2real_joint_indexes
action_real_order = action[sim2real_map]

# 3. MANUALLY scale and offset (replicate ActionManager)
joint_targets = action_real_order * self.action_scale + self.robot.default_joint_pos

# 4. Return absolute joint positions
return joint_targets

# 5. Controller applies PD control to generate torques
```

### Critical: Action Scale Must Match

**The deployment action scale MUST match your training config!**

#### Option 1: Use Explicit Scale (Recommended)

Define scale as a shared constant:

**File**: `src/colosseum/tasks/velocity/__init__.py`

```python
"""Shared velocity task constants."""

# Action scale (must match training config!)
ACTION_SCALE = 0.25
```

**Training config**:
```python
# src/colosseum/train/tasks/velocity/config/t1/env_cfgs.py
from colosseum.tasks.velocity import ACTION_SCALE

def booster_t1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    # Use shared action scale
    cfg.actions["joint_pos"].scale = ACTION_SCALE

    # ...
```

**Deployment policy**:
```python
# src/colosseum/deploy/tasks/velocity/policy.py
from colosseum.tasks.velocity import ACTION_SCALE

class VelocityPolicy(Policy):
    def __init__(self, cfg, controller):
        # ...
        # Use same scale as training
        self.action_scale = ACTION_SCALE
```

#### Option 2: Compute from PD Gains

If your training computes scale from PD gains:

```python
# Training
action_scale = 0.25 * effort_limit / stiffness

# Deployment (must use SAME PD gains!)
self.action_scale = (
    0.25 * self.robot.effort_limit / self.robot.joint_stiffness
)
```

**WARNING**: Only use this if:
1. Your training actually computes scale this way
2. PD gains are identical in training and deployment
3. You understand why this formula makes sense for your task

### Verifying Action Consistency

#### Step 1: Check Training Action Scale

```python
# During training, print actual scale used
print("Training action scale:", env.action_manager.get_term("joint_pos")._scale)
```

#### Step 2: Match in Deployment

```python
# In deployment policy
print("Deployment action scale:", self.action_scale)
# Should match training output!
```

#### Step 3: Test Action Output

Create a simple test:

```python
# test_action_consistency.py
import torch

# Mock normalized action from policy
normalized_action = torch.randn(23) * 0.5  # Range: ~[-1, 1]

# Training-style scaling
training_scale = 0.25
default_pos = torch.zeros(23)
training_output = normalized_action * training_scale + default_pos

# Deployment-style scaling
deployment_scale = 0.25  # Must match!
deployment_output = normalized_action * deployment_scale + default_pos

# Should be identical
assert torch.allclose(training_output, deployment_output)
print("✅ Action scaling matches!")
```

### Complete Deployment Policy Example

Here's the full `inference()` method showing action handling:

```python
class VelocityPolicy(Policy):
    def __init__(self, cfg, controller):
        super().__init__(cfg, controller)
        # ...

        # Action scale - MUST match training!
        from colosseum.tasks.velocity import ACTION_SCALE
        self.action_scale = ACTION_SCALE

        # Store joint mapping indices
        self.sim2real_map = self.robot.data.sim2real_joint_indexes

    def inference(self) -> torch.Tensor:
        """Run policy inference and return joint targets.

        Returns:
            Joint position targets in REAL robot order (num_joints,)
        """
        with torch.no_grad():
            # 1. Compute observations
            obs = self.compute_observation()

            # 2. Run model (normalized actions in simulation order)
            action_normalized = self._model(obs).flatten()

            # 3. Store for next observation
            self.last_action = action_normalized

            # 4. Map from simulation order to real robot order
            action_real_order = action_normalized[self.sim2real_map]

            # 5. Scale and add default positions
            joint_targets = (
                action_real_order * self.action_scale
                + self.robot.default_joint_pos
            )

            return joint_targets
```

### Why Not Share Action Functions?

Unlike observations, actions don't benefit much from shared functions because:

1. **Simple operations**: Just scaling and offset (one line)
2. **Different contexts**: Training uses ActionManager, deployment is manual
3. **Already minimal code**: No complex computation to share
4. **Different final forms**: Training stays normalized, deployment needs absolute positions

However, **sharing the scale constant** is important for consistency!

### Common Action Pitfalls

#### ❌ Pitfall 1: Wrong Action Scale

```python
# Training uses 0.25
cfg.actions["joint_pos"].scale = 0.25

# Deployment uses 0.5 (WRONG!)
self.action_scale = 0.5

# Result: Robot moves twice as much as expected!
```

**Solution**: Use shared constant or verify scales match.

#### ❌ Pitfall 2: Forgetting Joint Order Mapping

```python
# BAD: Returns actions in simulation order
def inference(self):
    action = self._model(obs).flatten()
    return action * self.action_scale + self.robot.default_joint_pos
    # Robot will move incorrectly!
```

**Solution**: Always map `sim → real` before returning.

#### ❌ Pitfall 3: Using Wrong Default Pose

```python
# Training uses custom default pose
cfg.actions["joint_pos"].offset = custom_default_pos

# Deployment uses zeros (WRONG!)
self.robot.default_joint_pos = torch.zeros(23)

# Result: Robot will try to reach wrong base pose!
```

**Solution**: Ensure default pose matches in both contexts.

### Action Checklist

Before deploying, verify:

- [ ] Action scale matches training config
- [ ] Default joint positions match training
- [ ] Joint order mapping is applied (`sim2real_map`)
- [ ] Action output is in real robot order
- [ ] PD gains match training (if using computed scale)

---

## Testing Strategy

### Unit Tests for Observation Functions

**File**: `tests/test_shared_observations.py`

```python
"""Unit tests for shared observation functions."""

import torch
import pytest
from colosseum.tasks.velocity.observations import (
    compute_base_ang_vel,
    compute_projected_gravity,
    compute_joint_pos_rel,
    compute_joint_vel,
    concatenate_observations,
    compute_observation_size,
    VELOCITY_OBS_ORDER,
)


class MockRobotData:
    """Mock robot data for testing."""

    def __init__(self, batch_size=None):
        self.batch_size = batch_size
        if batch_size is None:
            # Single instance (deployment)
            self.root_ang_vel_b = torch.randn(3)
            self.root_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0])
            self.joint_pos = torch.randn(12)
            self.joint_vel = torch.randn(12)
        else:
            # Batched (training)
            self.root_ang_vel_b = torch.randn(batch_size, 3)
            self.root_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(
                batch_size, 1
            )
            self.joint_pos = torch.randn(batch_size, 12)
            self.joint_vel = torch.randn(batch_size, 12)


def test_base_ang_vel_single():
    """Test base angular velocity (single instance)."""
    data = MockRobotData()
    result = compute_base_ang_vel(data)
    assert result.shape == (3,)
    assert torch.allclose(result, data.root_ang_vel_b)


def test_base_ang_vel_batched():
    """Test base angular velocity (batched)."""
    data = MockRobotData(batch_size=4)
    result = compute_base_ang_vel(data)
    assert result.shape == (4, 3)
    assert torch.allclose(result, data.root_ang_vel_b)


def test_projected_gravity_single():
    """Test projected gravity (single instance)."""
    data = MockRobotData()
    result = compute_projected_gravity(data)
    assert result.shape == (3,)
    # With identity quaternion, should point down
    expected = torch.tensor([0.0, 0.0, -1.0])
    assert torch.allclose(result, expected, atol=1e-5)


def test_projected_gravity_batched():
    """Test projected gravity (batched)."""
    data = MockRobotData(batch_size=4)
    result = compute_projected_gravity(data)
    assert result.shape == (4, 3)


def test_joint_pos_rel():
    """Test joint positions relative to default."""
    data = MockRobotData()
    default_pos = torch.zeros(12)
    result = compute_joint_pos_rel(data, default_pos)
    assert result.shape == (12,)
    assert torch.allclose(result, data.joint_pos)


def test_joint_pos_rel_with_mapping():
    """Test joint positions with index mapping."""
    data = MockRobotData()
    default_pos = torch.zeros(12)
    joint_map = list(range(11, -1, -1))  # Reverse order
    result = compute_joint_pos_rel(data, default_pos, joint_map=joint_map)
    assert result.shape == (12,)
    expected = data.joint_pos[joint_map]
    assert torch.allclose(result, expected)


def test_concatenate_observations():
    """Test observation concatenation."""
    obs_dict = {
        "velocity_commands": torch.randn(3),
        "base_ang_vel": torch.randn(3),
        "projected_gravity": torch.randn(3),
        "joint_pos_rel": torch.randn(12),
        "joint_vel": torch.randn(12),
        "last_action": torch.randn(12),
    }
    result = concatenate_observations(obs_dict)
    expected_size = compute_observation_size(12)
    assert result.shape == (expected_size,)


def test_concatenate_observations_missing():
    """Test that missing observations raise error."""
    obs_dict = {
        "velocity_commands": torch.randn(3),
        "base_ang_vel": torch.randn(3),
        # Missing other observations
    }
    with pytest.raises(ValueError, match="Missing observations"):
        concatenate_observations(obs_dict)


def test_observation_size():
    """Test observation size calculation."""
    assert compute_observation_size(12) == 3 + 3 + 3 + 12 + 12 + 12
    assert compute_observation_size(23) == 3 + 3 + 3 + 23 + 23 + 23
```

Run tests:
```bash
pixi run pytest tests/test_shared_observations.py -v
```

### Integration Test: Training vs Deployment

**File**: `tests/test_training_deployment_consistency.py`

```python
"""Test that training and deployment produce same observations."""

import torch
from colosseum.tasks.velocity.observations import (
    compute_base_ang_vel,
    compute_projected_gravity,
    compute_joint_pos_rel,
)


def test_observation_consistency():
    """Verify training and deployment observations match."""

    # Create mock data that works in both contexts
    class MockData:
        def __init__(self):
            self.root_ang_vel_b = torch.tensor([0.1, 0.2, 0.3])
            self.root_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0])
            self.joint_pos = torch.randn(12)

    data = MockData()
    default_pos = torch.zeros(12)

    # Compute observations using shared functions
    ang_vel = compute_base_ang_vel(data)
    gravity = compute_projected_gravity(data)
    joint_pos = compute_joint_pos_rel(data, default_pos)

    # Verify shapes
    assert ang_vel.shape == (3,)
    assert gravity.shape == (3,)
    assert joint_pos.shape == (12,)

    # Verify values are deterministic
    ang_vel2 = compute_base_ang_vel(data)
    assert torch.allclose(ang_vel, ang_vel2)
```

---

## Migration Guide

### For Existing Velocity Task

If you already have a velocity task, follow these steps to migrate:

#### 1. Extract Observation Logic

From your existing training code in `src/colosseum/train/tasks/velocity/mdp/`:

```python
# OLD (before migration)
def base_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.root_ang_vel_b
```

Move the core logic to shared module:

```python
# NEW in src/colosseum/tasks/velocity/observations.py
def compute_base_ang_vel(robot_data) -> torch.Tensor:
    return robot_data.root_ang_vel_b
```

Update training wrapper:

```python
# NEW in src/colosseum/train/tasks/velocity/mdp/observations.py
from colosseum.tasks.velocity.observations import compute_base_ang_vel

def base_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    return compute_base_ang_vel(robot.data)  # Use shared function
```

#### 2. Document Observation Order

Create `VELOCITY_OBS_ORDER` constant matching your training config:

```python
# In src/colosseum/tasks/velocity/observations.py
VELOCITY_OBS_ORDER = [
    # List observation terms in exact order from training config
    "velocity_commands",
    "base_ang_vel",
    # ... etc
]
```

#### 3. Update Training Config

No functional changes needed, but update to use new MDP wrappers:

```python
# In env_cfgs.py
from colosseum.tasks.velocity.mdp import observations as obs_mdp

cfg.observations = {
    "policy": {
        "base_ang_vel": ObservationTermCfg(
            func=obs_mdp.base_ang_vel,  # Uses shared function internally
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        # ... etc
    }
}
```

#### 4. Create Deployment Policy

Use shared functions directly in deployment:

```python
# In src/colosseum/deploy/tasks/velocity/policy.py
from colosseum.tasks.velocity.observations import (
    compute_base_ang_vel,
    concatenate_observations,
)

def compute_observation(self) -> torch.Tensor:
    obs_dict = {
        "base_ang_vel": compute_base_ang_vel(self.robot.data),
        # ... etc
    }
    return concatenate_observations(obs_dict)
```

#### 5. Verify Consistency

Run both training and deployment, compare observations:

```python
# In training, add logging:
print("Training obs:", env.observation_manager.compute()["policy"][0])

# In deployment, add logging:
print("Deployment obs:", self.compute_observation()[0])

# Shapes should match!
```

---

## Benefits Summary

✅ **Single Source of Truth**: Observation logic defined once
✅ **Guaranteed Consistency**: Training and deployment use same code
✅ **Easy Maintenance**: Change once, works everywhere
✅ **Testable**: Unit test observation functions independently
✅ **Clear Documentation**: `VELOCITY_OBS_ORDER` documents observation space
✅ **Type Safety**: Shared functions are strongly typed
✅ **No Duplication**: Eliminate copy-paste errors

---

## Common Pitfalls

### ❌ Pitfall 1: Observation Order Mismatch

**Problem**: Different concatenation order in training vs deployment

**Solution**: Always use `VELOCITY_OBS_ORDER` constant

### ❌ Pitfall 2: Forgetting Joint Mapping

**Problem**: Using real robot order in deployment when model expects sim order

**Solution**: Always pass `joint_map` parameter in deployment

### ❌ Pitfall 3: Batch Dimension Handling

**Problem**: Functions fail when batch dimension is missing/present

**Solution**: Check `tensor.dim()` and handle both cases in shared functions

### ❌ Pitfall 4: Missing Observations

**Problem**: Adding observation to training, forgetting deployment

**Solution**: Use `concatenate_observations()` which validates all terms present

---

## Next Steps

1. ✅ Implement shared observation functions
2. ✅ Create MDP wrappers for training
3. ✅ Update deployment policy to use shared functions
4. ⬜ Write unit tests
5. ⬜ Test training → export → deployment pipeline
6. ⬜ Verify observations match exactly

For complete deployment workflow, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).
