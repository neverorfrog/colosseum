# Observation Specification System

This document explains Colosseum's observation specification system that ensures consistency between training and deployment.

## Table of Contents

1. [Overview](#overview)
2. [The Observation Specification Pattern](#the-observation-specification-pattern)
3. [Implementation Guide](#implementation-guide)
4. [Validation and Debugging](#validation-and-debugging)
5. [Best Practices](#best-practices)

---

## Overview

### The Problem

Traditional RL deployment requires careful manual alignment of observations between training and deployment:

```python
# Training (mjlab)
def compute_observation_training(env):
    return torch.cat([
        env.robot.base_ang_vel,      # Component 1
        env.robot.projected_gravity,  # Component 2
        env.robot.joint_pos,          # Component 3
    ])

# Deployment (manual implementation)
def compute_observation_deployment(robot):
    return torch.cat([
        robot.base_ang_vel,           # Must match training order!
        robot.projected_gravity,      # Easy to make mistakes!
        robot.joint_pos,
    ])
```

**Problems:**
- ❌ Easy to forget a component or change order
- ❌ No validation that training and deployment match
- ❌ Hard to debug size mismatches
- ❌ Documentation lives separately from code

### The Solution: Observation Specification

Instead of sharing implementation, we share a **specification** that defines the observation structure:

```python
# Define specification ONCE
class VelocityObservationSpec(ObservationSpec):
    @property
    def observation_names(self) -> List[str]:
        return ["base_ang_vel", "projected_gravity", "joint_pos"]

    def get_component_size(self, name: str, num_joints: int) -> int:
        if name in ["base_ang_vel", "projected_gravity"]:
            return 3
        if name == "joint_pos":
            return num_joints
        raise ValueError(f"Unknown: {name}")
```

**Benefits:**
- ✅ Single source of truth for observation structure
- ✅ Runtime validation ensures consistency
- ✅ Self-documenting (lists all components and sizes)
- ✅ Easy debugging (can split observations back into components)
- ✅ Training and deployment can implement differently

---

## The Observation Specification Pattern

### Abstract Base Class

All observation specs inherit from `ObservationSpec`:

```python
# deploy/core/observation_spec.py
from abc import ABC, abstractmethod
from typing import List, Dict
import torch
from torch import Tensor

class ObservationSpec(ABC):
    """Abstract base class for task observation specifications."""

    @property
    @abstractmethod
    def observation_names(self) -> List[str]:
        """Ordered list of observation component names.

        This defines the concatenation order for observations.
        """
        pass

    @abstractmethod
    def get_component_size(self, name: str, num_joints: int) -> int:
        """Get the size of a specific observation component.

        Args:
            name: Component name from observation_names
            num_joints: Number of robot joints

        Returns:
            Size of the component
        """
        pass

    def compute_total_size(self, num_joints: int) -> int:
        """Compute total observation size."""
        return sum(
            self.get_component_size(name, num_joints)
            for name in self.observation_names
        )

    def validate_observation(self, obs: Tensor, num_joints: int):
        """Validate observation tensor matches this spec.

        Args:
            obs: Observation tensor (shape: [..., obs_size])
            num_joints: Number of robot joints

        Raises:
            ValueError: If observation size doesn't match spec
        """
        expected_size = self.compute_total_size(num_joints)
        actual_size = obs.shape[-1]

        if actual_size != expected_size:
            raise ValueError(
                f"Observation size mismatch!\n"
                f"  Expected: {expected_size}\n"
                f"  Got: {actual_size}\n"
                f"  Difference: {actual_size - expected_size}\n"
                f"Use describe() to see expected structure."
            )

        # Check for NaN/Inf
        if torch.isnan(obs).any():
            raise ValueError("Observation contains NaN values!")
        if torch.isinf(obs).any():
            raise ValueError("Observation contains Inf values!")

    def split_observation(
        self, obs: Tensor, num_joints: int
    ) -> Dict[str, Tensor]:
        """Split observation into named components.

        Args:
            obs: Observation tensor (shape: [..., obs_size])
            num_joints: Number of robot joints

        Returns:
            Dictionary mapping component names to tensors
        """
        self.validate_observation(obs, num_joints)

        components = {}
        idx = 0
        for name in self.observation_names:
            size = self.get_component_size(name, num_joints)
            components[name] = obs[..., idx:idx+size]
            idx += size

        return components

    def describe(self, num_joints: int) -> str:
        """Get human-readable description of observation structure.

        Args:
            num_joints: Number of robot joints

        Returns:
            Formatted string describing the observation
        """
        lines = ["Observation Structure:"]
        lines.append(f"  Total size: {self.compute_total_size(num_joints)}")
        lines.append(f"  Components:")

        idx = 0
        for name in self.observation_names:
            size = self.get_component_size(name, num_joints)
            lines.append(f"    [{idx:3d}:{idx+size:3d}] {name:20s} (size: {size})")
            idx += size

        return "\n".join(lines)
```

### Task-Specific Implementation

Each task implements its own specification:

```python
# tasks/velocity/mdp/observation_spec.py
from colosseum.deploy.core.observation_spec import ObservationSpec

class VelocityObservationSpec(ObservationSpec):
    """Velocity tracking task observation specification.

    Observation components (in order):
    - velocity_commands: (3,) - desired [vx, vy, vyaw]
    - base_ang_vel: (3,) - base angular velocity in base frame
    - projected_gravity: (3,) - gravity vector in base frame
    - joint_pos_rel: (num_joints,) - joint positions relative to default
    - joint_vel: (num_joints,) - joint velocities
    - last_action: (num_joints,) - previous action
    """

    @property
    def observation_names(self) -> List[str]:
        return [
            "velocity_commands",
            "base_ang_vel",
            "projected_gravity",
            "joint_pos_rel",
            "joint_vel",
            "last_action",
        ]

    def get_component_size(self, name: str, num_joints: int) -> int:
        # Fixed-size components
        if name in ["velocity_commands", "base_ang_vel", "projected_gravity"]:
            return 3

        # Joint-dependent components
        if name in ["joint_pos_rel", "joint_vel", "last_action"]:
            return num_joints

        raise ValueError(f"Unknown observation component: {name}")

# Create global instance
VELOCITY_OBS_SPEC = VelocityObservationSpec()
```

---

## Implementation Guide

### Step 1: Define Observation Specification

Create `tasks/<task>/mdp/observation_spec.py`:

```python
from colosseum.deploy.core.observation_spec import ObservationSpec

class YourTaskObservationSpec(ObservationSpec):
    """Your task observation specification."""

    @property
    def observation_names(self) -> List[str]:
        """List observation components in concatenation order."""
        return [
            "component1",
            "component2",
            # ... add all components
        ]

    def get_component_size(self, name: str, num_joints: int) -> int:
        """Return size for each component."""
        if name == "component1":
            return 3  # Example: fixed size
        if name == "component2":
            return num_joints  # Example: joint-dependent
        raise ValueError(f"Unknown component: {name}")

# Create global instance for easy import
YOUR_TASK_OBS_SPEC = YourTaskObservationSpec()
```

### Step 2: Implement Training Observations

Create simple extraction functions in `tasks/<task>/mdp/observations.py`:

```python
# tasks/<task>/mdp/observations.py
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import SceneEntityCfg
import torch

def component1(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Extract component1 (training)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.some_property

def component2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Extract component2 (training)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.another_property

# Use in training config
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg

cfg.observations = {
    "policy": ObservationGroupCfg(
        terms={
            "component1": ObservationTermCfg(
                func=component1,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "component2": ObservationTermCfg(
                func=component2,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
        },
        concatenate_terms=True,  # IMPORTANT: Must be True!
    )
}
```

### Step 3: Implement Deployment Policy

Compute observations in `tasks/<task>/deploy/<robot>/policy.py`:

```python
# tasks/<task>/deploy/<robot>/policy.py
import torch
from colosseum.deploy.core.controllers import Policy, RobotData
from colosseum.tasks.<task>.mdp.observation_spec import YOUR_TASK_OBS_SPEC

class YourTaskPolicy(Policy):
    def compute_observation(self) -> torch.Tensor:
        """Compute observation following spec."""

        # Compute each component
        component1 = self.compute_component1()  # Your implementation
        component2 = self.compute_component2()  # Your implementation

        # Concatenate in order specified by spec
        obs = torch.cat([
            component1,
            component2,
            # ... all components in spec order
        ])

        # Validate
        YOUR_TASK_OBS_SPEC.validate_observation(obs, self.robot.num_joints)

        return obs
```

### Step 4: Validation (Optional)

Validate that training config matches spec:

```python
# In a test or validation script
from colosseum.tasks.<task>.config.<robot>.env_cfgs import your_env_cfg
from colosseum.tasks.<task>.mdp.observation_spec import YOUR_TASK_OBS_SPEC

# Load training config
cfg = your_env_cfg()

# Check observation structure
print(YOUR_TASK_OBS_SPEC.describe(num_joints=23))

# Verify component names match
training_components = set(cfg.observations["policy"].terms.keys())
spec_components = set(YOUR_TASK_OBS_SPEC.observation_names)

if training_components != spec_components:
    print("WARNING: Training and spec components don't match!")
    print(f"  Missing in training: {spec_components - training_components}")
    print(f"  Extra in training: {training_components - spec_components}")
```

---

## Validation and Debugging

### Runtime Validation

Validation happens automatically in deployment:

```python
# In policy
obs = self.compute_observation()  # Automatically validates

# If there's a size mismatch, you get a helpful error:
# ValueError: Observation size mismatch!
#   Expected: 78
#   Got: 75
#   Difference: -3
#   Use describe() to see expected structure.
```

### Describing Structure

Get a human-readable description:

```python
from colosseum.tasks.velocity.mdp.observation_spec import VELOCITY_OBS_SPEC

print(VELOCITY_OBS_SPEC.describe(num_joints=23))

# Output:
# Observation Structure:
#   Total size: 78
#   Components:
#     [  0:  3] velocity_commands    (size: 3)
#     [  3:  6] base_ang_vel         (size: 3)
#     [  6:  9] projected_gravity    (size: 3)
#     [  9: 32] joint_pos_rel        (size: 23)
#     [ 32: 55] joint_vel            (size: 23)
#     [ 55: 78] last_action          (size: 23)
```

### Splitting Observations

Debug by splitting observations back into components:

```python
# In policy or debug script
obs = policy.compute_observation()

components = VELOCITY_OBS_SPEC.split_observation(obs, num_joints=23)

print("Velocity commands:", components["velocity_commands"])
print("Base ang vel:", components["base_ang_vel"])
print("Projected gravity:", components["projected_gravity"])
print("Joint pos rel:", components["joint_pos_rel"])
print("Joint vel:", components["joint_vel"])
print("Last action:", components["last_action"])
```

### Checking for NaN/Inf

Validation automatically checks for invalid values:

```python
obs = compute_observation()  # Contains NaN

VELOCITY_OBS_SPEC.validate_observation(obs, num_joints=23)
# ValueError: Observation contains NaN values!
```

---

## Best Practices

### 1. Define Spec First

Always define the observation specification before implementing training or deployment:

```python
# 1. Define spec
class MyTaskObservationSpec(ObservationSpec):
    ...

# 2. Implement training
def training_observation_function(env, asset_cfg):
    ...

# 3. Implement deployment
def deployment_observation_computation(self):
    ...
```

### 2. Use Descriptive Component Names

Choose clear, descriptive names for observation components:

```python
# Good
observation_names = [
    "velocity_commands",
    "base_angular_velocity",
    "projected_gravity_vector",
]

# Bad
observation_names = [
    "vel_cmd",
    "ang_vel",
    "grav",
]
```

### 3. Document Component Meanings

Add docstrings explaining what each component represents:

```python
class MyTaskObservationSpec(ObservationSpec):
    """Task observation specification.

    Observation components:
    - velocity_commands: (3,) desired [vx, vy, vyaw] in m/s and rad/s
    - base_ang_vel: (3,) IMU angular velocity in base frame (rad/s)
    - projected_gravity: (3,) gravity vector rotated to base frame
    - joint_pos_rel: (num_joints,) joint positions relative to default (rad)
    - joint_vel: (num_joints,) joint velocities (rad/s)
    - last_action: (num_joints,) previous policy output
    """
```

### 4. Validate During Development

Add validation calls during development to catch errors early:

```python
# In training config
obs = compute_training_observation(env)
MY_TASK_OBS_SPEC.validate_observation(obs, num_joints=robot.num_joints)

# In deployment policy
obs = self.compute_observation()
MY_TASK_OBS_SPEC.validate_observation(obs, self.robot.num_joints)
```

### 5. Use Global Instances

Create a global instance for easy importing:

```python
# tasks/velocity/mdp/observation_spec.py
class VelocityObservationSpec(ObservationSpec):
    ...

# Global instance
VELOCITY_OBS_SPEC = VelocityObservationSpec()

# Usage
from colosseum.tasks.velocity.mdp.observation_spec import VELOCITY_OBS_SPEC
VELOCITY_OBS_SPEC.validate_observation(obs, num_joints)
```

### 6. Keep Specs Simple

Observation specs should only define structure, not computation:

```python
# Good - just structure
class MyObservationSpec(ObservationSpec):
    def get_component_size(self, name, num_joints):
        if name == "gravity":
            return 3
        ...

# Bad - includes computation
class MyObservationSpec(ObservationSpec):
    def compute_gravity(self, quat):  # Don't do this!
        return quat_rotate_inverse(quat, [0, 0, -1])
```

---

## Summary

The observation specification system provides:

1. **Single Source of Truth**: Define observation structure once
2. **Runtime Validation**: Automatic checks ensure consistency
3. **Self-Documentation**: Specs describe what observations contain
4. **Easy Debugging**: Split observations back into components
5. **Flexibility**: Training and deployment implement differently

This pattern ensures training and deployment observations match while allowing each side to implement optimally for its context.
