# Velocity Task MDP Functions

This directory contains MDP functions for the velocity tracking task, organized following the three-layer architecture.

## Files

- `observations.py`: **Pure observation functions** (shared between training and deployment)
- `wrappers.py`: **Training wrappers** (mjlab-specific, NOT used in deployment)

## Usage in Training

When configuring your training environment, import wrappers from `wrappers.py`:

```python
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import ObservationTermCfg, SceneEntityCfg

# Import training wrappers (NOT pure functions!)
from colosseum.tasks.velocity.mdp import wrappers

# Configure observations using wrappers
cfg.observations = {
    "policy": {
        "base_ang_vel": ObservationTermCfg(
            func=wrappers.base_ang_vel,  # Use wrapper for training
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "projected_gravity": ObservationTermCfg(
            func=wrappers.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "joint_pos_rel": ObservationTermCfg(
            func=wrappers.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "joint_vel": ObservationTermCfg(
            func=wrappers.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        # ... etc
    }
}
```

## Usage in Deployment

When writing deployment policies, import pure functions from `observations.py`:

```python
# Import pure functions (NOT wrappers!)
from colosseum.tasks.velocity.mdp.observations import (
    compute_base_ang_vel,
    compute_projected_gravity,
    compute_joint_pos_rel,
    concatenate_observations,
)

class VelocityPolicy(Policy):
    def compute_observation(self):
        obs_dict = {
            "base_ang_vel": compute_base_ang_vel(self.robot.data),
            "projected_gravity": compute_projected_gravity(self.robot.data),
            "joint_pos_rel": compute_joint_pos_rel(
                self.robot.data,
                self.robot.default_joint_pos,
                joint_map=self.robot.data.real2sim_joint_indexes
            ),
            # ... etc
        }
        return concatenate_observations(obs_dict)
```

## Why Two Modules?

### `observations.py` - Pure Functions (Shared)

- **No environment dependencies**: Works with generic `robot_data` objects
- **Used by both**: Training wrappers AND deployment policies
- **Single source of truth**: Ensures observations match exactly

### `wrappers.py` - Training Adapters (Training-Only)

- **mjlab interface**: Wraps pure functions for `ManagerBasedRlEnv`
- **Used only in training**: Deployment doesn't need these
- **Thin adapters**: Extract entity, call pure function

## Adding New Observations

1. **Add pure function** to `observations.py`:

```python
def compute_my_observation(robot_data) -> torch.Tensor:
    """New observation (works in training AND deployment)."""
    return robot_data.some_sensor_data
```

2. **Add wrapper** to `wrappers.py`:

```python
def my_observation(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """MDP wrapper for training."""
    robot = env.scene[asset_cfg.name]
    return compute_my_observation(robot.data)
```

3. **Update observation order** in `observations.py`:

```python
VELOCITY_OBS_ORDER = [
    "velocity_commands",
    "base_ang_vel",
    "my_observation",  # Add here!
    # ... etc
]
```

4. **Use wrapper in training config**:

```python
cfg.observations["policy"]["my_observation"] = ObservationTermCfg(
    func=wrappers.my_observation,
    params={"asset_cfg": SceneEntityCfg("robot")},
)
```

5. **Use pure function in deployment**:

```python
obs_dict["my_observation"] = compute_my_observation(self.robot.data)
```

## Benefits

✅ **Consistency**: Training and deployment use identical observation logic
✅ **Maintainability**: Change observation logic in ONE place
✅ **Testability**: Pure functions are easy to unit test
✅ **Clarity**: Clear separation between computation and environment integration
