# Quick Start: Deployment V2

## Using Pre-Configured Deployment

```python
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY_DEPLOY_CFG
from colosseum.deploy.core.controllers import MujocoController

# Create controller with pre-configured bundle
controller = MujocoController(T1_23DOF_VELOCITY_DEPLOY_CFG)

# Run deployment
controller.run()
```

## Custom Configuration

```python
from colosseum.tasks.velocity.deploy.t1_23dof import create_deployment_cfg

# Create custom config
cfg = create_deployment_cfg(
    checkpoint_path="models/my_model_v2.pt",
    vx_max=1.5,      # Max forward velocity
    vy_max=0.8,      # Max lateral velocity
    vyaw_max=1.2,    # Max yaw rate
    policy_dt=0.02,  # 50Hz policy
)

# Run
controller = MujocoController(cfg)
controller.run()
```

## File Locations

**Training:**
- Config: `src/colosseum/tasks/velocity/config/t1/env_cfgs.py`
- MDP functions: `src/colosseum/tasks/velocity/mdp/observations.py`
- Observation spec: `src/colosseum/tasks/velocity/mdp/observation_spec.py`

**Deployment:**
- Everything in: `src/colosseum/tasks/velocity/deploy/t1_23dof/`
  - `robot_cfg.py` - Robot configuration
  - `policy.py` - Policy implementation
  - `config.py` - Deployment configuration
  - `models/` - Model checkpoints

## Adding Your Own Model

1. Train your model using velocity task configs
2. Place checkpoint in `tasks/velocity/deploy/t1_23dof/models/`
3. Update config:

```python
cfg = create_deployment_cfg(
    checkpoint_path="models/your_model.pt",
)
```

## Validation

The observation spec ensures consistency:

```python
from colosseum.tasks.velocity.mdp.observation_spec import VELOCITY_OBS_SPEC

# Check expected size
obs_size = VELOCITY_OBS_SPEC.compute_size(num_joints=23)
print(f"Expected observation size: {obs_size}")  # 78

# Describe structure
print(VELOCITY_OBS_SPEC.describe(num_joints=23))

# Validate at runtime (done automatically by policy)
VELOCITY_OBS_SPEC.validate_observation(obs, num_joints=23)
```

## Debugging

Split observations to inspect components:

```python
components = VELOCITY_OBS_SPEC.split_observation(obs, num_joints=23)

print("Velocity commands:", components["velocity_commands"])
print("Base ang vel:", components["base_ang_vel"])
print("Projected gravity:", components["projected_gravity"])
print("Joint pos rel:", components["joint_pos_rel"])
print("Joint vel:", components["joint_vel"])
print("Last action:", components["last_action"])
```

## Next Steps

- **New robot?** Create `tasks/velocity/deploy/<robot>/`
- **New task?** Create `tasks/<task>/deploy/<robot>/`
- **See also**: `docs/ARCHITECTURE_V2.md` for design philosophy
