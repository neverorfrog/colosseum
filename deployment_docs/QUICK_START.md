# Quick Start: Deploying Policies

This guide shows you how to quickly deploy a trained policy using Colosseum's deployment system.

## Prerequisites

- Trained policy model checkpoint (`.pt` file)
- Robot configuration (or use existing T1 configs)

## Basic Usage

### Using a Pre-Configured Deployment

The simplest way to deploy is using a pre-configured task bundle:

```python
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY_DEPLOY_CFG
from colosseum.deploy.core.controllers import MujocoController

# Create controller with pre-configured bundle
controller = MujocoController(T1_23DOF_VELOCITY_DEPLOY_CFG)

# Run deployment
controller.run()
```

### Custom Configuration

Create a custom deployment configuration:

```python
from colosseum.tasks.velocity.deploy.t1_23dof import create_deployment_cfg
from colosseum.deploy.core.controllers import MujocoController

# Create custom config
cfg = create_deployment_cfg(
    checkpoint_path="models/my_model_v2.pt",
    vx_max=1.5,      # Max forward velocity
    vy_max=0.8,      # Max lateral velocity
    vyaw_max=1.2,    # Max yaw rate
    policy_dt=0.02,  # 50Hz policy frequency
)

# Run deployment
controller = MujocoController(cfg)
controller.run()
```

## File Organization

**Training Files:**
```
src/colosseum/tasks/velocity/
├── config/t1/env_cfgs.py          # Training environment config
├── mdp/
│   ├── observation_spec.py        # Observation contract
│   └── observations.py            # Training observation functions
└── rl/                            # RL algorithm configs
```

**Deployment Files:**
```
src/colosseum/tasks/velocity/deploy/t1_23dof/
├── robot_cfg.py                   # Robot configuration
├── policy.py                      # Policy implementation
├── config.py                      # Deployment configuration
└── models/                        # Model checkpoints
    └── velocity_v1.pt
```

## Adding Your Own Model

1. **Train your model** using the velocity task configs
2. **Place checkpoint** in `tasks/velocity/deploy/t1_23dof/models/`
3. **Update config**:

```python
cfg = create_deployment_cfg(
    checkpoint_path="models/your_model.pt",
)
```

## Validation

The observation specification automatically ensures consistency between training and deployment:

```python
from colosseum.tasks.velocity.mdp.observation_spec import VELOCITY_OBS_SPEC

# Check expected observation size
obs_size = VELOCITY_OBS_SPEC.compute_total_size(num_joints=23)
print(f"Expected observation size: {obs_size}")  # 78

# Describe structure
print(VELOCITY_OBS_SPEC.describe(num_joints=23))

# Runtime validation (done automatically by policy)
VELOCITY_OBS_SPEC.validate_observation(obs, num_joints=23)
```

## Debugging

Split observations to inspect individual components:

```python
components = VELOCITY_OBS_SPEC.split_observation(obs, num_joints=23)

print("Velocity commands:", components["velocity_commands"])
print("Base ang vel:", components["base_ang_vel"])
print("Projected gravity:", components["projected_gravity"])
print("Joint pos rel:", components["joint_pos_rel"])
print("Joint vel:", components["joint_vel"])
print("Last action:", components["last_action"])
```

## Common Tasks

### Deploy on Different Robot

To deploy the same task on a different robot, create a new deployment bundle:

```
tasks/velocity/deploy/<new_robot>/
├── robot_cfg.py                   # New robot configuration
├── policy.py                      # Policy (may reuse from t1_23dof)
├── config.py                      # Deployment config
└── models/                        # Model checkpoints
```

### Create New Task

To create a completely new task:

1. **Create task structure**:
```
tasks/<new_task>/
├── config/                        # Training configs
├── mdp/
│   ├── observation_spec.py        # Define observation contract
│   └── observations.py            # Training functions
├── rl/                            # RL algorithm configs
└── deploy/<robot>/                # Deployment bundle
    ├── robot_cfg.py
    ├── policy.py
    ├── config.py
    └── models/
```

2. **Define observation specification** (see [ARCHITECTURE.md](ARCHITECTURE.md#observation-specification-system))

3. **Implement policy** following the observation spec

4. **Train and deploy**

## Troubleshooting

### Model Not Found

Use absolute path or path relative to the deployment bundle:

```python
policy=VelocityPolicyCfg(checkpoint_path="/absolute/path/to/model.pt")
# or
policy=VelocityPolicyCfg(checkpoint_path="models/velocity.pt")
```

### Observation Size Mismatch

Check that your trained model's observation space matches the specification:

```python
expected_size = VELOCITY_OBS_SPEC.compute_total_size(num_joints=23)
print(f"Expected: {expected_size}")
print(f"Model input size: {your_model_input_size}")
```

### Joint Order Issues

The system automatically handles joint order differences via `real2sim_indexes` and `sim2real_indexes`. Ensure:

1. `joint_names` matches your **hardware** order
2. `sim_joint_names` matches your **training** order (usually alphabetical)

### Action Scale Mismatch

The action scale must match your training configuration:

```python
policy=VelocityPolicyCfg(
    action_scale_factor=0.25,  # Check your training config!
)
```

## Next Steps

- **Architecture Details**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Deployment Guide**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Integration Guide**: See [BOOSTER_DEPLOY_INTEGRATION.md](BOOSTER_DEPLOY_INTEGRATION.md)
