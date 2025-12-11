## Velocity Task Deployment

This directory contains task-specific deployment implementations for velocity tracking on different robots.

### Structure

```
deploy/
├── t1_23dof/              # T1 full body deployment
│   ├── robot_cfg.py       # Robot config (PD gains, joint order, etc.)
│   ├── policy.py          # Policy implementation
│   ├── config.py          # Deployment config
│   └── models/            # Model checkpoints
│       └── velocity_v1.pt
└── README.md
```

Each robot deployment is **self-contained** - robot config, policy, and trained model are bundled together to ensure compatibility.

### Usage

#### Option 1: Use Pre-Configured Bundle

```python
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY_DEPLOY_CFG
from colosseum.deploy.core.controllers import MujocoController

controller = MujocoController(T1_23DOF_VELOCITY_DEPLOY_CFG)
controller.run()
```

#### Option 2: Custom Configuration

```python
from colosseum.tasks.velocity.deploy.t1_23dof import create_deployment_cfg

cfg = create_deployment_cfg(
    checkpoint_path="models/my_model.pt",
    vx_max=1.5,
    vy_max=0.8,
    vyaw_max=1.2,
)

controller = MujocoController(cfg)
controller.run()
```

### Adding a New Robot

To deploy velocity tracking on a new robot:

1. **Create robot directory**: `velocity/deploy/<robot_name>/`

2. **Create robot_cfg.py**:
```python
from colosseum.deploy.core.controllers import RobotCfg

YOUR_ROBOT_VELOCITY_CFG = RobotCfg(
    name="YourRobot_Velocity",
    joint_names=[...],      # Real robot order
    sim_joint_names=[...],  # Simulation order (from training)
    joint_stiffness=[...],  # Must match training!
    # ... etc
)
```

3. **Create policy.py**:
```python
from colosseum.deploy.core.controllers import Policy
from colosseum.tasks.velocity.mdp.observation_spec import VELOCITY_OBS_SPEC

class YourRobotVelocityPolicy(Policy):
    def compute_observation(self):
        # Compute observations following VELOCITY_OBS_SPEC
        obs = torch.cat([
            vel_commands,
            base_ang_vel,
            projected_gravity,
            joint_pos_rel,
            joint_vel,
            last_action,
        ])
        VELOCITY_OBS_SPEC.validate_observation(obs, self.robot.num_joints)
        return obs
```

4. **Create config.py**:
```python
from colosseum.deploy.core.controllers import ControllerCfg

def create_deployment_cfg(checkpoint_path: str) -> ControllerCfg:
    return ControllerCfg(
        robot=YOUR_ROBOT_VELOCITY_CFG,
        policy=YourRobotVelocityPolicyCfg(checkpoint_path),
        # ... etc
    )
```

### Key Features

#### Observation Contract

All velocity policies must follow `VelocityObservationSpec`:

```python
from colosseum.tasks.velocity.mdp.observation_spec import VELOCITY_OBS_SPEC

# Get expected size
obs_size = VELOCITY_OBS_SPEC.compute_size(num_joints=23)  # 78

# Validate observation
VELOCITY_OBS_SPEC.validate_observation(obs, num_joints=23)

# Debug observation
components = VELOCITY_OBS_SPEC.split_observation(obs, num_joints=23)
print(components["velocity_commands"])
print(components["joint_pos_rel"])
```

#### Automatic Joint Mapping

The robot config handles joint order differences:
- `joint_names`: Real robot order (hardware interface)
- `sim_joint_names`: Simulation order (training)

The policy automatically maps between them:
```python
# Observations → simulation order (for model input)
joint_pos_sim = joint_pos[robot.data.real2sim_joint_indexes]

# Actions → real order (for hardware output)
targets_real = action[robot.data.sim2real_joint_indexes]
```

### Differences from Training

**Training** (`tasks/velocity/mdp/observations.py`):
- Extracts from pre-computed `Entity.data`
- Batched across thousands of environments
- Uses mjlab infrastructure

**Deployment** (`tasks/velocity/deploy/<robot>/policy.py`):
- Computes from sensor data
- Single instance (real robot or single sim)
- Standalone policy

**Shared Contract** (`tasks/velocity/mdp/observation_spec.py`):
- Defines observation structure
- Both training and deployment follow this
- Ensures consistency
