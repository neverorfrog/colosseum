# Deployment Documentation

This directory contains comprehensive documentation for deploying trained policies in Colosseum.

## Quick Links

- **New to deployment?** Start with [QUICK_START.md](QUICK_START.md)
- **Understanding the architecture?** Read [ARCHITECTURE.md](ARCHITECTURE.md)
- **Creating a deployment bundle?** Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Working with observations?** See [SHARED_OBSERVATIONS.md](SHARED_OBSERVATIONS.md)
- **Integrating with booster_deploy?** Check [BOOSTER_DEPLOY_INTEGRATION.md](BOOSTER_DEPLOY_INTEGRATION.md)

## Documentation Overview

### [QUICK_START.md](QUICK_START.md)
**Quick start guide for deploying policies**

Get up and running quickly with:
- Basic usage examples
- File organization
- Adding your own models
- Validation and debugging
- Common troubleshooting

**Start here if:** You just want to deploy a trained policy quickly.

### [ARCHITECTURE.md](ARCHITECTURE.md)
**Comprehensive architecture overview**

Understand the deployment system:
- Core philosophy (training vs deployment)
- Observation specification system
- Task-specific deployment bundles
- Directory structure
- Deployment components
- Comparison with other frameworks

**Read this if:** You want to understand how the deployment system works and why it's designed this way.

### [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
**Step-by-step deployment guide**

Detailed walkthrough of the deployment pipeline:
- Deployment pipeline (train → bundle → sim2sim → robot)
- Observation specification usage
- Creating deployment bundles (robot config, policy, deployment config)
- Testing with sim2sim
- Deploying to real robots
- Advanced topics (custom observations, multiple robots, versioning)

**Use this when:** You're creating a new deployment bundle or deploying to a new robot.

### [SHARED_OBSERVATIONS.md](SHARED_OBSERVATIONS.md)
**Observation specification system**

Deep dive into observation specifications:
- The problem and solution
- ObservationSpec base class
- Implementation guide
- Validation and debugging
- Best practices

**Read this when:** You're defining observations for a new task or debugging observation mismatches.

### [BOOSTER_DEPLOY_INTEGRATION.md](BOOSTER_DEPLOY_INTEGRATION.md)
**Integration with booster_deploy patterns**

Explains integration with existing booster_deploy codebase:
- What was integrated (deployment script, robot configs, controllers)
- Differences from booster_deploy
- Task structure comparison

**Useful if:** You're familiar with booster_deploy and want to understand how Colosseum differs.

## Key Concepts

### Observation Specification
The **observation specification** defines the contract between training and deployment. Instead of sharing implementation code, we share a specification that both sides must follow.

**Example:**
```python
# Define spec ONCE
class VelocityObservationSpec(ObservationSpec):
    @property
    def observation_names(self) -> List[str]:
        return ["velocity_commands", "base_ang_vel", ...]

# Training: extract from environment
def base_ang_vel(env, asset_cfg):
    return env.scene[asset_cfg.name].data.root_ang_vel_b

# Deployment: compute from sensors
def compute_observation(self):
    obs = torch.cat([vel_cmd, self.robot.data.root_ang_vel_b, ...])
    VELOCITY_OBS_SPEC.validate_observation(obs, self.robot.num_joints)
    return obs
```

### Task-Specific Deployment Bundles
Deployment bundles live in `tasks/<task>/deploy/<robot>/` and contain everything needed for deployment:

```
tasks/velocity/deploy/t1_23dof/
├── robot_cfg.py    # Robot hardware configuration
├── policy.py       # Policy implementation
├── config.py       # Deployment configuration
└── models/         # Model checkpoints
```

This makes the coupling between robot and policy **explicit** and keeps everything organized.

### Automatic Joint Mapping
The system automatically handles joint order differences between simulation (alphabetical) and hardware (specific order):

```python
# Observations → simulation order (for model input)
obs = compute_joint_pos(data, joint_map=robot.data.real2sim_indexes)

# Actions → real order (for hardware output)
targets = action[robot.data.sim2real_indexes] * scale + default_pos
```

## Common Workflows

### Deploying a Trained Policy

1. **Use pre-configured deployment**:
```python
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY_DEPLOY_CFG
from colosseum.deploy.core.controllers import MujocoController

controller = MujocoController(T1_23DOF_VELOCITY_DEPLOY_CFG)
controller.run()
```

2. **Or create custom configuration**:
```python
from colosseum.tasks.velocity.deploy.t1_23dof import create_deployment_cfg

cfg = create_deployment_cfg(
    checkpoint_path="models/my_model.pt",
    vx_max=1.5,
    policy_dt=0.02,
)
controller = MujocoController(cfg)
controller.run()
```

### Creating a New Deployment Bundle

1. **Define observation specification** (`tasks/<task>/mdp/observation_spec.py`)
2. **Implement training observations** (`tasks/<task>/mdp/observations.py`)
3. **Create deployment bundle** (`tasks/<task>/deploy/<robot>/`)
   - `robot_cfg.py` - Robot configuration
   - `policy.py` - Policy implementation
   - `config.py` - Deployment configuration
4. **Test in sim2sim** (MujocoController)
5. **Deploy to robot** (RobotController)

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed steps.

### Debugging Observation Mismatches

1. **Check expected structure**:
```python
from colosseum.tasks.velocity.mdp.observation_spec import VELOCITY_OBS_SPEC
print(VELOCITY_OBS_SPEC.describe(num_joints=23))
```

2. **Split observations to inspect components**:
```python
components = VELOCITY_OBS_SPEC.split_observation(obs, num_joints=23)
for name, value in components.items():
    print(f"{name}: {value}")
```

3. **Validate during development**:
```python
obs = policy.compute_observation()
VELOCITY_OBS_SPEC.validate_observation(obs, num_joints=23)
# Raises ValueError with helpful message if mismatch
```

## Design Philosophy

Colosseum's deployment architecture is based on these principles:

1. **Accept Reality**: Training and deployment are fundamentally different operations. Don't force them to share implementation code.

2. **Share Contracts, Not Implementation**: Use observation specifications to define what must match, but let each side implement optimally.

3. **Explicit Coupling**: Bundle robot configurations with policies (task-specific deployment bundles) rather than trying to maintain robot-agnostic abstractions.

4. **Runtime Validation**: Validate observations at runtime to catch mismatches early.

5. **Keep it Simple**: Avoid artificial abstractions. The code should reflect the reality of RL deployment.

## Getting Help

- **Questions about architecture?** See [ARCHITECTURE.md](ARCHITECTURE.md)
- **How to create observations?** See [SHARED_OBSERVATIONS.md](SHARED_OBSERVATIONS.md)
- **Step-by-step guide?** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Quick example?** See [QUICK_START.md](QUICK_START.md)

## Related Documentation

- **Project Overview**: [../CLAUDE.md](../CLAUDE.md) - Main project documentation
- **Training**: See training task configs in `src/colosseum/tasks/*/config/`
- **Controllers**: See `src/colosseum/deploy/core/controllers/`
