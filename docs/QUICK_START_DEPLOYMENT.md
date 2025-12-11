# Quick Start: Deployment

This guide shows you how to quickly deploy a trained policy using Colosseum's robot-agnostic deployment system.

## Prerequisites

- Trained policy model checkpoint (`.pt` file)
- Robot configuration (or use existing T1 configs)

## Option 1: Use Pre-Configured Task (Easiest)

```python
from colosseum.deploy import get_task, list_tasks
from colosseum.deploy.core.controllers import MujocoController

# List available tasks
print(list_tasks())
# {'t1_23dof_velocity': <ControllerCfg>, 't1_12dof_velocity': <ControllerCfg>}

# Get configuration
cfg = get_task("t1_23dof_velocity")

# Create and run controller
controller = MujocoController(cfg)
controller.run()
```

## Option 2: Custom Configuration

```python
from colosseum.deploy.core.controllers import (
    ControllerCfg,
    VelocityCommandCfg,
    MujocoControllerCfg,
    MujocoController,
)
from colosseum.robots.booster_t1.deploy import T1_23DOF_DEPLOY_CFG
from colosseum.deploy.tasks.velocity import VelocityPolicyCfg

# Configure task
cfg = ControllerCfg(
    policy_dt=0.02,  # 50Hz policy frequency

    # Robot configuration
    robot=T1_23DOF_DEPLOY_CFG,

    # Velocity commands
    vel_command=VelocityCommandCfg(
        vx_max=1.0,
        vy_max=0.5,
        vyaw_max=1.0,
    ),

    # Policy
    policy=VelocityPolicyCfg(
        checkpoint_path="path/to/your/model.pt",
        action_scale_factor=0.25,  # Must match training!
    ),

    # MuJoCo simulation settings
    mujoco=MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.6],
        init_quat=[1.0, 0.0, 0.0, 0.0],
        decimation=10,  # 500Hz physics / 50Hz policy
    ),
)

# Run
controller = MujocoController(cfg)
controller.run()
```

## Option 3: Swap Robots

```python
# Just change the robot configuration!
from colosseum.robots.booster_t1.deploy import T1_12DOF_DEPLOY_CFG

cfg = ControllerCfg(
    robot=T1_12DOF_DEPLOY_CFG,  # 12-DOF instead of 23-DOF
    policy=VelocityPolicyCfg(checkpoint_path="models/t1_12dof_velocity.pt"),
    # ... rest same
)
```

## Adding a New Robot

### Step 1: Create Robot Configuration

Create `src/colosseum/robots/your_robot/deploy/robot_cfg.py`:

```python
from colosseum.deploy.core.controllers import RobotCfg

YOUR_ROBOT_DEPLOY_CFG = RobotCfg(
    name="YourRobot",

    # Real robot joint order (from hardware interface)
    joint_names=[
        "joint1",
        "joint2",
        # ...
    ],

    # Simulation joint order (alphabetical, from mjlab)
    sim_joint_names=[
        "joint1",
        "joint2",
        # ...
    ],

    # PD gains (must match training!)
    joint_stiffness=[50.0, 50.0, ...],
    joint_damping=[2.0, 2.0, ...],

    # Default standing pose
    default_joint_pos=[0.0, 0.0, ...],

    # Motor effort limits
    effort_limit=[45.0, 45.0, ...],

    # MuJoCo model path
    mjcf_path="path/to/your_robot.xml",
)
```

### Step 2: Create Task Configuration

Create `src/colosseum/robots/your_robot/deploy/__init__.py`:

```python
from colosseum.deploy.core.controllers import ControllerCfg
from colosseum.deploy import register_task
from colosseum.deploy.tasks.velocity import VelocityPolicyCfg
from .robot_cfg import YOUR_ROBOT_DEPLOY_CFG

cfg = ControllerCfg(
    robot=YOUR_ROBOT_DEPLOY_CFG,
    policy=VelocityPolicyCfg(checkpoint_path="models/your_robot_velocity.pt"),
)

register_task("your_robot_velocity", cfg)
```

### Step 3: Use It!

```python
from colosseum.deploy import get_task

cfg = get_task("your_robot_velocity")
# Deploy as usual
```

## Troubleshooting

### "Model not found" Error

```python
# Use absolute path
policy=VelocityPolicyCfg(checkpoint_path="/absolute/path/to/model.pt")

# Or relative to velocity task directory
policy=VelocityPolicyCfg(checkpoint_path="models/velocity.pt")
```

### Joint Order Mismatch

The system automatically handles joint order differences via `real2sim_joint_indexes` and `sim2real_joint_indexes`. Make sure:

1. `joint_names` matches your **hardware** order
2. `sim_joint_names` matches your **training** order (usually alphabetical)

### Action Scale Mismatch

```python
# Must match training config!
policy=VelocityPolicyCfg(
    action_scale_factor=0.25,  # Check your training config
)
```

### Observation Size Mismatch

The policy automatically computes expected observation size:
```
expected_obs_size = 3 (vel_commands)
                  + 3 (base_ang_vel)
                  + 3 (projected_gravity)
                  + num_joints (joint_pos_rel)
                  + num_joints (joint_vel)
                  + num_joints (last_action)
                  = 9 + 3 * num_joints
```

If you get a size mismatch, check:
1. Your trained model's observation space matches this
2. `VELOCITY_OBS_ORDER` in `tasks/velocity/mdp/observations.py`

## Next Steps

- **Custom Policy**: See `deploy/tasks/velocity/policy.py` for example
- **Custom Observations**: See `tasks/velocity/mdp/README.md`
- **Full Guide**: See `docs/SHARED_OBSERVATIONS.md`
