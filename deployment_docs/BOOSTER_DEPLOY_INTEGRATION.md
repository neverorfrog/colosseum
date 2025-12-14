# Booster Deploy Integration

This document explains how Colosseum integrates patterns from `booster_deploy` while maintaining the V2 architecture.

## What Was Integrated from booster_deploy

### 1. Deployment Script (`scripts/deploy.py`)

**From:** `booster_deploy/scripts/deploy.py`
**To:** `colosseum/scripts/deploy.py`

**Features:**
- ✅ Task discovery and listing (`--list`)
- ✅ MuJoCo simulation support (`--mujoco`)
- ✅ Real robot deployment (default)
- ✅ Webots simulation support (`--webots`)
- ✅ Network interface configuration (`--net`)

**Colosseum Enhancements:**
- Auto-discovers tasks from `tasks/*/deploy/*/` structure
- No manual task registration needed
- Follows V2 task-specific deployment pattern

**Usage:**
```bash
# List available tasks
python scripts/deploy.py --list

# Run in MuJoCo
python scripts/deploy.py --task velocity_t1_23dof --mujoco

# Run on real robot
python scripts/deploy.py --task velocity_t1_23dof --net 192.168.123.161
```

### 2. Robot Configurations

**From:** `booster_deploy/robots/booster.py`
**To:** `tasks/<task>/deploy/<robot>/robot_cfg.py`

**Key Fields (Matched):**
- ✅ `joint_names` - Real robot joint order
- ✅ `sim_joint_names` - Simulation joint order (from booster_deploy)
- ✅ `body_names` / `sim_body_names` - Body names and orders
- ✅ `joint_stiffness` / `joint_damping` - PD gains (from booster_deploy)
- ✅ `default_joint_pos` - Default poses (from booster_deploy)
- ✅ `effort_limit` - Motor limits (from booster_deploy)
- ✅ `parallel_joint_indices` - Mechanically coupled joints
- ✅ `prepare_state` - Preparation pose/gains for real robot (from booster_deploy)

**Differences:**
- **V2 Location**: Configs are in `tasks/<task>/deploy/<robot>/robot_cfg.py`
- **V2 Naming**: `T1_23DOF_VELOCITY_ROBOT_CFG` (task-specific)
- **booster_deploy Naming**: `T1_23DOF_CFG` (global)

### 3. Controller Infrastructure

**From:** `booster_deploy/controllers/`
**To:** `colosseum/deploy/core/controllers/`

**Ported:**
- ✅ `base_controller.py` - BaseController, Policy, RobotData
- ✅ `controller_cfg.py` - ControllerCfg, RobotCfg, PolicyCfg
- ✅ `mujoco_controller.py` - MujocoController for sim2sim
- ✅ `booster_robot_controller.py` - BoosterRobotPortal for real robot
  - ROS 2 integration (/low_state, /joint_ctrl topics)
  - Booster SDK integration
  - Multi-process architecture (sensor thread + inference process)
  - State machine (IDLE → CUSTOM → RUNNING)
  - Safety checks and prepare state

### 4. Task Structure

**booster_deploy Pattern:**
```
tasks/beyond_mimic/
├── __init__.py              # Register task
├── beyond_mimic.py          # Policy + ControllerCfg
├── models/                  # Model checkpoints
└── motions/                 # Motion data
```

**Colosseum V2 Pattern:**
```
tasks/velocity/deploy/t1_23dof/
├── __init__.py              # Export bundle
├── robot_cfg.py             # Robot config
├── policy.py                # Policy implementation
├── config.py                # ControllerCfg
└── models/                  # Model checkpoints
```

**Key Differences:**
- **V2**: Robot config bundled with task
- **V2**: Explicit separation (robot_cfg, policy, config)
- **booster_deploy**: Robot config global, task imports it

## What's Still Missing

### 1. Example Beyond Mimic Task

**Could be ported as:**
```
tasks/beyond_mimic/deploy/k1_22dof/
├── __init__.py
├── robot_cfg.py             # K1 22-DOF config
├── policy.py                # BeyondMimicPolicy
├── config.py                # Various ControllerCfgs (mj2, fight, etc.)
├── models/
│   ├── k1_mj_dance_002.pt
│   └── k1_fight_001.pt
└── motions/
    ├── k1_mj2_seg1.npz
    └── k1_fight_final_deploy.npz
```

### 2. MotionLoader Utility

**File:** `deploy/core/utils/motion_loader.py`

Already present in colosseum! ✅

### 3. Additional Utilities

From booster_deploy that may be useful:
- `utils/metrics.py` - Performance metrics tracking ✅ (already in colosseum)
- `utils/synced_array.py` - Shared memory arrays ✅ (already in colosseum)
- `utils/remote_control_service.py` - Remote control interface ✅ (already in colosseum)

## Integration Checklist

### Completed ✅
- [x] Deployment script (`scripts/deploy.py`)
- [x] Robot config structure (matched booster_deploy)
- [x] Base controller infrastructure
- [x] MujocoController for sim2sim
- [x] BoosterRobotPortal for real robot deployment
- [x] BoosterRobotController for policy inference
- [x] Task discovery system
- [x] Observation specification contract
- [x] All utility infrastructure (synced_array, metrics, remote_control_service)

### TODO ❌
- [ ] Test real robot deployment
- [ ] Port example task (beyond_mimic or create velocity example)
- [ ] Add ROS 2 dependencies to requirements
- [ ] Document real robot setup process

## Key Design Decisions

### Why Task-Specific Robot Configs?

**booster_deploy** approach:
```python
# Global robot config
from booster_deploy.robots.booster import K1_CFG

# Task imports it
cfg = ControllerCfg(robot=K1_CFG, ...)
```

**Colosseum V2** approach:
```python
# Task-specific robot config
# tasks/velocity/deploy/t1_23dof/robot_cfg.py
T1_23DOF_VELOCITY_ROBOT_CFG = RobotCfg(...)

# Bundled in task
cfg = ControllerCfg(robot=T1_23DOF_VELOCITY_ROBOT_CFG, ...)
```

**Rationale:**
1. Different tasks may need different PD gains for the same robot
2. Explicit coupling (robot config matched to policy)
3. Easy to version (whole folder is one deployable unit)
4. If configs are identical across tasks, can still share via import

### How to Share Robot Configs

If you want to share a base robot config:

```python
# robots/booster_t1/deploy_base_cfg.py
BASE_T1_23DOF_CFG = RobotCfg(...)  # Shared base

# tasks/velocity/deploy/t1_23dof/robot_cfg.py
from colosseum.robots.booster_t1.deploy_base_cfg import BASE_T1_23DOF_CFG
import copy

T1_23DOF_VELOCITY_ROBOT_CFG = copy.deepcopy(BASE_T1_23DOF_CFG)
T1_23DOF_VELOCITY_ROBOT_CFG.joint_stiffness = [...]  # Task-specific override
```

## Real Robot Deployment Workflow

### Prerequisites

1. **Booster Firmware** >= v1.4
2. **Booster Robotics SDK** (Python bindings)
3. **ROS 2 Humble** (already on robot)
4. **Python dependencies**: `pip install -r requirements.txt`

### Steps

1. **Train policy** using Colosseum training configs
2. **Export model** as ONNX and copy it to `tasks/<task>/deploy/<robot>/models/policy.onnx` *(run `pixi run export-velocity-onnx -- --task-id Velocity-Flat-Booster-T1 --checkpoint logs/rsl_rl/t1_velocity/<run>/model_28500.pt --output-dir tasks/velocity/deploy/t1_23dof/models` to automate this, or rename the `<run>.onnx` file that is already saved beside every checkpoint under `logs/rsl_rl/...` / `wandb/run-*`; TorchScript/raw PyTorch checkpoints remain supported for advanced debugging)*
3. **Test in MuJoCo**:
   ```bash
   python scripts/deploy.py --task velocity_t1_23dof --mujoco
   ```
4. **Copy to robot**:
   ```bash
   scp -r colosseum/ robot@192.168.123.161:~/
   ```
5. **SSH to robot and run**:
   ```bash
   ssh robot@192.168.123.161
   source /opt/booster/BoosterRos2Interface/install/setup.bash
   cd colosseum
   python scripts/deploy.py --task velocity_t1_23dof
   ```

> ℹ️ Deployment policies now auto-detect `.onnx` files and run them through ONNX Runtime. Keep the exported model alongside the task (e.g., `tasks/velocity/deploy/t1_23dof/models/policy.onnx`) so the bundle stays intact when syncing to the robot. TorchScript and raw training checkpoints are still supported as fallbacks.

## Comparison: booster_deploy vs Colosseum V2

| Aspect | booster_deploy | Colosseum V2 |
|--------|----------------|--------------|
| **Robot Configs** | Global (`robots/booster.py`) | Task-specific (`tasks/*/deploy/*/robot_cfg.py`) |
| **Task Structure** | Single file (policy + cfg) | Separated (robot_cfg, policy, config) |
| **Task Discovery** | Manual registration | Auto-discovery from filesystem |
| **Observation Spec** | None (implicit) | Explicit contract with validation |
| **Deployment Script** | Manual imports | Auto-discovery |
| **Training Integration** | Separate repo | Same repo (tasks/ folder) |

## Benefits of Colosseum V2 Integration

1. **Unified Repo**: Training and deployment in one place
2. **Explicit Contracts**: ObservationSpec ensures consistency
3. **Auto-Discovery**: No manual task registration
4. **Task-Specific**: Robot configs bundled with policies
5. **Validation**: Runtime checks catch errors early
6. **Extensible**: Easy to add new tasks/robots

## Next Steps

1. **Test real robot deployment** on physical T1 robot
2. **Port example task** (beyond_mimic or similar) for demonstration
3. **Add ROS 2 dependencies** to requirements.txt
4. **Document hardware setup** process in detail
5. **Add CI/CD** for deployment testing

## Summary

The integration is **functionally complete**! All core booster_deploy patterns have been ported:
- ✅ Deployment script with auto-discovery
- ✅ Robot controller infrastructure (BoosterRobotPortal/Controller)
- ✅ MuJoCo simulation support
- ✅ Real robot support (ROS 2 + Booster SDK)
- ✅ Webots simulation support
- ✅ All utility infrastructure

The integration preserves booster_deploy's proven patterns while adding Colosseum's architectural improvements (observation contracts, task-specific bundles, auto-discovery).
