# Deployment Architecture

## Two Backends

Colosseum has two deployment backends, each serving a different purpose:

### 1. Python MuJoCo Backend (quick local testing)

Self-contained sim-to-sim. MuJoCo runs the physics, the policy runs in the same process. No external dependencies.

```
MujocoController (Python)
  ├── MuJoCo physics (200Hz, 4x decimation)
  ├── Policy inference (50Hz, ONNX)
  ├── Joystick/keyboard input
  └── Viewer
```

Location: `src/colosseum/deploy/backends/mujoco.py`

Use this for: fast iteration, debugging observations, verifying a policy works at all.

### 2. C++ Sim2Sim (realistic pipeline via booster-motion)

Standalone C++ executable that plugs into the Circus + SimBridge + booster-motion stack. Reads sensor data and publishes joint position targets over FastDDS. The same code runs against both the simulator and the real robot.

```
Circus (MuJoCo physics)
  ↕ TCP/msgpack
SimBridge (DDS bridge)
  ↕ FastDDS
booster-motion (PD controller)
  ↕ FastDDS: rt/low_state, rt/joint_ctrl
sim2sim executable (policy inference, 50Hz)
```

Location: `src/sim2sim/`

Use this for: realistic sim-to-sim testing, validation before real robot, and (with zero code changes) deployment on real hardware.

On the real robot, Circus + SimBridge are removed. booster-motion reads from hardware directly. The sim2sim executable sees the same DDS interface.

## Directory Layout

```
src/
├── colosseum/                  # Python package
│   └── deploy/
│       ├── core/               # Base abstractions (Policy, RobotData, BaseController)
│       ├── backends/
│       │   └── mujoco.py       # Python MuJoCo backend
│       ├── config/             # Frozen dataclasses (ControllerConfig, RobotConfig, etc.)
│       └── tasks/              # Task-specific deployment (policy.py, config.py per task)
│
└── sim2sim/                    # C++ standalone executable
    ├── CMakeLists.txt
    ├── pixi.toml
    ├── include/
    │   ├── robot_portal.hpp    # DDS subscribe/publish (rt/low_state, rt/joint_ctrl)
    │   ├── policy.hpp          # ONNX inference
    │   ├── robot_data.hpp      # State container
    │   └── config.hpp          # Robot constants (joints, gains, defaults)
    └── src/
        ├── main.cpp            # 50Hz control loop
        ├── robot_portal.cpp
        ├── policy.cpp
        └── config.cpp
```

ONNX models live in `models/` at the repository root.

## Key Interfaces

### DDS Topics (C++ sim2sim)

| Topic | Direction | Type | Content |
|-------|-----------|------|---------|
| `rt/low_state` | subscribe | `LowState` | IMU (rpy, gyro) + 23 joint states (q, dq, tau) |
| `rt/joint_ctrl` | publish | `LowCmd` | 23 joint targets (q, kp, kd, weight) |
| `rt/remote_controller_state` | subscribe | `RemoteControllerState` | Joystick axes and buttons |

### Python Abstractions

**BaseController**: abstract control loop (update_state, policy_step, ctrl_step)
**Policy**: loads ONNX model, builds observation, runs inference, returns joint targets
**RobotData**: joint_pos, joint_vel, root_quat_w, root_ang_vel_b, projected_gravity_b
**ControllerConfig**: frozen dataclass composing RobotConfig + PolicyConfig + backend config

## Action Pipeline

Both backends follow the same action pipeline:

```
1. Read sensor state (joint positions, velocities, IMU)
2. Read velocity commands (joystick)
3. Build observation vector (82 dims for velocity task)
4. Run ONNX inference → normalized action (23 dims, [-1, 1])
5. Scale: joint_targets = action * action_scale + default_joint_pos
6. Send to actuators (MuJoCo ctrl or LowCmd with kp/kd)
```
