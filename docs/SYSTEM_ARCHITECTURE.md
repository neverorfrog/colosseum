# System Architecture: Colosseum Deployment Ecosystem

This document describes the complete system architecture for deploying RL policies trained in Colosseum to both simulation (Circus) and real hardware (Booster T1), including all intermediate components.

## Table of Contents

1. [System Overview](#system-overview)
2. [Circus (Physics Simulator)](#circus-physics-simulator)
3. [SimBridge (Simulation Bridge)](#simbridge-simulation-bridge)
4. [Booster-Motion (Low-Level Controller)](#booster-motion-low-level-controller)
5. [spqrbooster2026 (Robot Application Framework)](#spqrbooster2026-robot-application-framework)
6. [Colosseum Deploy Pipeline (Python)](#colosseum-deploy-pipeline-python)
7. [Booster Robotics SDK (C++)](#booster-robotics-sdk-c)
8. [Data Flow: Simulation Pipeline](#data-flow-simulation-pipeline)
9. [Data Flow: Real Robot Pipeline](#data-flow-real-robot-pipeline)
10. [Message Types and DDS Topics](#message-types-and-dds-topics)
11. [Robot Configuration Reference](#robot-configuration-reference)
12. [Policy Observation Contract](#policy-observation-contract)

---

## System Overview

The ecosystem consists of six main components that together enable training RL policies and deploying them to both simulation and real hardware:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SIMULATION                                   │
│                                                                     │
│  Circus ←TCP/msgpack→ SimBridge ←FastDDS→ booster-motion            │
│  (MuJoCo)              (bridge)            (PD controller)          │
│                                                 ↕ FastDDS           │
│                                            Policy Node              │
│                                            (reads sensors,          │
│                                             writes joint targets)   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        REAL ROBOT                                   │
│                                                                     │
│  Hardware ←DDS→ booster-motion ←DDS→ spqrbooster2026                │
│  (T1 robot)     (PD controller)       (Bridge + AI + Controller)    │
│                       ↕ FastDDS                                     │
│                  Policy Node                                        │
│                  (same interface as simulation)                     │
└─────────────────────────────────────────────────────────────────────┘
```

The key invariant: **booster-motion** is always present in both pipelines. It reads sensor data, publishes `rt/low_state`, receives joint position targets on `rt/joint_ctrl`, computes PD torques, and outputs them. The policy node sees identical interfaces in both environments.

---

## Circus (Physics Simulator)

**Repository:** `external/circus/`

Circus is a MuJoCo-based multi-robot physics simulator designed for RoboCup humanoid soccer.

### Architecture

- **SimulationThread**: Runs MuJoCo physics at `model->opt.timestep` (e.g., 0.002s = 500Hz)
- **RobotManager**: TCP server on port 5555, manages multiple robot connections
- **Robot classes**: `BoosterT1` (23 DOF), `BoosterK1` (22 DOF)
- **Sensors**: Pose, IMU, Joints, Oracle (ground truth), Cameras (shared memory)

### Communication Protocol

**Transport:** TCP socket with MessagePack binary serialization

**Framing:**
- Header: 4 bytes (uint32_t, big-endian) = message length
- Body: MessagePack-encoded data

**Connection flow:**
1. Client connects to port 5555
2. Sends robot name (e.g., `"team1_Booster-T1_0"`) as initial identification
3. Server responds with initial robot state
4. Bidirectional: client sends commands, server responds with sensor data

### Sensor Data (Circus → SimBridge)

```
{
  "pose": {
    "position": [x, y, z],
    "quat_orientation": [w, x, y, z],
    "euler_orientation": [roll, pitch, yaw],
    "rotation_matrix": [9 floats],
    "transformation_matrix": [16 floats]
  },
  "imu": {
    "linear_acceleration": [ax, ay, az],
    "angular_velocity": [wx, wy, wz]
  },
  "joints": {
    "position": [23 values],       // rad
    "velocity": [23 values],       // rad/s
    "acceleration": [23 values],   // rad/s^2
    "torque": [23 values]          // Nm
  },
  "oracle": {
    "ball_position": [x, y, z],
    "teammates_positions": {...},
    "opponents_positions": {...},
    "goal_posts_positions": {...}
  }
}
```

### Command Input (SimBridge → Circus)

```
{
  "robot_name": "team1_Booster-T1_0",
  "joint_torques": [23 torque values in Nm]
}
```

Circus always receives **torque commands**. Position-to-torque conversion is handled by booster-motion.

### Camera Data

Shared memory ring buffer at `/dev/shm/circus_ipc/`:
- `{robot_name}_rgb.shm` — RGB images
- `{robot_name}_depth.shm` — Depth images
- Ring buffer with 3 slots, atomic sequence counter

### Joint Order (23-DOF T1)

```
[0]  AAHead_yaw
[1]  Head_pitch
[2]  Left_Shoulder_Pitch
[3]  Left_Shoulder_Roll
[4]  Left_Elbow_Pitch
[5]  Left_Elbow_Yaw
[6]  Right_Shoulder_Pitch
[7]  Right_Shoulder_Roll
[8]  Right_Elbow_Pitch
[9]  Right_Elbow_Yaw
[10] Waist
[11] Left_Hip_Pitch
[12] Left_Hip_Roll
[13] Left_Hip_Yaw
[14] Left_Knee_Pitch
[15] Left_Ankle_Pitch
[16] Left_Ankle_Roll
[17] Right_Hip_Pitch
[18] Right_Hip_Roll
[19] Right_Hip_Yaw
[20] Right_Knee_Pitch
[21] Right_Ankle_Pitch
[22] Right_Ankle_Roll
```

### Key Files

| File | Purpose |
|------|---------|
| `include/RobotManager.h` | TCP server, robot lifecycle |
| `include/robots/BoosterT1.h` | T1 robot implementation |
| `include/sensors/Pose.h` | Position/orientation sensor |
| `include/sensors/Joint.h` | Joint state sensor |
| `include/sensors/Imu.h` | IMU sensor |
| `include/sensors/Oracle.h` | Ground-truth oracle |
| `include/sensors/ImageSharedMemoryWriter.h` | Camera shared memory writer |
| `src/SimulationThread.cpp` | Physics simulation loop |

---

## SimBridge (Simulation Bridge)

**Repository:** `external/simbridge/`

SimBridge is a ROS2 node that bridges Circus (TCP) with the ROS2 DDS topics consumed by booster-motion.

### Architecture

SimBridge is a standalone ROS2 node running in a Docker container alongside booster-motion. It translates between Circus's TCP/msgpack protocol and ROS2 standard message types.

**Key design choice:** SimBridge has **no dependency on the Booster SDK**. It uses only standard `sensor_msgs` types (`JointState`, `Imu`) on the DDS side. The conversion from these standard types into the Booster SDK's `LowState` format (with IMU state and per-joint serial motor states) is done entirely inside booster-motion's `simulator_out` (`RsSdkOutput`) module. Similarly, booster-motion's `simulator_in` (`RsSdkInput`) module converts computed torques back into `sensor_msgs/JointState` on `/booster/ros2_k2_joint_cmd` for SimBridge to forward to Circus.

This means **the policy node never interacts with SimBridge at all** — it only communicates with booster-motion via `rt/low_state` and `rt/joint_ctrl`.

### Configuration

Environment variables:
```bash
ROBOT_NAME=team1_Booster-T1_0   # Robot identifier in Circus
SERVER_IP=127.0.0.1              # Circus TCP server IP
CIRCUS_PORT=5555                 # Circus TCP port
JOYSTICK_DEVICE=/dev/input/js0   # Optional gamepad device
```

### Topics Published (SimBridge → DDS/ROS2)

| Topic | Message Type | Source |
|-------|--------------|--------|
| `/booster/ros2_k2_joint_states` | `sensor_msgs/JointState` | Circus joint data |
| `/booster/ros2_k2_imu` | `sensor_msgs/Imu` | Circus IMU sensor |
| `ball_position` | `geometry_msgs/Point` | Oracle data |
| `robot_pose` | `geometry_msgs/Pose` | Pose sensor |
| `teammates_positions` | `geometry_msgs/PoseArray` | Oracle data |
| `opponents_positions` | `geometry_msgs/PoseArray` | Oracle data |
| `goal_posts_positions` | `geometry_msgs/Polygon` | Oracle data |
| `/camera/.../image_raw` | `sensor_msgs/Image` | RGB/Depth (shared memory) |
| `remote_controller_state` | `booster_interface/RemoteControllerState` | Joystick |

### Topics Subscribed (DDS/ROS2 → SimBridge → Circus)

| Topic | Message Type | Destination |
|-------|--------------|-------------|
| `/booster/ros2_k2_joint_cmd` | `sensor_msgs/JointState` | Circus (as torques) |

### Joystick Input

Background thread reads from `/dev/input/js0`:
- Analog sticks: `lx`, `ly`, `rx`, `ry` (normalized [-1, 1])
- Buttons: `a`, `b`, `x`, `y`, `lb`, `rb`, `lt`, `rt`, `back`, `start`, `ls`, `rs`
- D-pad: `hat_u`, `hat_d`, `hat_l`, `hat_r`
- Published as `booster_interface/msg/RemoteControllerState`

### Socket Protocol

Both directions use the same framing:
```
[uint32_t: message_size (big-endian)] [msgpack_data]
```

Buffer sizes: send=10MB, recv=1MB, max message=1MB

### Key Files

| File | Purpose |
|------|---------|
| `src/node/src/bridge_node.cpp` | Main bridge logic, TCP I/O, publishers/subscribers |
| `src/node/include/bridge_node.hpp` | Bridge node class definition |
| `src/node/include/Message.hpp` | MessagePack extraction helpers |
| `src/node/include/ImageSharedMemoryReader.hpp` | Camera shared memory reader |
| `src/node/src/main.cpp` | Entry point |
| `src/msgs/booster_interface/msg/*.msg` | ROS2 message definitions |

### booster_interface Message Package (in simbridge)

The `booster_interface` ROS2 message package lives in `src/msgs/booster_interface/msg/`. It is compiled as part of the Docker environment and provides the **shared type definitions** for the Booster SDK DDS topics (`rt/low_state`, `rt/joint_ctrl`, etc.).

**Important:** SimBridge's own `bridge_node.cpp` does NOT publish or subscribe to `LowState`/`LowCmd` topics. These type definitions are used by **booster-motion** (running in the same Docker container) via its internal SDK.

**LowState.msg:**
```
ImuState imu_state
MotorState[] motor_state_parallel
MotorState[] motor_state_serial
```

**LowCmd.msg:**
```
int8 cmd_type     # 0=parallel, 1=serial
MotorCmd[] motor_cmd
```

**MotorState.msg:**
```
int8 mode
float32 q         # position (rad)
float32 dq        # velocity (rad/s)
float32 ddq       # acceleration (rad/s^2)
float32 tau_est   # estimated torque (Nm)
int8 temperature
uint32 lost
uint32[2] reserve
```

**MotorCmd.msg:**
```
int8 mode
float32 q         # target position
float32 dq        # target velocity
float32 tau       # feedforward torque
float32 kp        # stiffness
float32 kd        # damping
float32 weight    # 0=defer to default, 1=full override
```

**ImuState.msg:**
```
float32[3] rpy    # roll, pitch, yaw
float32[3] gyro   # angular velocity
float32[3] acc    # linear acceleration
```

---

## Booster-Motion (Low-Level Controller)

**Location:** `external/simbridge/tools/booster_motion/`

Booster-motion is Booster Robotics' proprietary motion control engine. It is a **closed-source binary** that runs on both the real robot and in simulation. It is the critical middleman between high-level policy commands and low-level motor torques.

### What It Does

1. **Reads sensor data** via module `simulator_out` (`RsSdkOutput`) — subscribes to `/booster/ros2_k2_joint_states` and `/booster/ros2_k2_imu` (published by SimBridge) and converts them to internal format
2. **Publishes `rt/low_state`** — reformatted sensor feedback (Booster SDK `LowState` type) for any subscriber (policy node, spqrbooster2026 Bridge)
3. **Subscribes to `rt/joint_ctrl`** — receives `LowCmd` messages with position targets (q, kp, kd) from the policy node
4. **Runs state estimation** (EKF) and **parallel mechanism conversion** (ankle joints use a parallel linkage; booster-motion converts serial virtual joint positions ↔ parallel motor positions internally)
5. **Computes PD torques**: `τ = kp * (q_target - q_measured) + kd * (dq_target - dq_measured) + τ_feedforward`
6. **Publishes torque commands** via module `simulator_in` (`RsSdkInput`) → `/booster/ros2_k2_joint_cmd` (`sensor_msgs/JointState`, effort field = torques) → SimBridge → Circus

### Internal Module Graph (Lua)

Booster-motion uses a Lua-configured dataflow graph. Key modules for the RL policy use-case:

```
simulator_out (RsSdkOutput)     ← reads /booster/ros2_k2_joint_states + /booster/ros2_k2_imu
    ↓ joint_states, imu
joint_map_output / parallel_mech_output   ← parallel mechanism FK (serial→parallel)
    ↓ joint_states_mapped
[EKF state estimation, planners, rl_locomotion modules, ...]
    ↓ motor_pvt_commands
intercept_motor_cmd (PvtCmdIntercept)     ← merges planner + user commands
    ↓ motor_pvt_intercepted
joint_map_input / parallel_mech_input     ← parallel mechanism IK (serial→parallel)
    ↓ motor_pvt_commands_mapped
simulator_in (RsSdkInput)       → publishes /booster/ros2_k2_joint_cmd (torques)
```

When the policy publishes `rt/joint_ctrl` (LowCmd), booster-motion's internal PD controller uses those q/kp/kd values to compute `motor_pvt_commands` which feed into this graph.

**Note:** The `rl_locomotion` module (`librl_locomotion.so`) is Booster's **own built-in RL controller** — not our external policy. Our policy replaces it by publishing position targets directly on `rt/joint_ctrl`.

### Execution

```bash
# Launch script: booster-simulate-run.sh
export LD_LIBRARY_PATH=$(pwd)/lib:$(pwd)/lib-usr-local:$(pwd)/lib-x86_64-linux-gnu:$LD_LIBRARY_PATH
sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./booster-motion -mode sim -config ./configs/config_isaac.lua
```

Modes:
- `-mode sim` — simulation mode (reads from DDS topics published by SimBridge)
- No flag — real robot mode (reads from hardware DDS directly)

### DDS Topics

**Published by booster-motion:**

| Topic | Type | Purpose |
|-------|------|---------|
| `rt/low_state` | `LowState` | Sensor state (IMU + 23 joint states) |
| `rt/odometer_state` | `Odometer` | Odometry/position estimate |
| `rt/fall_down` | `FallDownState` | Fall detection |
| `rt/remote_controller_state` | `RemoteControllerState` | Joystick passthrough |
| `/booster/ros2_k2_joint_cmd` | `sensor_msgs/JointState` (effort = torques) | Computed torques → SimBridge → Circus (sim only) |

**Subscribed by booster-motion:**

| Topic | Type | Purpose |
|-------|------|---------|
| `rt/joint_ctrl` | `LowCmd` (booster_interface) | Position targets with PD gains from policy |
| `/booster/ros2_k2_joint_states` | `sensor_msgs/JointState` | Joint positions/velocities from SimBridge (sim only) |
| `/booster/ros2_k2_imu` | `sensor_msgs/Imu` | IMU data from SimBridge (sim only) |

### PD Control Law

```
τ_i = kp_i * (q_target_i - q_measured_i)
    + kd_i * (dq_target_i - dq_measured_i)
    + tau_feedforward_i

τ_final_i = clamp(τ_i, -effort_limit_i, +effort_limit_i)
```

### Configuration

Lua-based data flow graph system:
- `configs/config_isaac.lua` — main simulator config
- `configs/common_module_options_t1_rl_isaac.lua` — PD gains, torque limits
- `configs/common_graph_define_t1_rl_isaac.lua` — module connection graph

**Default PD Gains (from Lua config):**
```lua
kp_joint = {
    10.0, 10.0,                                         -- Head (yaw, pitch)
    10.0, 10.0, 10.0, 10.0,                             -- Left arm
    10.0, 10.0, 10.0, 10.0,                             -- Right arm
    100.0,                                               -- Waist
    100.0, 100.0, 100.0, 100.0, 25.0, 25.0,             -- Left leg
    100.0, 100.0, 100.0, 100.0, 25.0, 25.0              -- Right leg
}
kd_joint = {
    0.2, 0.2,
    0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5,
    0.2,
    1.0, 1.0, 1.0, 1.0, 0.5, 0.5,
    1.0, 1.0, 1.0, 1.0, 0.5, 0.5
}
```

### FastDDS Profile

`fastdds_profile.xml` restricts to loopback only:
```xml
<interfaceWhiteList>
    <address>127.0.0.1</address>
</interfaceWhiteList>
```

### Directory Structure

```
tools/booster_motion/
├── booster-motion          # Main binary (23MB)
├── booster-server          # Companion server (87MB)
├── booster-simulate-run.sh # Launch script
├── fastdds_profile.xml     # DDS config (loopback only)
├── mck                     # Webots runner (19MB)
├── configs/                # Lua configuration files
│   ├── config_isaac.lua
│   ├── common_module_options_t1_rl_isaac.lua
│   ├── common_graph_define_t1_rl_isaac.lua
│   ├── robot.urdf / t1.urdf
│   └── joint_position_*.txt  # Trajectory data
├── lib/                    # Core libraries (141MB)
│   ├── librs_sdk_interface.so
│   ├── librl_locomotion.so
│   └── libpvt.so
├── lib-usr-local/          # Third-party (18MB)
│   ├── libfastrtps.so.2.13
│   └── librbdl.so.3.1.3
└── lib-x86_64-linux-gnu/   # System libs (13MB)
```

---

## spqrbooster2026 (Robot Application Framework)

**Repository:** `external/spqrbooster2026/`

The C++ application framework that runs on the real Booster T1 robot. It provides behavior, vision, and motor control through a modular architecture.

### Module System

All processing nodes inherit from `Module` and declare typed `Input<T>`/`Output<T>` ports. A `Scheduler` determines execution order based on data dependencies.

### Key Components

#### Bridge (Sensor Aggregation)

**Files:** `src/app/nodes/bridge/Bridge.h`, `Bridge.cpp`

Subscribes to real robot DDS topics via `FastDDSSubscriber<T>`:
- `rt/low_state` → `ImuState` (rpy, gyro, acc)
- `rt/odometer_state` → `Odometry` (x, y, theta)
- `rt/fall_down` → `FallDownState`
- Camera topics → `RGBImage`, `DepthImage`
- `rt/remote_controller_state` → `Joystick`

Implements head pose buffering (10-sample ring buffer) for camera-pose synchronization.

#### OracleBridge (Simulation Ground Truth)

**Files:** `src/app/nodes/bridge/OracleBridge.h`, `OracleBridge.cpp`

Reads ground-truth data from Circus (via SimBridge DDS topics):
- `rt/robot_pose` → `RobotPose`
- `rt/ball_position` → `Ball`
- `rt/opponents_positions` → `Obstacles`
- `rt/goal_posts_positions` → `FieldMarkers`

#### BoosterController (Command Executor)

**Files:** `src/app/nodes/booster/BoosterController.h`, `BoosterController.cpp`

Reads internal command topics from behavior modules:
- `RCVelocity` → velocity commands (vx, vy, vyaw)
- `RCHeadRotation` → head pitch/yaw
- `RobotModeCommand` → mode transitions
- `Joystick` → emergency stop detection (RB+RT combo)

Processing:
1. Apply deadzone filtering (|vx|, |vy| < 0.01 → 0)
2. Apply low-pass filter (alpha=0.8) for smooth motion
3. Forward to BoosterRobotPortal

#### BoosterRobotPortal (SDK Wrapper)

**Files:** `src/app/booster/BoosterRobotPortal.h`, `BoosterRobotPortal.cpp`

Singleton wrapping three Booster SDK clients:
- `B1LocoClient` — locomotion (Move, ChangeMode, GetUp, RotateHead)
- `VisionClient` — camera detection
- `LightControlClient` — LED control

**Key methods:**
- `move(vx, vy, vyaw)` — high-level velocity command (uses built-in gait)
- `changeMode(mode)` — switch robot mode (kDamping, kPrepare, kWalking, kCustom)
- `setArmStiffness(kp, kd, weight)` — direct joint control for arms (indices 2-9)
- `rotateHead(pitch, yaw)` — head position control

**Direct joint control** via `ChannelPublisher<LowCmd>` on `rt/joint_ctrl`:
- Rate-limited to 50Hz
- Per-joint: q, dq, tau, kp, kd, weight

### Internal Data Types

```cpp
struct RCVelocity { float vx, vy, vyaw; };
struct RCHeadRotation { float pitch, yaw; };
struct RobotModeCommand { RobotMode mode; bool request_get_up; bool manual; };
struct Odometry { Eigen::Vector3f pose; };  // {x, y, theta}
struct ImuState { Eigen::Vector3f rpy, gyro, acc; };
struct FallDownState { enum State { IS_READY, IS_FALLING, HAS_FALLEN, IS_GETTING_UP }; };
struct Joystick { float lx,ly,rx,ry; bool a,b,x,y,lb,rb,...; };
```

### Build System

- **CMake** with C++20
- **pixi** for dependency management
- Dependencies: `booster_robotics_sdk==1.5.0`, `Eigen3`, `yaml-cpp`, `rerun-sdk`
- Cross-compilation: x86_64 (development) and aarch64 (robot deployment)

---

## Colosseum Deploy Pipeline (Python)

**Location:** `src/colosseum/deploy/`

The Python deployment system provides sim-to-sim testing via MuJoCo and real robot deployment.

### Configuration (Frozen Dataclasses)

```python
ControllerConfig:
  policy_dt: float = 0.02          # 50Hz policy execution
  robot: RobotConfig               # Joint names, gains, limits
  policy: PolicyConfig             # ONNX checkpoint, action scale
  vel_command: VelocityCommandConfig
  mujoco: MujocoConfig             # Sim-to-sim settings
  booster: BoosterConfig           # Real robot settings
```

### Backends

**MujocoController** (`backends/mujoco.py`):
- Loads MJCF, creates position actuators programmatically
- Control loop: update_state → policy_step → ctrl_step (4x decimation at 200Hz)
- Used for local sim-to-sim testing with joystick/keyboard input

**BoosterRobotPortal** (`backends/booster.py`):
- Multi-process: main process (ROS2) + inference process (policy)
- Cross-process communication via `SyncedArray` (shared memory + file locks)
- Subscribes to `rt/low_state`, publishes `LowCmd` to `rt/joint_ctrl`

### Policy Architecture

```python
class T1VelocityPolicy(Policy):
    def compute_observation(self) -> torch.Tensor:
        # 82-dim: [lin_vel(3), ang_vel(3), proj_gravity(3),
        #          joint_pos_rel(23), joint_vel(23), last_action(23),
        #          vel_commands(3)]

    def inference(self) -> torch.Tensor:
        obs = self.compute_observation()     # (1, 82)
        action = self._model(obs).flatten()  # (23,) normalized
        targets = action * action_scale + default_joint_pos
        return targets                       # (23,) in hardware order
```

---

## Booster Robotics SDK (C++)

**Location:** `external/booster_robotics_sdk/`

### Publisher/Subscriber Templates

```cpp
// Publishing
booster::robot::ChannelPublisher<booster_interface::msg::LowCmd> publisher(
    booster::robot::b1::kTopicJointCtrl);  // "rt/joint_ctrl"
publisher.InitChannel();
publisher.Write(&low_cmd_msg);

// Subscribing (with callback)
booster::robot::ChannelSubscriber<booster_interface::msg::LowState> subscriber(
    booster::robot::b1::kTopicLowState);  // "rt/low_state"
subscriber.InitChannel(
    [](const void* msg) {
        auto* state = static_cast<const booster_interface::msg::LowState*>(msg);
        // process state...
    }
);
```

### B1LocoClient (High-Level Control)

```cpp
booster::robot::b1::B1LocoClient client;
client.Init();
client.ChangeMode(booster::robot::RobotMode::kCustom);
client.Move(vx, vy, vyaw);
client.RotateHead(pitch, yaw);
```

### DDS Topic Constants

```cpp
namespace booster::robot::b1 {
    constexpr auto kTopicLowState = "rt/low_state";
    constexpr auto kTopicJointCtrl = "rt/joint_ctrl";
    constexpr auto kTopicOdometerState = "rt/odometer_state";
    constexpr auto kTopicFallDown = "rt/fall_down";
    // ...
}
```

### Linking

```cmake
find_package(booster_robotics_sdk REQUIRED)
target_link_libraries(target booster::booster_robotics_sdk)
# Transitively links: libfastrtps, libfastcdr, libfoonathan_memory
```

---

## Data Flow: Simulation Pipeline

The complete data loop when running with Circus:

```
┌─────────────┐    TCP/msgpack     ┌─────────────┐
│   CIRCUS    │◄──────────────────►│  SIMBRIDGE   │
│  (MuJoCo    │  sensor data up    │  (ROS2/DDS   │
│   physics)  │  torques down      │   bridge)    │
└─────────────┘                    └──────┬───────┘
                                          │ DDS topics
                                          │ (sensor data up,
                                          │  torques down)
                                          ▼
                                   ┌──────────────┐
                                   │BOOSTER-MOTION│
                                   │ (PD control) │
                                   │              │
                                   │ publishes:   │
                                   │  rt/low_state│
                                   │              │
                                   │ subscribes:  │
                                   │ rt/joint_ctrl│
                                   │              │
                                   │ outputs:     │
                                   │ torques →    │
                                   │ SimBridge    │
                                   └──────┬───────┘
                                          │ FastDDS
                                          │ rt/low_state (up)
                                          │ rt/joint_ctrl (down)
                                          ▼
                                   ┌──────────────┐
                                   │ POLICY NODE  │
                                   │              │
                                   │ reads:       │
                                   │  rt/low_state│
                                   │              │
                                   │ computes:    │
                                   │  observation │
                                   │  → ONNX      │
                                   │  → action    │
                                   │              │
                                   │ publishes:   │
                                   │ rt/joint_ctrl│
                                   │ (q, kp, kd)  │
                                   └──────────────┘
```

**Timing:**
- Circus physics: ~500Hz (0.002s timestep)
- SimBridge polling: continuous (1μs timeout)
- booster-motion: internal control rate
- Policy node: 50Hz (0.02s policy_dt)

### Docker Deployment (from circus/dockerfiles/booster.conf)

```
Container:
  1. booster-motion (priority=1, autostart=true)
  2. simbridge (started manually)
  3. application layer (started manually)
```

---

## Data Flow: Real Robot Pipeline

On the real Booster T1, Circus and SimBridge are removed. booster-motion reads directly from hardware:

```
┌─────────────┐
│  HARDWARE   │
│  (T1 Robot) │
│             │
│ Sensors:    │──DDS──► booster-motion ──DDS──► Bridge (spqrbooster2026)
│  IMU        │                │                    │
│  Encoders   │                │ rt/low_state       │ ImuState, Odometry,
│  Cameras    │                │                    │ Joystick, etc.
│             │                ▼                    ▼
│ Actuators:  │◄─DDS── booster-motion ◄─DDS── BoosterController
│  Motors     │         (PD torques)          (via BoosterRobotPortal)
└─────────────┘
                               ▲
                               │ rt/joint_ctrl
                               │
                        ┌──────────────┐
                        │ POLICY NODE  │
                        │ (reads       │
                        │  rt/low_state│
                        │  writes      │
                        │ rt/joint_ctrl│
                        └──────────────┘
```

The policy node code is **identical** in both pipelines — it only sees `rt/low_state` and `rt/joint_ctrl`.

---

## Message Types and DDS Topics

### Complete Topic Map

| Topic | Publisher | Subscriber(s) | Message Type | Layer |
|-------|----------|---------------|--------------|-------|
| `rt/low_state` | booster-motion | Policy node, Bridge | `booster_interface/LowState` | Booster SDK DDS |
| `rt/joint_ctrl` | Policy node | booster-motion | `booster_interface/LowCmd` | Booster SDK DDS |
| `rt/odometer_state` | booster-motion | Bridge | `booster_interface/Odometer` | Booster SDK DDS |
| `rt/fall_down` | booster-motion | Bridge | `booster_interface/FallDownState` | Booster SDK DDS |
| `rt/remote_controller_state` | booster-motion | Bridge, Policy | `booster_interface/RemoteControllerState` | Booster SDK DDS |
| `/booster/ros2_k2_joint_cmd` | booster-motion | SimBridge | `sensor_msgs/JointState` (effort=torques) | ROS2 standard (sim only) |
| `/booster/ros2_k2_joint_states` | SimBridge | booster-motion | `sensor_msgs/JointState` | ROS2 standard (sim only) |
| `/booster/ros2_k2_imu` | SimBridge | booster-motion | `sensor_msgs/Imu` | ROS2 standard (sim only) |

**Two distinct type layers exist in simulation:**
- **Booster SDK DDS** (`rt/*` topics): used between the policy node and booster-motion — these use `booster_interface` types
- **ROS2 standard** (`/booster/ros2_k2_*` topics): used between booster-motion and SimBridge — these use `sensor_msgs` types

The policy node only ever touches the first layer. SimBridge only touches the second. Booster-motion bridges both layers internally.

### LowState Message Structure

```
LowState:
  imu_state:
    rpy[3]    — roll, pitch, yaw (rad)
    gyro[3]   — angular velocity (rad/s)
    acc[3]    — linear acceleration (m/s^2)
  motor_state_serial[23]:
    q         — joint position (rad)
    dq        — joint velocity (rad/s)
    ddq       — joint acceleration (rad/s^2)
    tau_est   — estimated torque (Nm)
    temperature, lost, reserve
```

### LowCmd Message Structure

```
LowCmd:
  cmd_type  — 0=parallel, 1=serial
  motor_cmd[23]:
    q       — target position (rad)
    dq      — target velocity (rad/s)
    tau     — feedforward torque (Nm)
    kp      — proportional gain (stiffness)
    kd      — derivative gain (damping)
    weight  — 0=defer to default, 1=override
```

---

## Robot Configuration Reference

### T1 23-DOF Joint Parameters

| Index | Joint Name | Kp (training) | Kd (training) | Effort Limit | Default Pos |
|-------|-----------|---------------|---------------|-------------|-------------|
| 0 | AAHead_yaw | 5.0 | 0.5 | 7 | 0.0 |
| 1 | Head_pitch | 5.0 | 0.5 | 7 | 0.0 |
| 2 | Left_Shoulder_Pitch | 20.0 | 0.5 | 48 | 0.0 |
| 3 | Left_Shoulder_Roll | 20.0 | 0.5 | 30 | 0.0 |
| 4 | Left_Elbow_Pitch | 20.0 | 0.5 | 30 | 0.0 |
| 5 | Left_Elbow_Yaw | 20.0 | 0.5 | 48 | 0.0 |
| 6 | Right_Shoulder_Pitch | 20.0 | 0.5 | 48 | 0.0 |
| 7 | Right_Shoulder_Roll | 20.0 | 0.5 | 30 | 0.0 |
| 8 | Right_Elbow_Pitch | 20.0 | 0.5 | 30 | 0.0 |
| 9 | Right_Elbow_Yaw | 20.0 | 0.5 | 48 | 0.0 |
| 10 | Waist | 200.0 | 5.0 | 8 | 0.0 |
| 11-16 | Left Leg | 200/50 | 5.0/3.0 | 24-60 | task-specific |
| 17-22 | Right Leg | 200/50 | 5.0/3.0 | 24-60 | task-specific |

*Note: Kp/Kd values in training (colosseum) may differ from booster-motion's defaults. The policy publishes its own kp/kd in the LowCmd, which override booster-motion's defaults when weight=1.*

### Joint Armature

All joints: 0.3 (reflected motor inertia, critical for simulation stability)

### Action Scale

Uniform: 0.25 across all 23 joints

---

## Policy Observation Contract

### Velocity Task Observation (82 dimensions)

The ONNX model expects a single input tensor of shape `(1, 82)` with components concatenated in this exact order:

| Component | Dimensions | Description | Source |
|-----------|-----------|-------------|--------|
| `base_lin_vel` | 3 | Linear velocity in body frame (m/s) | `LowState` IMU or derived |
| `base_ang_vel` | 3 | Angular velocity in body frame (rad/s) | `LowState.imu_state.gyro` |
| `projected_gravity` | 3 | Gravity vector in body frame | Computed from `imu_state.rpy` → quaternion → rotate [0,0,-1] |
| `joint_pos_rel` | 23 | Joint positions minus default pose (rad) | `motor_state_serial[i].q - default_pos[i]` |
| `joint_vel` | 23 | Joint velocities (rad/s) | `motor_state_serial[i].dq` |
| `last_action` | 23 | Previous policy output (normalized) | Stored from last inference step |
| `velocity_commands` | 3 | [vx_cmd, vy_cmd, vyaw_cmd] (m/s, rad/s) | Joystick input scaled by max velocities |

**Total: 3 + 3 + 3 + 23 + 23 + 23 + 3 = 81** *(verify against actual exported model)*

### ONNX Model I/O

- **Input name:** `"obs"`, shape `(1, N)` where N = observation size
- **Output name:** `"actions"`, shape `(1, 23)` — normalized actions in [-1, 1]
- **Observation normalizer:** Baked into the ONNX model (exported as `nn.Sequential(normalizer, actor)`)

### Action → Joint Target Conversion

```
action = onnx_model(observation)           # (23,) in [-1, 1]
scaled_action = action * action_scale      # action_scale = 0.25
joint_targets = scaled_action + default_joint_pos  # (23,) in rad
```

The joint_targets are then published as `LowCmd.motor_cmd[i].q` with corresponding `kp` and `kd` values.
