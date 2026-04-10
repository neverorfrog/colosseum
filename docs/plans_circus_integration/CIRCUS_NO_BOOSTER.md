# Circus Without Booster: Direct TCP Control

This document describes how to use Circus as a **plain MuJoCo simulator** by connecting a client directly to its TCP server, bypassing SimBridge, booster-motion, the Booster SDK, and Docker entirely.

This approach is simpler than the full booster stack, has zero external dependencies beyond Circus itself, and is the right basis for formalizing a standalone Circus release.

---

## What Changes vs the Booster Stack

| | Booster stack | No-booster (this doc) |
|---|---|---|
| Physics | Circus (MuJoCo) | Circus (MuJoCo) |
| Sensor bridge | SimBridge (ROS2 node) | **None** — client reads directly |
| Low-level controller | booster-motion (PD, parallel mech) | **PD computed in-process** |
| Communication | DDS (`rt/low_state`, `rt/joint_ctrl`) | **TCP/msgpack to port 5555** |
| Docker container | Required | **Not required** |
| Booster SDK | Required | **Not required** |
| Dependencies | circus + simbridge + booster_robotics_sdk + booster_robotics_sdk_ros2 | **circus only** |

The client speaks the same protocol that SimBridge currently uses. Instead of SimBridge bridging Circus to DDS, the client is the TCP peer directly.

---

## Architecture

```
Host
├── circus (Qt GUI, MuJoCo physics, TCP server on port 5555)
└── policy_node (connects to localhost:5555, reads sensors, sends torques)
```

No Docker. No supervisord. Two processes on the same machine.

---

## Required Change in Circus: `--no-containers` Flag

Currently `RobotManager::startContainers()` always spawns one Docker container per robot. The TCP server is started inside this call (`startCommunicationServer(frameworkCommunicationPort)`).

To support the no-booster mode, these two responsibilities need to be separated:

```cpp
// RobotManager.h — split startContainers() into two methods:

void startServer() {
    startCommunicationServer(frameworkCommunicationPort);
}

void startContainers() {
    startServer();  // existing TCP server startup

    // ... existing Docker container creation ...
    for (std::shared_ptr<Robot> r : robots_) {
        r->container = std::make_unique<Container>("CIRCUS_" + r->name + "_container");
        r->container->create(r->name, image, binds);
        r->container->start();
    }
}
```

Then Circus accepts a `--no-containers` CLI flag:

```cpp
// CircusApplication.cpp
if (noContainers) {
    RobotManager::instance().startServer();  // TCP only, no Docker
} else {
    RobotManager::instance().startContainers();  // full stack
}
```

This is the only Circus change required. Everything else (physics, sensors, visualization) is unaffected.

---

## Circus TCP Protocol

The client speaks the same protocol as SimBridge. Both directions use the same framing:

```
[uint32_t: message_length (big-endian)] [msgpack_payload]
```

### Connection Handshake

```
Client connects to TCP port 5555
Client sends: msgpack(robot_name_string)
              e.g. msgpack("red_Booster-T1_0")
```

The robot name must match what's declared in the scene YAML.

### Sensor Data (Circus → Client)

Circus sends one message per simulation tick after the client connects:

```json
{
  "robot_name": "red_Booster-T1_0",
  "pose": {
    "position": [x, y, z],
    "quat_orientation": [w, x, y, z],
    "euler_orientation": [roll, pitch, yaw]
  },
  "imu": {
    "linear_acceleration": [ax, ay, az],
    "angular_velocity":    [wx, wy, wz]
  },
  "joints": {
    "position":     [23 values, rad],
    "velocity":     [23 values, rad/s],
    "acceleration": [23 values, rad/s²],
    "torque":       [23 values, Nm]
  },
  "oracle": {
    "ball_position":         [x, y, z],
    "teammates_positions":   {...},
    "opponents_positions":   {...},
    "goal_posts_positions":  {...}
  }
}
```

### Command (Client → Circus)

```json
{
  "robot_name": "red_Booster-T1_0",
  "joint_torques": [23 torque values, Nm]
}
```

**Circus always receives torques.** The position-to-torque PD conversion that booster-motion normally handles must be done by the client.

### Joint Order (23-DOF T1)

```
[0]  AAHead_yaw         [6]  Right_Shoulder_Pitch  [12] Left_Hip_Roll
[1]  Head_pitch         [7]  Right_Shoulder_Roll   [13] Left_Hip_Yaw
[2]  Left_Shoulder_Pitch[8]  Right_Elbow_Pitch     [14] Left_Knee_Pitch
[3]  Left_Shoulder_Roll [9]  Right_Elbow_Yaw       [15] Left_Ankle_Pitch
[4]  Left_Elbow_Pitch   [10] Waist                 [16] Left_Ankle_Roll
[5]  Left_Elbow_Yaw     [11] Left_Hip_Pitch        [17] Right_Hip_Pitch
                                                    [18] Right_Hip_Roll
                                                    [19] Right_Hip_Yaw
                                                    [20] Right_Knee_Pitch
                                                    [21] Right_Ankle_Pitch
                                                    [22] Right_Ankle_Roll
```

**Note on ankle joints:** In the booster stack, `motor_state_serial` contains virtual serial joint positions after booster-motion's parallel mechanism FK. Here, `joints.position` from Circus contains **serial (virtual) positions directly** — MuJoCo models the T1 with serial joints, the parallel linkage is a Circus-level approximation. So the values are compatible with the booster stack's `motor_state_serial`.

---

## Client Architecture

The client replaces the SimBridge + booster-motion + policy node trinity with a single process:

```cpp
class CircusClient {
public:
    CircusClient(const std::string& host, int port, const std::string& robot_name);

    // Receive one sensor packet from Circus (blocks until available)
    SensorData receive();

    // Send joint torques to Circus
    void send(const float torques[23]);

private:
    int fd_;
    std::string robot_name_;
};

struct SensorData {
    float joint_pos[23];      // rad
    float joint_vel[23];      // rad/s
    float position[3];        // base position in world frame (m)
    float quat[4];            // base orientation (w, x, y, z)
    float ang_vel[3];         // angular velocity body frame (rad/s)
    float lin_acc[3];         // linear acceleration (m/s²)
    // Derived fields (computed client-side):
    float projected_gravity[3];
    float lin_vel[3];         // see "Base Linear Velocity" section
};
```

### PD Control (Replaces booster-motion)

```cpp
// Compute torques from position targets
void pd_control(
    const float q_target[23],   // from policy
    const float q[23],          // measured joint positions
    const float dq[23],         // measured joint velocities
    const float kp[23],         // stiffness (match training config)
    const float kd[23],         // damping   (match training config)
    const float effort_limit[23],
    float torques_out[23])
{
    for (int i = 0; i < 23; i++) {
        float tau = kp[i] * (q_target[i] - q[i]) - kd[i] * dq[i];
        torques_out[i] = std::clamp(tau, -effort_limit[i], effort_limit[i]);
    }
}
```

Use the **training kp/kd values** (from `robots/booster_t1/deploy_config.py`), not the booster-motion Lua defaults — the policy was trained with those dynamics.

### Main Loop

```cpp
CircusClient client("127.0.0.1", 5555, "red_Booster-T1_0");
Policy policy("model.onnx", config);
policy.reset();

float torques[23] = {};

while (true) {
    SensorData state = client.receive();   // blocks on each Circus tick

    // Derive projected gravity from quaternion
    compute_projected_gravity(state.quat, state.projected_gravity);

    // Derive linear velocity (see below)
    estimate_lin_vel(state, /* prev_state, dt */ ...);

    // Run policy
    float q_target[23];
    policy.inference(state, vx_cmd, vy_cmd, vyaw_cmd, q_target);

    // PD → torques
    pd_control(q_target, state.joint_pos, state.joint_vel,
               config.kp, config.kd, config.effort_limit, torques);

    client.send(torques);
}
```

---

## Base Linear Velocity

Circus's TCP protocol does not send `base_lin_vel` directly. Options in order of preference:

### Option A: Add a MuJoCo velocity sensor to the robot XML (recommended)

Add a `framelinvel` sensor to `resources/robots/Booster-T1/instance.xml`:

```xml
<sensor>
    <!-- existing sensors ... -->
    <framelinvel name="base_linvel" objtype="body" objname="Trunk" />
</sensor>
```

Then extend the Circus protocol to include this sensor value in the outgoing packet. This is the cleanest solution and only requires a small Circus change.

### Option B: Differentiate pose position

```cpp
// Client-side, tracks previous position
void estimate_lin_vel(const SensorData& cur, const SensorData& prev,
                      float dt, float lin_vel_w[3]) {
    for (int i = 0; i < 3; i++)
        lin_vel_w[i] = (cur.position[i] - prev.position[i]) / dt;

    // Rotate world-frame velocity into body frame using quaternion inverse
    rotate_by_quat_inverse(cur.quat, lin_vel_w, cur.lin_vel);
}
```

This works but is noisy at low control rates. Apply a low-pass filter if needed.

### Option C: Zero (acceptable for many locomotion policies)

If the policy was trained with `root_lin_vel_b` and the training used `rt/low_state` (which also doesn't have linear velocity natively and may have used zeros or odometry), simply zero it:

```cpp
state.lin_vel[0] = state.lin_vel[1] = state.lin_vel[2] = 0.0f;
```

Check what the training environment actually provided by inspecting the `base_lin_vel` observation wrapper in `tasks/velocity/mdp/wrappers.py`.

---

## Launch Flow

Without Docker, launching is trivial:

```bash
# Terminal 1: start Circus in no-containers mode
circus --scene configs/scenes/training_1v1.yaml \
       --simulation-config configs/simulation/training.yaml \
       --no-containers

# Terminal 2: start the policy node
./policy_node --model models/t1-velocity-flat_ppo_latest.onnx \
              --robot red_Booster-T1_0
```

Or as pixi tasks in `sim2sim/pixi.toml`:

```toml
[tasks.circus]
cmd = "circus --scene configs/scenes/training_1v1.yaml --no-containers"

[tasks.policy]
cmd = "main --model ../../models/t1-velocity-flat_ppo_latest.onnx"

[tasks.launch]
depends-on = ["circus"]  # start circus first, then run policy
cmd = "main ..."
```

---

## sim2sim pixi.toml for No-Booster Mode

Only `circus` is needed. No `simbridge`, no `booster_robotics_sdk_ros2`:

```toml
[workspace]
platforms = ["linux-64", "linux-aarch64"]
channels = [
    "https://prefix.dev/pixi-build-backends",
    "https://prefix.dev/conda-forge",
    "https://prefix.dev/spqr",
]
preview = ["pixi-build"]

[dependencies]
sim2sim = { path = "." }
circus  = "*"   # from spqr channel — the only runtime dependency

[package]
name = "sim2sim"
version = "0.1.0"

[package.build.backend]
name = "pixi-build-cmake"
version = "*"

[package.host-dependencies]
onnxruntime-cpp = "*"
eigen           = "*"
msgpack-cxx     = "*"   # for TCP/msgpack communication with Circus

[package.run-dependencies]
onnxruntime-cpp = "*"
msgpack-cxx     = "*"
```

No Booster SDK dependency at all.

---

## Comparison: When to Use Which Approach

| Criterion | No-booster (this doc) | Booster stack |
|---|---|---|
| Dependencies | circus only | circus + simbridge + booster SDK |
| Docker required | No | Yes |
| Setup complexity | Low | High |
| Parallel mech handling | Manual (serial joints, no conversion needed — MuJoCo models serial joints) | Automatic (booster-motion) |
| Real robot portability | No — Circus-specific client | Yes — same code runs on T1 |
| Multi-robot | Yes — one client per robot | Yes |
| Suitable for | Fast iteration, research, multi-robot sim, Circus release | Real robot deployment path |

---

## Toward a Standalone Circus Release

This no-booster setup defines the minimal API surface Circus needs to expose as a standalone simulator. A formal Circus release requires:

### 1. CLI arguments (see `LAUNCH_ARCHITECTURE.md`)

```
--scene <path>              Scene YAML file
--simulation-config <path>  Simulation config YAML (optional, has defaults)
--no-containers             Start TCP server only, no Docker spawning
--headless                  Run without Qt GUI (for CI/automated testing)
--port <n>                  TCP server port (default: 5555)
```

### 2. Stable protocol versioning

Add a version handshake after the robot name:
```
Client → Server: msgpack({"version": 1, "robot_name": "red_Booster-T1_0"})
Server → Client: msgpack({"version": 1, "ok": true})
```

### 3. Optional: `circus-client` helper library

A minimal C++ header-only library (`circus_client.hpp`) that handles:
- TCP connection + framing
- msgpack serialize/deserialize
- `SensorData` struct matching the protocol

Packaged as a conda package (`circus-client`) so any application can depend on it without depending on the full Circus build.

### 4. Scene format stability

The scene YAML format (`teams`, `type`, `position`, `orientation`) should be considered a stable public API and versioned accordingly.

### 5. Remove hardcoded `PROJECT_ROOT` paths

`SceneParser.cpp` currently resolves robot XMLs relative to `PROJECT_ROOT` (the compile-time source directory). For a released conda package, paths should be resolved relative to the installed prefix (`$CONDA_PREFIX/share/circus/`).
