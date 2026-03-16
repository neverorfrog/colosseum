# Sim2Sim Integration Plan: RL Policy in Circus

This document describes the plan for integrating a trained RL policy into the Circus simulation pipeline, and later into the real robot via spqrbooster2026.

## Goal

Build a standalone C++ executable that:
1. Reads sensor data from `rt/low_state` (published by booster-motion)
2. Reads joystick input from `rt/remote_controller_state`
3. Runs ONNX policy inference at 50Hz
4. Publishes joint position targets to `rt/joint_ctrl` (consumed by booster-motion)

This executable plugs into the existing **Circus + SimBridge + booster-motion** stack. The same code (or a module version of it) will later run on the real robot.

## Prerequisites

The following must already be running before launching the policy node:

```
1. Circus        — MuJoCo physics simulator (TCP server on port 5555)
2. SimBridge     — TCP↔DDS bridge
3. booster-motion — low-level PD controller (publishes rt/low_state, subscribes rt/joint_ctrl)
```

These three components are already integrated and data flows between them.

## Architecture

### Location

```
src/sim2sim/                    # Standalone C++ project (NOT in src/colosseum/)
├── CMakeLists.txt
├── pixi.toml                   # pixi-managed C++ dependencies
├── include/
│   ├── robot_portal.hpp        # DDS communication (subscribe + publish)
│   ├── policy.hpp              # ONNX model loading and inference
│   ├── robot_data.hpp          # Sensor state container
│   └── config.hpp              # Robot constants (joints, gains, defaults)
└── src/
    ├── main.cpp                # Entry point and control loop
    ├── robot_portal.cpp
    ├── policy.cpp
    └── config.cpp
```

The executable accesses ONNX models from the repository root: `../../models/` (e.g., `t1-velocity-flat_ppo_latest.onnx`).

### Component Design

#### RobotData (State Container)

Holds the latest sensor state, populated by DDS callbacks.

```cpp
struct RobotData {
    // From LowState.imu_state
    float rpy[3];            // roll, pitch, yaw (rad)
    float gyro[3];           // angular velocity in body frame (rad/s)
    float acc[3];            // linear acceleration (m/s^2)

    // From LowState.motor_state_serial
    float joint_pos[23];     // joint positions (rad)
    float joint_vel[23];     // joint velocities (rad/s)
    float feedback_torque[23]; // estimated torques (Nm)

    // Derived
    float projected_gravity[3]; // gravity vector in body frame
    float root_quat[4];        // quaternion from RPY (w,x,y,z)
};
```

#### RobotPortal (Communication)

Handles all DDS communication using the Booster SDK's ChannelPublisher/ChannelSubscriber.

```cpp
class RobotPortal {
public:
    RobotPortal();
    ~RobotPortal();

    void init();
    void shutdown();

    // Read latest state (populated by callback)
    const RobotData& get_state() const;

    // Read joystick (populated by callback)
    float get_vx_cmd() const;   // [-1, 1]
    float get_vy_cmd() const;
    float get_vyaw_cmd() const;

    // Publish joint position targets
    void publish_command(const float* joint_targets,   // [23] rad
                         const float* kp,              // [23]
                         const float* kd);             // [23]

    // Mode management
    void change_mode(RobotMode mode);

private:
    // DDS channels
    ChannelSubscriber<LowState> low_state_sub_;        // rt/low_state
    ChannelSubscriber<RemoteControllerState> rc_sub_;   // rt/remote_controller_state
    ChannelPublisher<LowCmd> joint_ctrl_pub_;           // rt/joint_ctrl
    B1LocoClient loco_client_;

    // State (updated by callbacks, read by main loop)
    RobotData state_;
    std::mutex state_mutex_;

    // Joystick (updated by callback)
    float vx_raw_, vy_raw_, vyaw_raw_;
    std::mutex rc_mutex_;

    void on_low_state(const LowState& msg);
    void on_remote_controller(const RemoteControllerState& msg);
};
```

**DDS topics used:**

| Direction | Topic | Type | Purpose |
|-----------|-------|------|---------|
| Subscribe | `rt/low_state` | `LowState` | Sensor feedback (IMU + joints) |
| Subscribe | `rt/remote_controller_state` | `RemoteControllerState` | Joystick input |
| Publish | `rt/joint_ctrl` | `LowCmd` | Joint position targets (q, kp, kd) |

#### Policy (ONNX Inference)

Loads the ONNX model and runs inference.

```cpp
class Policy {
public:
    Policy(const std::string& onnx_path, const Config& config);
    ~Policy();

    // Reset policy state (call before starting control loop)
    void reset();

    // Run one inference step, returns joint position targets [23]
    void inference(const RobotData& state,
                   float vx_cmd, float vy_cmd, float vyaw_cmd,
                   float* joint_targets_out);  // [23] output

private:
    // ONNX Runtime session (details TBD)
    // ...

    // Config
    const Config& config_;

    // State
    float last_action_[23];       // stored normalized action from previous step
    float observation_[82];       // reusable observation buffer

    // Build observation vector in the exact training order
    void build_observation(const RobotData& state,
                           float vx_cmd, float vy_cmd, float vyaw_cmd);

    // Convert normalized action → joint position targets
    void action_to_targets(const float* action, float* targets);
};
```

**Observation construction** (must match training exactly):

```cpp
void Policy::build_observation(const RobotData& state,
                                float vx_cmd, float vy_cmd, float vyaw_cmd) {
    int idx = 0;

    // base_lin_vel (3) — from IMU or derived
    // NOTE: rt/low_state may not provide linear velocity directly.
    // Options: use zeros, derive from odometry, or use accelerometer integration.
    // This must match what the training environment provides.
    observation_[idx++] = ???;  // vx body frame
    observation_[idx++] = ???;  // vy body frame
    observation_[idx++] = ???;  // vz body frame

    // base_ang_vel (3) — from gyroscope
    observation_[idx++] = state.gyro[0];
    observation_[idx++] = state.gyro[1];
    observation_[idx++] = state.gyro[2];

    // projected_gravity (3) — computed from RPY → quaternion → rotate [0,0,-1]
    observation_[idx++] = state.projected_gravity[0];
    observation_[idx++] = state.projected_gravity[1];
    observation_[idx++] = state.projected_gravity[2];

    // joint_pos_rel (23) — joint positions minus default pose
    for (int i = 0; i < 23; i++) {
        observation_[idx++] = state.joint_pos[i] - config_.default_joint_pos[i];
    }

    // joint_vel (23) — joint velocities
    for (int i = 0; i < 23; i++) {
        observation_[idx++] = state.joint_vel[i];
    }

    // last_action (23) — previous normalized action
    for (int i = 0; i < 23; i++) {
        observation_[idx++] = last_action_[i];
    }

    // velocity_commands (3) — scaled by max velocities
    observation_[idx++] = vx_cmd;
    observation_[idx++] = vy_cmd;
    observation_[idx++] = vyaw_cmd;
}
```

**Action conversion:**

```cpp
void Policy::action_to_targets(const float* action, float* targets) {
    for (int i = 0; i < 23; i++) {
        last_action_[i] = action[i];  // store for next observation
        targets[i] = action[i] * config_.action_scale + config_.default_joint_pos[i];
    }
}
```

#### Config (Robot Constants)

```cpp
struct Config {
    // Joint configuration
    static constexpr int NUM_JOINTS = 23;
    float default_joint_pos[NUM_JOINTS];  // standing pose (rad)
    float joint_stiffness[NUM_JOINTS];    // kp values for LowCmd
    float joint_damping[NUM_JOINTS];      // kd values for LowCmd
    float action_scale;                   // 0.25

    // Velocity command limits
    float vx_max;     // max forward velocity (m/s)
    float vy_max;     // max lateral velocity (m/s)
    float vyaw_max;   // max yaw rate (rad/s)

    // Policy
    float policy_dt;  // 0.02s (50Hz)
};
```

Values must be extracted from the training configuration (`src/colosseum/tasks/velocity/config/t1_23dof/env_cfgs.py` and `src/colosseum/robots/t1_23dof/deploy.py`).

#### Main Loop

```cpp
int main(int argc, char** argv) {
    // Parse args (model path, config)
    std::string model_path = "../../models/t1-velocity-flat_ppo_latest.onnx";

    Config config = load_t1_velocity_config();
    RobotPortal portal;
    Policy policy(model_path, config);

    portal.init();

    // Wait for first low_state message
    wait_for_state(portal);

    // Switch to custom mode (allows direct joint control)
    portal.change_mode(RobotMode::kCustom);

    // Initialize policy
    policy.reset();

    // Control loop at 50Hz
    float joint_targets[23];
    auto next_tick = std::chrono::steady_clock::now();

    while (running) {
        // Wait for next policy tick
        next_tick += std::chrono::microseconds(
            static_cast<int>(config.policy_dt * 1e6));
        std::this_thread::sleep_until(next_tick);

        // Read latest state
        const RobotData& state = portal.get_state();

        // Read joystick commands (scaled to [-max, +max])
        float vx = portal.get_vx_cmd() * config.vx_max;
        float vy = portal.get_vy_cmd() * config.vy_max;
        float vyaw = portal.get_vyaw_cmd() * config.vyaw_max;

        // Run policy inference
        policy.inference(state, vx, vy, vyaw, joint_targets);

        // Publish to booster-motion
        portal.publish_command(joint_targets,
                               config.joint_stiffness,
                               config.joint_damping);
    }

    portal.shutdown();
    return 0;
}
```

### Build System

**pixi.toml** (pixi-managed C++ project):
```toml
[project]
name = "sim2sim"
version = "0.1.0"
platforms = ["linux-64"]

[dependencies]
booster_robotics_sdk = "==1.5.0"
eigen = "*"
cmake = ">=3.20"
# onnxruntime — TBD (conda-forge onnxruntime-cpp or manual download)
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.20)
project(sim2sim LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(booster_robotics_sdk REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(onnxruntime REQUIRED)  # TBD

add_executable(sim2sim
    src/main.cpp
    src/robot_portal.cpp
    src/policy.cpp
    src/config.cpp
)

target_include_directories(sim2sim PRIVATE include)
target_link_libraries(sim2sim PRIVATE
    booster::booster_robotics_sdk
    Eigen3::Eigen
    # onnxruntime  # TBD
)
```

## Open Items

### 1. ONNX Runtime C++ Integration

How to get ONNX Runtime for C++. Options to evaluate:
- `conda-forge::onnxruntime-cpp` via pixi
- Manual download of ONNX Runtime release binaries
- TensorRT conversion (available in spqrbooster2026 already) as alternative

### 2. Base Linear Velocity

The observation includes `base_lin_vel` (3D linear velocity in body frame). On the real robot and in simulation via booster-motion, `rt/low_state` provides IMU data (RPY, gyro, acc) but **may not directly provide linear velocity**. Options:
- Check if the training observation actually uses linear velocity or if it's zeroed
- Derive from odometry (`rt/odometer_state`)
- Integrate accelerometer data
- Match whatever the MuJoCo sim-to-sim backend does (which reads `qvel[:3]`)

This must match exactly what the policy saw during training.

### 3. Joint Order Verification

The joint order in `rt/low_state.motor_state_serial` must match the order used during training. The T1 23-DOF joint order appears to be identical between hardware and simulation (no remapping needed), but this must be verified by comparing:
- Circus joint names (from `ROBOT_JOINTS_NAMES_MAP`)
- booster-motion's motor_state_serial ordering
- Training environment's joint ordering (`sim_joint_names` in deploy config)

### 4. Projected Gravity Computation

Must implement the exact same quaternion math as training:
```
1. Convert RPY (roll, pitch, yaw) → quaternion (w, x, y, z)
2. Rotate gravity vector [0, 0, -1] by inverse quaternion
3. Result = projected gravity in body frame
```

Verify the quaternion convention (scalar-first vs scalar-last) matches between the Booster SDK and the training code.

### 5. Kp/Kd Values in LowCmd

The policy publishes kp/kd values per joint in the LowCmd. These tell booster-motion what PD gains to use. Two options:
- Use the **training kp/kd** values (from colosseum's robot config)
- Use **booster-motion's default kp/kd** (from Lua config)

These may differ. Using training kp/kd is more correct since the policy was trained with those dynamics. Set `weight=1.0` in the MotorCmd to override booster-motion's defaults.

### 6. Mode Transition Sequence

Before sending joint targets, the robot must be in `kCustom` mode:
1. Wait for booster-motion to publish first `rt/low_state`
2. Call `B1LocoClient::ChangeMode(RobotMode::kCustom)`
3. Optionally send a "prepare pose" (move joints to default position slowly)
4. Then start the policy loop

The prepare pose transition avoids sudden jumps when the policy starts.

## Cleanup: Remove Python Booster Backend

Once the C++ sim2sim executable is working, `src/colosseum/deploy/backends/booster.py` becomes redundant and can be removed. This simplifies the Python deploy pipeline:

- **Keep**: `backends/mujoco.py` — lightweight sim-to-sim for fast local iteration (self-contained, no external dependencies)
- **Remove**: `backends/booster.py` — 573 lines of multi-process ROS2 code, shared memory buffers, signal handlers, etc.

The Python deploy code stays focused on what it does best (quick MuJoCo testing), while the C++ sim2sim handles the realistic booster-motion pipeline that maps directly to real hardware.

## Phase 2: Integration into spqrbooster2026

After the standalone sim2sim executable works, the same logic can be ported into spqrbooster2026 as a Module:

```cpp
class RLPolicyModule : public Module {
    Input<ImuState> imuState;          // From Bridge
    Input<JointState> jointState;      // From Bridge (new output needed)
    Input<Joystick> joystick;          // From Bridge
    Output<RCJointTargets> targets;    // New type → BoosterController

    // Holds ONNX model, observation buffer, last_action
    Policy policy_;

    void update() override {
        // Build observation from inputs
        // Run inference
        // Write to output
    }
};
```

This would replace the current `RCVelocity`-based flow (which uses booster-motion's built-in gait) with direct joint-level control from the RL policy.

**Changes needed in spqrbooster2026:**
- Bridge: Add `Output<JointState>` that exposes raw joint positions/velocities from `rt/low_state`
- BoosterController: Add handler for `Input<RCJointTargets>` that publishes LowCmd via BoosterRobotPortal
- BoosterRobotPortal: Already has `setArmStiffness()` pattern for direct joint control — extend to all joints
- New: `RLPolicyModule` with ONNX inference

## Testing Plan

### Step 1: Verify DDS connectivity
Build a minimal program that subscribes to `rt/low_state` and prints IMU + joint data. Confirm data flows from Circus → SimBridge → booster-motion → our subscriber.

### Step 2: Publish static pose
Publish a LowCmd with the default standing pose (all joints at default_joint_pos). Confirm the robot holds the pose in Circus.

### Step 3: Run policy with zero velocity
Run the full policy with `vx=vy=vyaw=0`. The robot should stand still (or exhibit minor balance corrections).

### Step 4: Joystick control
Enable joystick input. Walk the robot around the Circus scene using the gamepad.

### Step 5: Validate against Python sim2sim
Compare joint targets output from the C++ policy node with the Python MuJoCo sim-to-sim backend to verify numerical equivalence.
