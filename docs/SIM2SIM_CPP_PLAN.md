# C++ Sim2Sim Implementation Plan

This document is the detailed implementation plan for the `src/sim2sim` executable,
synthesized from the design discussion and the Python deployment system
(`src/colosseum/deploy/`).

---

## Goal

A standalone C++ executable that:
1. Reads sensor data from `rt/low_state` (DDS, published by booster-motion)
2. Reads joystick input from `/dev/input/jsX`
3. Runs ONNX policy inference at 50 Hz
4. Publishes joint position targets to `rt/joint_ctrl` (consumed by booster-motion)

It must support **multiple tasks** identified by a string name (e.g. `"t1-velocity-flat"`),
mirroring the Python `TaskRegistry` + `@register_task` pattern.

---

## Current State

Already implemented:
- `Policy.h / Policy.cpp` — ONNX session loading and `infer(obs) → action`
- `RobotPortal.h / RobotPortal.cpp` — DDS subscription (LowState), joystick loop,
  RPY→quat and projected gravity computation
- `run.cpp` — skeleton only (prints state, sleeps 1 s)
- `main.cpp` — `dlopen` bootstrap that calls `run()`

**Nothing** in the actual control loop is wired yet: no observation building,
no inference call, no LowCmd publishing.

---

## New File Structure

```
src/sim2sim/
├── include/
│   ├── RobotState.h          (existing)
│   ├── Policy.h              (existing)
│   ├── RobotPortal.h         (existing — needs additions, see below)
│   ├── config.hpp            (NEW) TaskConfig struct
│   ├── policy_base.hpp       (NEW) Abstract PolicyBase + REGISTER_TASK macro
│   └── task_registry.hpp     (NEW) TaskRegistry singleton
├── src/
│   ├── main.cpp              (modify — pass argc/argv to run)
│   ├── run.cpp               (modify — full control loop)
│   ├── Policy.cpp            (existing)
│   ├── RobotPortal.cpp       (modify — add publish_command, has_state, mutex)
│   ├── policy_base.cpp       (NEW) PolicyBase::reset(), step()
│   ├── task_registry.cpp     (NEW) TaskRegistry singleton impl
│   └── tasks/
│       └── t1_velocity.cpp   (NEW) TaskConfig values + T1VelocityPolicy + REGISTER_TASK
```

`CMakeLists.txt` already uses `file(GLOB_RECURSE SOURCES ... src/*.cpp)`,
which recurses into subdirectories — `src/tasks/t1_velocity.cpp` is picked up
automatically with no CMake changes.

---

## Component Designs

### 1. `include/config.hpp` — TaskConfig

Mirrors Python's `ControllerConfig` + `RobotConfig` frozen dataclasses.

```cpp
struct TaskConfig {
    static constexpr int NUM_JOINTS = 23;

    std::string task_name;
    std::string model_path;   // absolute or relative to executable

    float policy_dt = 0.02f;  // 50 Hz

    // Per-joint action scale = policy_action_scale * effort_limit / stiffness
    // (matches Python Policy.__init__: self.action_scale = cfg.action_scale
    //                                                     * robot.effort_limit
    //                                                     / robot.joint_stiffness)
    // See "Action Decoding" section for precomputed values.
    std::array<float, NUM_JOINTS> action_scale;

    // Default standing pose (rad) — subtracted in joint_pos_rel observation,
    // added back after action decoding. Source: deploy.py default_joint_pos
    std::array<float, NUM_JOINTS> default_joint_pos;

    // PD gains sent in every LowCmd. Source: deploy.py joint_stiffness/damping
    std::array<float, NUM_JOINTS> kp;
    std::array<float, NUM_JOINTS> kd;

    // Velocity command limits (m/s or rad/s)
    float vx_max   = 1.0f;
    float vy_max   = 1.0f;
    float vyaw_max = 1.0f;
};
```

### 2. `include/task_registry.hpp` — TaskRegistry

A purpose-built singleton for tasks — simpler than the generic `spqr::Registry<Base>`
(no CRTP, no type_index) since the factory always returns `unique_ptr<PolicyBase>` with no args.
Inspired by the `REGISTER_MODULE` pattern in spqrbooster.

```cpp
class TaskRegistry {
public:
    using Factory = std::function<std::unique_ptr<PolicyBase>()>;

    static TaskRegistry& instance();

    void register_task(const std::string& name, Factory factory);
    std::unique_ptr<PolicyBase> create(const std::string& name) const;
    bool has(const std::string& name) const;

    // Nested helper: construct one of these as a static local variable
    // to register a task before main() runs.
    struct Registrar {
        Registrar(const std::string& name, Factory factory) {
            TaskRegistry::instance().register_task(name, std::move(factory));
        }
    };

private:
    TaskRegistry() = default;
    std::unordered_map<std::string, Factory> factories_;
};

// Convenience macro — mirrors REGISTER_MODULE from spqrbooster.
// Place once per task .cpp file (outside any namespace/class).
#define REGISTER_TASK(task_name, ClassName)                                 \
    static TaskRegistry::Registrar __##ClassName##_registrar(               \
        task_name,                                                           \
        []() -> std::unique_ptr<PolicyBase> {                               \
            return std::make_unique<ClassName>();                            \
        }                                                                    \
    );
```

`task_registry.cpp` just implements the singleton body and the two methods
(`register_task`, `create`, `has`).

### 3. `include/policy_base.hpp` + `src/policy_base.cpp` — PolicyBase

Abstract base class that owns the ONNX runner and `last_action_`.
Subclasses implement only `build_observation()`.

**Header:**
```cpp
class PolicyBase {
public:
    explicit PolicyBase(TaskConfig cfg);
    virtual ~PolicyBase() = default;

    void reset();     // zeros last_action_
    std::array<float, TaskConfig::NUM_JOINTS> step(const RobotState& state);

    const TaskConfig& config() const { return config_; }

protected:
    TaskConfig config_;
    float last_action_[TaskConfig::NUM_JOINTS]{};

    // Subclass fills obs[0..OBS_DIM-1].
    // OBS_DIM = 82 for T1 23-DOF velocity task.
    virtual void build_observation(const RobotState& state, float* obs) = 0;

private:
    Policy onnx_;  // existing Policy.h class
};
```

**`step()` implementation (policy_base.cpp):**
```
1. call build_observation(state, obs_buf)          // fills float[82]
2. obs_vec = Eigen::Map<VectorXf>(obs_buf, 82)
3. action  = onnx_.infer(obs_vec)                  // Eigen::VectorXf, shape (23,)
4. store action → last_action_                     // for next step's obs
5. targets[i] = action[i] * config_.action_scale[i] + config_.default_joint_pos[i]
6. return targets
```

### 4. `src/tasks/t1_velocity.cpp` — T1VelocityPolicy

Fully self-contained (no header). Defines the task config and observation logic,
registers itself via `REGISTER_TASK`.

#### `build_observation()` — exact 82-dim vector

Mirrors `T1VelocityPolicy.compute_observation()` in
`tasks/velocity/deploy/t1/policy.py` and `VELOCITY_OBS_SPEC` in
`tasks/velocity/mdp/observation_spec.py`.

```
Index  Size  Field             Source
-----  ----  ----------------  -----------------------------------------------
 0      3    base_lin_vel      zeros  (LowState has no linear velocity)
 3      3    base_ang_vel      state.gyro[0..2]
 6      3    projected_gravity state.projected_gravity[0..2]  (already computed
                               by RobotPortal::lowStateCallback)
 9     23    joint_pos_rel     state.joint_pos[i] - config_.default_joint_pos[i]
32     23    joint_vel         state.joint_vel[0..22]
55     23    last_action       last_action_[0..22]
78      3    vel_commands      state.v[0], state.v[1], state.v[2]
                               (set by joystick loop: vx, vy, vyaw — already scaled
                               by VX_MAX/VY_MAX/VYAW_MAX in RobotPortal::joystickLoop)
-----
       82    total
```

> **Joint order note:** For T1 23-DOF, `joint_names == sim_joint_names` (identical order,
> verified from `deploy.py`). No remapping is needed. `state.joint_pos[i]` maps directly
> to policy dimension `i`.

> **base_lin_vel note:** Python deploy also uses zeros here —
> `root_lin_vel_w = np.zeros(3)` in `booster.py:_low_state_handler`.
> This matches training if the training observation was zeroed (open item).

#### `make_config()` — T1 values

All values from `robots/t1_23dof/deploy.py` and
`tasks/velocity/deploy/t1/config.py`:

```cpp
cfg.policy_dt = 0.02f;

cfg.default_joint_pos = {
    0.0f, 0.0f,                           // AAHead_yaw, Head_pitch
    0.2f, -1.3f, 0.0f, -0.5f,            // Left arm
    0.2f,  1.3f, 0.0f,  0.5f,            // Right arm
    0.0f,                                  // Waist
    -0.2f, 0.0f, 0.0f, 0.4f, -0.2f, 0.0f, // Left leg
    -0.2f, 0.0f, 0.0f, 0.4f, -0.2f, 0.0f, // Right leg
};

cfg.kp = {
    5.0f, 5.0f,                                          // Head
    20.0f, 20.0f, 20.0f, 20.0f,                          // Left arm
    20.0f, 20.0f, 20.0f, 20.0f,                          // Right arm
    200.0f,                                               // Waist
    200.0f, 200.0f, 200.0f, 200.0f, 50.0f, 50.0f,       // Left leg
    200.0f, 200.0f, 200.0f, 200.0f, 50.0f, 50.0f,       // Right leg
};

cfg.kd = {
    0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, 0.5f,
    5.0f,
    5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 3.0f,
    5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 3.0f,
};

cfg.vx_max = 1.0f; cfg.vy_max = 1.0f; cfg.vyaw_max = 1.0f;
```

#### Action Decoding — per-joint `action_scale`

> ⚠️ **Critical:** Python `Policy.__init__` computes per-joint scaling:
> `self.action_scale = cfg.action_scale * robot.effort_limit / robot.joint_stiffness`
>
> This is **not** a uniform 0.25. The C++ `action_scale` array must be precomputed
> identically.

```
Joint               effort   stiffness  scale = 0.25 * effort / stiffness
----------------   -------  ---------  ----------------------------------
AAHead_yaw            7.0       5.0     0.3500
Head_pitch            7.0       5.0     0.3500
Left_Shoulder_*      18.0      20.0     0.2250  (×4)
Right_Shoulder_*     18.0      20.0     0.2250  (×4)
Waist                30.0     200.0     0.0375
Left_Hip_Pitch       45.0     200.0     0.0563
Left_Hip_Roll        30.0     200.0     0.0375
Left_Hip_Yaw         30.0     200.0     0.0375
Left_Knee_Pitch      60.0     200.0     0.0750
Left_Ankle_Pitch     12.0      50.0     0.0600
Left_Ankle_Roll      12.0      50.0     0.0600
Right_* (same as left leg)
```

```cpp
cfg.action_scale = {
    0.35f, 0.35f,                                            // Head
    0.225f, 0.225f, 0.225f, 0.225f,                          // Left arm
    0.225f, 0.225f, 0.225f, 0.225f,                          // Right arm
    0.0375f,                                                  // Waist
    0.05625f, 0.0375f, 0.0375f, 0.075f, 0.06f, 0.06f,       // Left leg
    0.05625f, 0.0375f, 0.0375f, 0.075f, 0.06f, 0.06f,       // Right leg
};
```

**Model path** uses the `PROJECT_ROOT` compile definition already set in `CMakeLists.txt`:
```cpp
cfg.model_path = std::string(PROJECT_ROOT) + "/models/t1-velocity-flat_ppo_latest.onnx";
```

**Registration** at end of file:
```cpp
REGISTER_TASK("t1-velocity-flat", T1VelocityPolicy)
```

---

### 5. `RobotPortal` additions

#### a) Thread-safety for `getState()`

The `lowStateCallback` runs on a DDS thread; the main loop calls `getState()` on the
main thread. Add `std::mutex state_mutex_` (private), lock it in both callback and getter.
Return by value in `getState()` to avoid holding the lock during policy inference:

```cpp
RobotState getState() const;   // returns copy under lock
```

#### b) `has_state()` — wait-for-first-message

Add `std::atomic<bool> has_state_{false}` (private), set to `true` in the first call to
`lowStateCallback`. Expose as:

```cpp
bool has_state() const { return has_state_.load(); }
```

`run.cpp` spins on this before changing mode and starting the loop.

#### c) `publish_command()`

New public method that writes a `LowCmd` and calls `low_cmd_pub->Write()`:

```cpp
void publish_command(const float* joint_targets,  // [23] rad
                     const float* kp,             // [23]
                     const float* kd);            // [23]
```

Implementation sketch:
```
booster_interface::msg::LowCmd cmd;
cmd.cmd_type(CMD_TYPE_SERIAL);
for i in [0..22]:
    cmd.motor_cmd()[i].q(joint_targets[i])
    cmd.motor_cmd()[i].kp(kp[i])
    cmd.motor_cmd()[i].kd(kd[i])
    cmd.motor_cmd()[i].dq(0.0f)
    cmd.motor_cmd()[i].tau(0.0f)
    cmd.motor_cmd()[i].weight(1.0f)   // ← overrides booster-motion defaults
low_cmd_pub->Write(cmd)
```

---

### 6. `src/run.cpp` — Full Control Loop

Change signature to accept `argc/argv` (task name from `argv[1]`):

```cpp
extern "C" int run(int argc, char** argv) {
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // 1. Init SDK and RobotPortal
    booster::robot::ChannelFactory::Instance()->Init(0);
    RobotPortal::instance().initialize();

    // 2. Parse task name
    std::string task_name = (argc > 1) ? argv[1] : "t1-velocity-flat";
    auto policy = TaskRegistry::instance().create(task_name);  // throws if unknown
    const TaskConfig& cfg = policy->config();

    // 3. Wait for first low_state message
    std::cout << "Waiting for rt/low_state...\n";
    while (!RobotPortal::instance().has_state() && !terminated)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (terminated) return 0;

    // 4. Switch to custom mode
    if (RobotPortal::instance().changeMode(booster::robot::RobotMode::kCustom) != 0) {
        std::cerr << "Failed to switch to kCustom mode\n";
        return -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 5. Initialize policy
    policy->reset();

    // 6. 50 Hz control loop
    std::array<float, 23> targets;
    auto next_tick = std::chrono::steady_clock::now();

    while (!terminated) {
        next_tick += std::chrono::microseconds(
            static_cast<long>(cfg.policy_dt * 1e6));
        std::this_thread::sleep_until(next_tick);

        targets = policy->step(RobotPortal::instance().getState());
        RobotPortal::instance().publish_command(
            targets.data(), cfg.kp.data(), cfg.kd.data());
    }

    std::cout << "Shutting down...\n";
    return 0;
}
```

### 7. `src/main.cpp` — Pass argc/argv to run

Update `RunFn` typedef and the `run()` call:

```cpp
using RunFn = int (*)(int, char**);
// ...
int ret = run(argc, argv);
```

Also forward `argc` and `argv` from `main(int argc, char** argv)`.

---

## Adding a New Task (Future)

To add `"t1-velocity-rough"` or any other task, only one new file is needed:

```
src/tasks/t1_velocity_rough.cpp
```

It defines a subclass of `PolicyBase` with its own `make_config()` and
`build_observation()`, then calls:

```cpp
REGISTER_TASK("t1-velocity-rough", T1VelocityRoughPolicy)
```

No changes to any existing file. The CMake glob picks it up automatically.

---

## Open Items

### 1. base_lin_vel in observation

Python deploy also zeros this field (`root_lin_vel_w = np.zeros(3)` in
`booster.py:_low_state_handler`). Verify against training: if the training observation
also used zeros (or `rt/odometer_state` was never connected), then zeroing in C++ is
correct and consistent. If training used actual body-frame velocity from MuJoCo `qvel[:3]`,
then sim-to-sim will have a systematic error in this slot.

### 2. Verify action_scale formula against training

The Python `Policy.__init__` computes `action_scale = 0.25 * effort_limit / stiffness`.
Confirm this matches the training action decoder. If training used a flat 0.25, replace
the per-joint array with a uniform value. The per-joint values are listed above.

### 3. LowCmd IDL field accessor names

The exact accessor method names on `booster_interface::msg::LowCmd` (e.g. `.q()`,
`.kp()`, `.weight()`) follow the DDS IDL getter/setter convention used elsewhere in
`RobotPortal.cpp`. Verify against the SDK headers if compilation fails.

### 4. Mode transition timing

The `sleep_for(500ms)` after `changeMode(kCustom)` is copied from the Python deploy.
If the robot doesn't stabilize, a prepare-pose step (move joints slowly from current
position to `default_joint_pos` before starting the policy loop) can be added between
steps 4 and 5 above.
