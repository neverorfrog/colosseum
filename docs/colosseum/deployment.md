# Deployment (Arena)

Deploying a trained policy to the Booster T1 involves two steps:

1. **Export** — convert the PyTorch checkpoint to ONNX from colosseum (Python).
2. **Run** — load the ONNX model and drive the robot with arena (C++).

Arena is the C++ runtime that runs on both a development PC (MuJoCo sim)
and directly on the robot (Booster SDK).

---

## Step 1 — Export to ONNX

From the colosseum repo, after training:

```bash
pixi run export-onnx task:t1-dribbling \
    --checkpoint ./logs/<run>/checkpoints/latest.pt \
    --output-dir ./models/
```

This traces the policy network (actor + privileged encoder frozen in Phase 1
weights, or adaptation encoder for Phase 2) and writes a `.onnx` file.

The exported model has a fixed input shape matching the training observation
layout (e.g. 82 floats for T1 23-DOF velocity flat). The arena `Policy`
class validates this at load time.

---

## Step 2 — Arena overview

Arena is a standalone C++ executable that implements the policy control loop.
It lives in its own repository ([github.com/neverorfrog/arena](https://github.com/neverorfrog/arena)) and consumes colosseum's exported ONNX models, as well as policy contracts defined in language-agnostic YAML files.

### Backends

Arena supports three backends, selected at runtime with `--backend`:

| Backend | Flag | Use case |
|---|---|---|
| `mujoco` | `--backend mujoco` | Sim-to-sim testing on dev PC (requires MuJoCo + GLFW build) |
| `booster` | `--backend booster` | Real Booster T1 robot (via Booster SDK / DDS) |
| `circus` | `--backend circus` | Remote robot via the Circus bridge |

### Running in simulation

Build arena with MuJoCo support (`mujoco` and `glfw` present in the pixi env):

```bash
pixi run compile          # CMake build, outputs arena binary
pixi run arena --backend mujoco --task t1-velocity-flat
```

A GLFW window opens showing the T1 in its default XML scene. The policy
runs at 50 Hz; physics steps at 200 Hz (decimation = 4). You can interact
with the robot using a joystick or keyboard.

### Running on the robot

```bash
# On dev PC: build and deploy to robot
scripts/deploy.sh --ip 192.168.10.102

# On robot:
cd ~/arena && ./run.sh run --task t1-velocity-flat
```

`deploy.sh` builds an aarch64 conda package, transfers it to the robot over
SSH, and installs it into a self-contained pixi environment. No dev
dependencies are needed on the robot.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        main.cpp                         │
│  parse flags → TaskRegistry::create() → IPortal::run()  │
└───────────┬─────────────────────────┬───────────────────┘
            │                         │
     ┌──────▼──────┐           ┌──────▼──────────┐
     │   Policy    │           │    IPortal      │
     │ (per-task)  │           │  (per-backend)  │
     └──────┬──────┘           └──────┬──────────┘
            │                         │
     ┌──────▼──────┐    ┌─────────────▼────────────────┐
     │IInference   │    │ MujocoPortal  │  RobotPortal │
     │Engine       │    │ (sim-to-sim)  │  (real robot)│
     │(ONNX / TRT) │    └───────────────┴──────────────┘
     └─────────────┘
```

### IPortal

The portal abstracts the hardware backend. Its interface is:

```cpp
class IPortal {
    void initialize();             // one-time setup
    bool hasState() const;         // true once first state arrives
    const RobotState& getState();  // joint pos/vel, IMU, projected gravity
    void updateState();            // re-read from source
    void publishCommand(          // send PD targets
        const float* targets,
        const float* kp,
        const float* kd);
    void tick();                   // sleep until next 50 Hz tick (+ physics steps for MuJoCo)
    bool shouldContinue() const;   // false when viewer window is closed
};
```

**MujocoPortal** loads the robot XML, merges it with a scene XML, sets
armature and PD gains programmatically (guaranteeing they match training),
and renders via GLFW.

**RobotPortal** connects to the Booster SDK via DDS, puts the robot into
Custom mode, runs the `prepare` pose for 3 s, then hands control to the
policy loop.

### Policy

`Policy` is the abstract C++ base class for task-specific inference:

```cpp
class Policy {
    void reset();   // zero last_action before first step
    std::array<float, NUM_JOINTS> get_action(const RobotState& state);

protected:
    virtual void update_input() {}                          // read joystick/keyboard
    virtual void build_observation(const RobotState&) = 0; // fill `observation`

    TaskConfig config_;
    RobotData<NUM_JOINTS> robot_data_;   // handles sim↔hardware joint remapping
    float last_action[NUM_JOINTS]{};     // in sim order, used next step
    std::vector<float> observation;
    std::unique_ptr<IInferenceEngine> engine_;
    std::unique_ptr<IInputSource> input_source_;
};
```

Subclasses implement `build_observation()` to match the exact observation
layout used during Python training. The sim→hardware joint remapping is
handled by `RobotData` using the `sim2real` index array computed from
`RobotConfig::sim_joint_names` vs `joint_names`.

### TaskConfig and RobotConfig

`TaskConfig` carries task-level settings:

```cpp
struct TaskConfig {
    std::string task_name;
    std::string model_path;         // path to .onnx
    float       policy_dt = 0.02f; // 50 Hz
    float       action_scale = 0.25f;
    std::string inference_backend = "onnx";  // or "trt"
    RobotConfig<NUM_JOINTS> robot;
    std::string scene_mjcf_path;
};
```

`RobotConfig` holds hardware specs: joint names (hardware order and MuJoCo
compiled order), PD gains, default pose, effort limits, armature values, and
the prepare-state configuration used before switching to Custom mode.

### Inference engines

The `IInferenceEngine` interface supports two backends:

- **ONNX Runtime** (`OnnxInferenceEngine`) — default, works everywhere.
- **TensorRT** — optional, enabled at build time if `nvinfer` is found. Gives
  faster inference on NVIDIA hardware (relevant on the robot's Jetson).

Select at runtime: `arena --inference trt --task t1-velocity-flat`

### Action decoding

The policy network outputs actions in **MuJoCo (sim) joint order**. Arena
decodes them to hardware targets:

```
target[i] = net_out[sim2real[i]] * action_scale + default_joint_pos[i]
```

For the T1, `sim2real` is the identity (MuJoCo's alphabetical order matches
hardware order), so the permutation is a no-op.

---

## Control loop

```cpp
policy->reset();
while (!terminated && portal->shouldContinue()) {
    portal->updateState();
    auto targets = policy->get_action(portal->getState());
    portal->publishCommand(targets.data(), cfg.robot.joint_stiffness.data(),
                           cfg.robot.joint_damping.data());
    portal->tick();  // sleeps until next 50 Hz slot (MuJoCo: also steps physics)
}
```

`tick()` on `RobotPortal` is a simple `sleep_until` to the next 20 ms
boundary. On `MujocoPortal` it additionally runs `decimation` MuJoCo physics
steps and syncs the GLFW viewer.

---

## Adding a new task

1. Create a subclass of `Policy` in `src/tasks/<task_name>/`:
   - Implement `build_observation()` matching the Python training layout.
   - Optionally override `update_input()` to read velocity commands from
     joystick/keyboard.
2. Register the task with `TaskRegistry`:
   ```cpp
   REGISTER_TASK("my-task", MyPolicy, MyTaskConfig{});
   ```
3. Export the ONNX model from colosseum and point `TaskConfig::model_path`
   at it (or use `ModelRegistry` for automatic path resolution).
4. Build arena and test with `--backend mujoco` before deploying to hardware.
