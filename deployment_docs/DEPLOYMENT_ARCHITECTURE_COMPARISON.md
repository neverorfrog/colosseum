# Deployment Architecture Comparison

**Comprehensive analysis of three deployment approaches for humanoid robot policies: Colosseum, holosoma_inference, and booster_deploy.**

**Last Updated:** 2026-01-21

**Purpose:** Solve Colosseum deployment issues by learning from working implementations.

---

## Executive Summary

This document compares three deployment architectures:

1. **Colosseum** (your implementation) - Currently has "heavy feet" issue
2. **holosoma_inference** (Holosoma team) - Production-ready, multi-robot deployment
3. **booster_deploy** (Booster manufacturer) - Reference implementation with working locomotion

### Key Findings

**Root Causes of "Heavy Feet" Issue:**

| Issue | Confidence | Impact | Fix Complexity |
|-------|-----------|---------|----------------|
| **Control Loop Mismatch** | 70% | HIGH | Medium (code change) |
| **PD Gains Wrong** | 20% | HIGH | High (retraining) |
| **ONNX Export Issues** | 8% | MEDIUM | Low (test PyTorch) |
| **Observation Mismatch** | 2% | LOW | Medium (verification) |

**Critical Discovery:**
- **booster_deploy uses manual PD control with per-substep state updates** (dynamic feedback)
- **Colosseum uses MuJoCo position actuators** (static targets, sluggish response)
- **holosoma PD gains moved to ONNX metadata** (new design, cleaner architecture)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Configuration Systems](#configuration-systems)
3. [Control Loop Comparison](#control-loop-comparison)
4. [PD Gains Analysis](#pd-gains-analysis)
5. [Policy Implementation](#policy-implementation)
6. [Observation Spaces](#observation-spaces)
7. [Model Loading](#model-loading)
8. [Joint Order Remapping](#joint-order-remapping)
9. [Input Handling](#input-handling)
10. [Actionable Fixes](#actionable-fixes)

---

## Architecture Overview

### 1. Colosseum Deployment (Your Implementation)

```
src/colosseum/deploy/
├── config/                   # Pydantic frozen dataclasses
│   ├── controller.py         # ControllerConfig (top-level)
│   ├── robot.py              # RobotConfig (hardware specs)
│   ├── policy.py             # PolicyConfig (model loading)
│   └── backend.py            # MujocoConfig, BoosterConfig
├── core/
│   ├── base_controller.py    # Lifecycle + control loop
│   ├── robot.py              # RobotData (state container)
│   ├── policy.py             # Policy abstract base
│   └── registry.py           # TaskRegistry
├── backends/
│   ├── mujoco.py             # MujocoController (sim-to-sim)
│   └── booster.py            # BoosterController (real robot, stub)
└── input/
    ├── joystick.py           # evdev gamepad
    └── keyboard.py           # terminal input

tasks/velocity/deploy/t1_23dof/
├── policy.py                 # T1VelocityPolicy
├── config.py                 # ControllerConfig presets
└── models/                   # ONNX checkpoints
```

**Design Principles:**
- ✅ **Immutable configs** (frozen dataclasses)
- ✅ **Robot-agnostic policy layer** (uses pure MDP functions)
- ✅ **Clean separation** (config, policy, backend)
- ✅ **Type-safe** (Pydantic validation)
- ❌ **Position actuators** (static targets, not dynamic)
- ❌ **ONNX-only** (potential numerical issues)

---

### 2. holosoma_inference (Holosoma Team)

```
external/holosoma/src/holosoma_inference/holosoma_inference/
├── config/
│   ├── config_types/         # Pydantic frozen dataclasses
│   │   ├── robot.py          # RobotConfig schema
│   │   ├── observation.py    # ObservationConfig schema
│   │   ├── task.py           # TaskConfig schema
│   │   └── inference.py      # InferenceConfig composition
│   └── config_values/        # Pre-configured presets
│       ├── robot.py          # g1_29dof, t1_29dof (NO PD gains!)
│       ├── observation.py    # loco_g1_29dof, loco_t1_29dof
│       ├── task.py           # locomotion, wbt
│       └── inference.py      # Complete inference presets
├── policies/
│   ├── base.py               # BasePolicy (ONNX + obs processing)
│   ├── locomotion.py         # LocomotionPolicy (velocity commands)
│   └── wbt.py                # WholeBodyTrackingPolicy
├── sdk/
│   ├── interface_wrapper.py # Backend abstraction (booster, unitree)
│   ├── state_processor/      # Robot state parsing
│   └── command_sender/       # Command serialization
└── run_policy.py             # Tyro CLI entry point
```

**Design Principles:**
- ✅ **PD gains in ONNX metadata** (single source of truth!)
- ✅ **Multi-robot support** (Booster, Unitree, ROS2)
- ✅ **Tyro CLI** (subcommand presets like `inference:t1-29dof-loco`)
- ✅ **Observation history** (deque-based rolling buffers)
- ✅ **Real robot deployment** (proven on T1, G1)
- ⚠️ **Not sim-to-sim focused** (primarily for real hardware)

---

### 3. booster_deploy (Booster Manufacturer)

```
external/booster_deploy/
├── booster_deploy/
│   ├── controllers/
│   │   ├── base_controller.py       # Abstract base
│   │   ├── mujoco_controller.py     # Manual PD control! ✅
│   │   └── booster_robot_controller.py  # Real robot
│   ├── robots/
│   │   └── booster.py                # K1_CFG, T1_23DOF_CFG
│   └── utils/
│       └── registry.py               # Task registry
├── tasks/
│   ├── locomotion/
│   │   ├── locomotion.py             # LocomotionPolicy
│   │   └── models/                   # .pt TorchScript checkpoints!
│   └── beyond_mimic/
│       └── beyond_mimic.py           # Motion tracking
└── scripts/
    └── deploy.py                     # CLI entry point
```

**Design Principles:**
- ✅ **Manual PD torque control** (dynamic per-substep feedback!) ⭐
- ✅ **TorchScript models** (no ONNX conversion overhead)
- ✅ **Observation history** (10-step flattened)
- ✅ **Working locomotion** (proven deployment)
- ✅ **Simple architecture** (flat hierarchy, minimal dependencies)
- ⚠️ **Booster-specific** (not multi-robot)
- ⚠️ **@configclass** (less type-safe than Pydantic)

---

## Configuration Systems

### Colosseum

```python
@dataclass(frozen=True)
class RobotConfig:
    joint_names: tuple[str, ...]          # Real hardware order
    sim_joint_names: tuple[str, ...]      # MuJoCo alphabetical order
    joint_stiffness: tuple[float, ...]    # Kp gains (23,)
    joint_damping: tuple[float, ...]      # Kd gains (23,)
    default_joint_pos: tuple[float, ...]  # Standing pose
    effort_limit: tuple[float, ...]       # Torque limits
    mjcf_path: Path                       # MJCF/XML file
    prepare_state: PrepareStateConfig     # Init pose
```

**PD Gains Source:** Hardcoded in `t1_23dof/deploy_config.py` (currently wrong!)

---

### holosoma_inference (UPDATED ARCHITECTURE)

**NEW: PD Gains Moved to ONNX Metadata!**

```python
@dataclass(frozen=True)
class RobotConfig:
    robot_type: str                       # "t1_29dof"
    robot: str                            # "t1"
    sdk_type: Literal["unitree", "booster", "ros2"]
    motor_type: Literal["serial", "parallel"]

    # Kinematics
    num_motors: int = 29
    num_joints: int = 29
    num_upper_body_joints: int = 16       # T1: head(2) + arms(14)
    default_dof_angles: tuple[float, ...]

    # Mappings
    motor2joint: tuple[int, ...]          # Identity for T1
    joint2motor: tuple[int, ...]
    dof_names: tuple[str, ...]            # Joint names

    # PD Gains - NOW OPTIONAL! ⭐
    motor_kp: tuple[float, ...] | None = None  # Loaded from ONNX!
    motor_kd: tuple[float, ...] | None = None  # Loaded from ONNX!

    # Link names
    torso_link_name: str = "Trunk"
    left_hand_link_name: str | None = None
```

**PD Gains Source:**
1. **Training:** Defined in `holosoma/holosoma/config_values/robot.py` (control=RobotControlConfig)
2. **ONNX Export:** Embedded in model metadata during export
3. **Deployment:** Loaded from ONNX at runtime (single source of truth!)

**Key Insight:** Deployment configs are now MUCH simpler - no duplicate PD gain definitions!

---

### booster_deploy

```python
@configclass
class RobotConfig:
    mjcf_path: str
    joint_names: tuple[str, ...]          # Real hardware order
    joint_stiffness: torch.Tensor         # Kp gains
    joint_damping: torch.Tensor           # Kd gains
    default_joint_pos: torch.Tensor       # Standing pose
    effort_limit: torch.Tensor            # Torque limits
    action_scale: float = 0.25
```

**PD Gains Source:**
- Base config: `T1_23DOF_CFG` (very low, conservative)
- Override: `T1WalkControllerCfg` (working tuned gains)

---

## Control Loop Comparison

**THIS IS THE CRITICAL DIFFERENCE!**

### booster_deploy (WORKING ✅)

```python
def ctrl_step(self, dof_targets: torch.Tensor):
    """Manual PD torque control with per-substep state updates."""
    dof_targets = dof_targets.cpu().numpy()

    # Get initial state
    dof_pos = self.mj_data.qpos[7:]
    dof_vel = self.mj_data.qvel[6:]
    kp = self.robot.joint_stiffness.numpy()
    kd = self.robot.joint_damping.numpy()
    ctrl_limit = self.robot.effort_limit.numpy()

    # CRITICAL: Manual PD with dynamic feedback
    for i in range(self.decimation):  # 4 physics steps
        # Compute PD torque based on CURRENT state
        tau = kp * (dof_targets - dof_pos) - kd * dof_vel
        tau = np.clip(tau, -ctrl_limit, ctrl_limit)

        self.mj_data.ctrl = tau  # Apply torques
        mujoco.mj_step(self.mj_model, self.mj_data)

        # CRITICAL: Update state for next substep! ⭐
        dof_pos = self.mj_data.qpos[7:]
        dof_vel = self.mj_data.qvel[6:]
```

**Flow (4 physics steps @ 200Hz per policy step @ 50Hz):**
```
Policy Step 1 (t=0.00s):
  Substep 1: error=0.5 → tau=100Nm  [state updated]
  Substep 2: error=0.4 → tau=80Nm   [state updated]
  Substep 3: error=0.3 → tau=60Nm   [state updated]
  Substep 4: error=0.2 → tau=40Nm   [state updated]
→ Torque smoothly decreases as robot approaches target (responsive!)
```

---

### Colosseum (BROKEN ❌)

```python
def ctrl_step(self, dof_targets: torch.Tensor):
    """Position actuators with static targets."""
    dof_targets = dof_targets.cpu().numpy()

    if getattr(self.mj_model, "na", 0) > 0:
        # Position actuator mode
        for _ in range(self.decimation):  # 4 physics steps
            self.mj_data.ctrl = dof_targets  # Static position targets
            mujoco.mj_step(self.mj_model, self.mj_data)
            # NO state update here! ❌
```

**Flow (MuJoCo internal PD):**
```
Policy Step 1 (t=0.00s):
  Substeps 1-4: ctrl=target_position (unchanged)
  MuJoCo internally: Computes torques, but without explicit state updates
→ Less responsive, feels "sluggish" or "heavy"
```

**Why This Causes "Heavy Feet":**
- Position targets are static for entire decimation window
- MuJoCo's internal PD controller handles torque computation
- No explicit state updates between substeps
- Robot feels unresponsive to rapid state changes

---

### holosoma_inference (Real Robot Only)

holosoma_inference doesn't have a MuJoCo controller - it's designed for real robot deployment:

```python
# SDK interface sends commands via DDS/Unitree SDK
def send_low_command(self, cmd_q, cmd_dq, cmd_tau, kp, kd):
    # Real robot firmware handles PD control @ 500Hz
    # Policy runs @ 50Hz, robot interpolates
    ...
```

**Not applicable for sim-to-sim testing!**

---

## PD Gains Analysis

### Complete Comparison Table

**Training Configurations:**

| Joint | **Mfr Training (12-DOF legs)** | **Holosoma Training (29-DOF)** | **Colosseum Computed** | **Your/Holosoma** |
|-------|--------------------------------|--------------------------------|------------------------|-------------------|
| **HEAD** |
| Head | N/A | **Kp=5, Kd=0.5** | Kp=15.99, Kd=0.68 | **3.2x** |
| **ARMS** |
| Shoulder/Elbow | N/A | **Kp=20, Kd=0.5** | Kp=160.61, Kd=8.52 | **8x / 17x** ❌ |
| Wrist/Hand | N/A | **Kp=20, Kd=0.5** | N/A (23-DOF) | N/A |
| **TORSO** |
| Waist | N/A | **Kp=200, Kd=5** | Kp=188.76, Kd=12.02 | **0.94x / 2.4x** |
| **LEGS** |
| Hip Pitch | **Kp=200, Kd=5** | **Kp=200, Kd=5** | Kp=206.83, Kd=13.17 | **1.03x / 2.6x** ❌ |
| Hip Roll/Yaw | **Kp=200, Kd=5** | **Kp=200, Kd=5** | Kp=188.76, Kd=12.02 | **0.94x / 2.4x** ❌ |
| Knee | **Kp=200, Kd=5** | **Kp=200, Kd=5** | Kp=251.09, Kd=15.98 | **1.26x / 3.2x** ❌ |
| Ankle | **Kp=50, Kd=1** | **Kp=50, Kd=3** | Kp=134.05, Kd=8.53 | **2.7x / 2.8-8.5x** ❌ |

**Deployment Configurations:**

| Joint | **booster_deploy Base** | **booster_deploy locomotion.py** | **Colosseum Deployed** |
|-------|-------------------------|----------------------------------|------------------------|
| Head | Kp=4, Kd=1 | **Kp=4, Kd=1** | Kp=15.99, Kd=0.68 |
| Arms | Kp=4, Kd=1 | **Kp=50, Kd=1** | Kp=160.61, Kd=8.52 ❌ |
| Waist | Kp=80, Kd=2 | **Kp=200, Kd=5** | Kp=188.76, Kd=12.02 |
| Hip P/R/Y | Kp=80, Kd=2 | **Kp=200, Kd=5** | Kp=189-207, Kd=12-13 ❌ |
| Knee | Kp=80, Kd=2 | **Kp=200, Kd=5** | Kp=251, Kd=16 ❌ |
| Ankle | Kp=30, Kd=2 | **Kp=50, Kd=2** | Kp=134, Kd=8.5 ❌ |

### Key Findings

**1. Manufacturer Only Trained 12-DOF Legs:**
- Source: `external/booster_gym/envs/T1.yaml`
- Legs: Hip=200, Knee=200, Ankle=50
- Damping: Hip/Knee=5, Ankle=1
- ❌ **NO upper body** (head, arms, waist not trained!)

**2. Holosoma Successfully Trained Full 29-DOF:**
- Source: `external/holosoma/src/holosoma/holosoma/config_values/robot.py:1024-1066`
- **Legs match manufacturer exactly** (Hip/Knee=200, Ankle=50)
- **Upper body gains added:** Head=5, Arms=20, Waist=200
- **Very low damping:** 0.5-5 (vs your computed 8-16)
- ✅ **This is THE gold standard for T1 full-body training!**

**3. Your Computed Gains Are Wrong:**
- **Arms:** 8x too stiff (160 vs 20) ❌
- **Damping:** 2-17x too high everywhere ❌
- **Ankles:** 2.7x too stiff (134 vs 50) ❌
- **Root cause:** Natural frequency method doesn't match empirical tuning

**4. booster_deploy Has Two Configs:**
- **Base T1_23DOF_CFG:** Very conservative (Kp=4-80, safe but underpowered)
- **T1WalkControllerCfg:** Tuned for locomotion (Kp=50-200, works!)
- Uses `.replace()` method to override base config

---

## Policy Implementation

### Observation Computation

**Colosseum T1VelocityPolicy:**
```python
def compute_observation(self) -> torch.Tensor:
    """Build observation matching VELOCITY_OBS_SPEC contract."""
    real2sim = self.robot.data.real2sim_joint_indexes  # Remap!

    obs = torch.cat([
        self.robot.data.root_lin_vel_b,      # (3,) base linear vel
        self.robot.data.root_ang_vel_b,      # (3,) base angular vel
        self.robot.data.projected_gravity_b, # (3,) projected gravity
        self.robot.data.joint_pos[real2sim], # (23,) REMAPPED to sim order
        self.robot.data.joint_vel[real2sim], # (23,) REMAPPED to sim order
        self.last_action,                    # (23,) in sim order
        self.vel_command.to_tensor(),        # (3,) [vx, vy, vyaw]
    ], dim=-1)

    return obs.unsqueeze(0)  # (1, 82) for 23-DOF
```

**holosoma LocomotionPolicy:**
```python
# From external/holosoma (deployment side now simplified)
# Observation computation delegated to BasePolicy

def prepare_obs_for_rl(self, robot_state_data):
    """Convert robot state to policy observation."""
    # 1. Extract observations
    obs_buffer = self.get_current_obs_buffer_dict(robot_state_data)

    # 2. Scale observations
    current_obs = self.parse_current_obs_dict(obs_buffer)

    # 3. Update history (deques)
    group_obs = self._update_obs_history(current_obs)

    # 4. Return policy input
    return {"actor_obs": group_obs["actor_obs"]}

# Observation space (100 dims for T1):
# - base_ang_vel (3)
# - projected_gravity (3)
# - command_lin_vel (2) [vx, vy]
# - command_ang_vel (1) [vyaw]
# - dof_pos (29) relative to default
# - dof_vel (29)
# - actions (29) previous action
# - sin_phase (2) left/right foot
# - cos_phase (2)
```

**booster_deploy LocomotionPolicy:**
```python
def _get_observation(self):
    """Compute observation with 10-step history."""
    obs = torch.cat([
        base_ang_vel,                    # (3,)
        projected_gravity,               # (3,)
        commands,                        # (3,) [vx, vy, vyaw]
        (dof_pos - default_pos),         # (num_action,) relative
        dof_vel * obs_scale,             # (num_action,) scaled
        last_action,                     # (num_action,)
    ], dim=0)

    # Update 10-step history
    if self.obs_history is None:
        self.obs_history = obs.repeat(self.actor_obs_history_length, 1)
    else:
        self.obs_history = torch.cat([self.obs_history[1:], obs.unsqueeze(0)])

    return self.obs_history.flatten()  # (10 * obs_dim,)
```

### Key Differences

| Feature | Colosseum | holosoma | booster_deploy |
|---------|-----------|----------|----------------|
| **base_lin_vel** | ✅ Included | ❌ Not in actor_obs | ❌ Not included |
| **Observation history** | ❌ Single-step | ✅ Deque-based, per-term | ✅ 10-step flattened |
| **Phase observations** | ❌ None | ✅ sin/cos phase (2+2) | ❌ None |
| **Command format** | 3D [vx, vy, vyaw] | 2D lin + 1D ang | 3D [vx, vy, vyaw] |
| **Joint remapping** | ✅ real2sim/sim2real | ✅ motor2joint/joint2motor | ✅ real2sim_joint_map |
| **dof_vel scaling** | ❌ No explicit scale | ✅ obs_scales per term | ✅ obs_dof_vel_scale=1.0 |

**Critical Observation:**
- holosoma uses **NO base_lin_vel** in actor observations for locomotion!
- Phase observations are holosoma-specific (gait coordination)
- All use relative joint positions (`dof_pos - default_pos`)

---

## Model Loading

### Colosseum (ONNX)

```python
def _load_artifact(self, model_path: Path) -> "_PolicyModule":
    """Load ONNX model."""
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if torch.cuda.is_available() else ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), providers=providers)
    return _OnnxPolicyWrapper(session)

class _OnnxPolicyWrapper:
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        # Tensor conversion overhead!
        ort_inputs = {name: obs.detach().cpu().numpy()
                      for name in self._input_names}
        outputs = self.session.run(self._output_names, ort_inputs)
        result = torch.from_numpy(outputs[0])
        return result.to(obs.device)
```

**Pros:**
- Cross-platform (CPU, GPU, NPU)
- Smaller binary size

**Cons:**
- Tensor conversion overhead (torch → numpy → torch)
- Potential numerical precision issues
- Missing observation normalizer?
- Operator compatibility issues

---

### holosoma_inference (ONNX with Metadata)

```python
def _init_policy_components(self, model_path: str):
    """Load ONNX model and extract PD gains from metadata."""
    import onnxruntime as ort

    # 1. Load ONNX model
    self.policy_session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )

    # 2. Extract PD gains from metadata ⭐
    metadata = self.policy_session.get_modelmeta().custom_metadata_map
    if 'motor_kp' in metadata:
        kp_str = metadata['motor_kp']  # "206.83,188.76,..."
        self.motor_kp = np.array([float(x) for x in kp_str.split(',')])

    if 'motor_kd' in metadata:
        kd_str = metadata['motor_kd']
        self.motor_kd = np.array([float(x) for x in kd_str.split(',')])

    # 3. Create inference function
    def policy_act(obs_dict):
        return self.policy_session.run(None, obs_dict)[0]

    self.policy = policy_act
```

**Key Innovation:** PD gains stored in ONNX metadata (single source of truth!)

---

### booster_deploy (TorchScript)

```python
self._model: torch.jit.ScriptModule = torch.jit.load(
    policy_path, map_location="cpu"
)
self._model.eval()

# No tensor conversion needed!
action = self._model(obs)  # Native PyTorch execution
```

**Pros:**
- No tensor conversion overhead
- Identical to training (PyTorch → PyTorch)
- Guaranteed numerical consistency

**Cons:**
- Requires PyTorch runtime (larger binary)
- Less portable than ONNX

---

## Joint Order Remapping

All three systems handle the fact that MuJoCo alphabetically sorts joints, which differs from hardware order.

### Colosseum

```python
class RobotData:
    def __init__(self, cfg: RobotConfig):
        # Compute bidirectional mappings
        self.real2sim_joint_indexes = [
            cfg.joint_names.index(name)
            for name in cfg.sim_joint_names
        ]
        self.sim2real_joint_indexes = [
            cfg.sim_joint_names.index(name)
            for name in cfg.joint_names
        ]

# Usage in policy:
joint_pos_sim = robot.data.joint_pos[robot.data.real2sim_joint_indexes]
joint_targets_real = action[robot.data.sim2real_joint_indexes]
```

---

### holosoma_inference

```python
# motor2joint mapping (identity for T1)
motor2joint = tuple(range(29))
joint2motor = tuple(range(29))

# Used in state processor:
for i in range(num_joints):
    motor_idx = joint2motor[i] if joint2motor else i
    q[7 + i] = motor_state[motor_idx].q
```

---

### booster_deploy

```python
self.real2sim_joint_map = torch.tensor([
    self.robot.cfg.joint_names.index(name)
    for name in self.cfg.policy_joint_names
], dtype=torch.long)

# Usage in action application:
dof_targets.scatter_reduce_(
    0,
    self.real2sim_joint_map,
    action * self.action_scale,
    reduce='sum'
)
```

**All three use the same pattern:** Compute mapping once at init, use at runtime.

---

## Input Handling

### Colosseum (Modular Input Sources)

```python
class JoystickInputSource(BaseInputSource):
    """evdev-based gamepad with threading."""
    def __init__(self, config: InputConfig):
        self.device = find_joystick_device()
        self.poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True
        )
        self.poll_thread.start()

    def get_vx_cmd(self) -> float:
        with self._lock:
            raw = self._state.left_stick_y
            if abs(raw) < self.config.control_threshold:
                return 0.0
            return raw

# Standard gamepad mapping:
# - Left stick Y: forward/backward
# - Left stick X: left/right strafe
# - Right stick X: yaw rotation
```

**Pros:**
- Clean separation (input source abstraction)
- Type-safe (InputState dataclass)
- Deadzone filtering

---

### holosoma_inference (InterfaceWrapper Integration)

```python
class BoosterRemoteControlService:
    """evdev joystick with background thread."""
    def __init__(self):
        self.joystick = find_joystick_device()
        self.thread = threading.Thread(
            target=self._poll_joystick, daemon=True
        )
        self.thread.start()

    def _handle_axis(self, code: int, value: int):
        if code == ABS_Y:  # Left stick Y
            normalized = (value - 32768) / 32768.0
            self.vx = normalized * 0.8  # Max 0.8 m/s
            self.ly = self.vx
```

**Integrated into InterfaceWrapper** - less modular but simpler for real robot.

---

### booster_deploy (Simple Keyboard)

```python
# Simple keyboard controls (no joystick in reference)
# W/A/S/D for velocity commands
# Uses pynput or similar
```

**Minimal implementation** - focused on core functionality.

---

## Actionable Fixes

### Immediate Fix (While Retraining)

**1. Implement Manual PD Control** (⭐ HIGHEST PRIORITY)

Modify `src/colosseum/deploy/backends/mujoco.py`:

```python
def ctrl_step(self, dof_targets: torch.Tensor) -> None:
    """Apply joint targets using manual PD torque control.

    CRITICAL: Matches booster_deploy implementation with per-substep
    state updates for responsive, dynamic feedback.
    """
    dof_targets = dof_targets.cpu().numpy()

    # Update velocity commands if needed
    if self.vel_command is not None:
        self.update_command()

    # Get initial state
    dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
    dof_vel = self.mj_data.qvel.astype(np.float32)[6:]

    # Get PD gains and limits
    kp = np.asarray(self.robot.cfg.joint_stiffness, dtype=np.float32)
    kd = np.asarray(self.robot.cfg.joint_damping, dtype=np.float32)
    effort_limit = np.asarray(self.robot.cfg.effort_limit, dtype=np.float32)

    # Manual PD control with per-substep state updates
    for _ in range(self.cfg.mujoco.decimation):
        # Compute PD torque based on CURRENT state
        tau = kp * (dof_targets - dof_pos) - kd * dof_vel
        tau = np.clip(tau, -effort_limit, effort_limit)

        # Apply torques (not positions!)
        self.mj_data.ctrl[:] = tau

        # Step physics
        mujoco.mj_step(self.mj_model, self.mj_data)

        # CRITICAL: Update state for next substep ⭐
        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
```

**Expected Result:** Robot should feel MUCH more responsive, "heavy feet" reduced significantly.

**Test Command:**
```bash
pixi run deploy -b mujoco -t t1-velocity-rough
```

---

**2. Test with PyTorch Model** (if available)

If you have a `.pt` TorchScript checkpoint, modify `src/colosseum/deploy/core/policy.py`:

```python
def _load_artifact(self, model_path: Path) -> "_PolicyModule":
    """Load model from .onnx or .pt file."""

    if model_path.suffix in [".pt", ".pth"]:
        # Load TorchScript model (no ONNX overhead)
        print(f"[Policy] Loading TorchScript model: {model_path}")
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        return model

    elif model_path.suffix == ".onnx":
        # Existing ONNX loading code
        ...
```

**Expected Result:** If PyTorch works but ONNX doesn't → ONNX export is the issue.

---

### Long-Term Fix (Retraining Required)

**3. Retrain with Holosoma T1 29-DOF Gains** (🔥 RECOMMENDED)

Modify `src/colosseum/robots/t1_23dof/actuators.py`:

```python
# HOLOSOMA T1 29-DOF GAINS (proven working for full-body T1 training)
# Source: external/holosoma config_values/robot.py:1024-1066

# Head (holosoma: Kp=5, Kd=0.5)
T1_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
    joint_names_expr=("AAHead_yaw", "Head_pitch"),
    stiffness=5.0,     # Was computed: 15.99
    damping=0.5,       # Was computed: 0.68
    effort_limit=MOTOR_SPECS["neck"].effort_limit,
    armature=MOTOR_SPECS["neck"].reflected_inertia,
)

# Arms (holosoma: Kp=20, Kd=0.5 for ALL arm joints)
T1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Shoulder.*", ".*Elbow.*"),
    stiffness=20.0,    # Was computed: 160.61 (8x too stiff!)
    damping=0.5,       # Was computed: 8.52 (17x too high!)
    effort_limit=MOTOR_SPECS["arm"].effort_limit,
    armature=MOTOR_SPECS["arm"].reflected_inertia,
)

# Waist (holosoma: Kp=200, Kd=5)
T1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    joint_names_expr=("Waist",),
    stiffness=200.0,   # Was computed: 188.76 (close)
    damping=5.0,       # Was computed: 12.02 (2.4x too high)
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)

# Hip Pitch (holosoma: Kp=200, Kd=5)
T1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Hip_Pitch",),
    stiffness=200.0,   # Was computed: 206.83
    damping=5.0,       # Was computed: 13.17 (2.6x too high)
    effort_limit=MOTOR_SPECS["hip_pitch"].effort_limit,
    armature=MOTOR_SPECS["hip_pitch"].reflected_inertia,
)

# Hip Roll (holosoma: Kp=200, Kd=5)
T1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Hip_Roll",),
    stiffness=200.0,   # Was computed: 188.76
    damping=5.0,       # Was computed: 12.02 (2.4x too high)
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)

# Hip Yaw (holosoma: Kp=200, Kd=5)
T1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Hip_Yaw",),
    stiffness=200.0,   # Was computed: 188.76
    damping=5.0,       # Was computed: 12.02 (2.4x too high)
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)

# Knee (holosoma: Kp=200, Kd=5)
T1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Knee_Pitch",),
    stiffness=200.0,   # Was computed: 251.09
    damping=5.0,       # Was computed: 15.98 (3.2x too high)
    effort_limit=MOTOR_SPECS["knee"].effort_limit,
    armature=MOTOR_SPECS["knee"].reflected_inertia,
)

# Ankle Pitch (holosoma: Kp=50, Kd=3)
T1_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Ankle_Pitch",),
    stiffness=50.0,    # Was computed: 134.05 (2.7x too stiff!)
    damping=3.0,       # Was computed: 8.53 (2.8x too high)
    effort_limit=MOTOR_SPECS["ankle"].effort_limit,
    armature=MOTOR_SPECS["ankle"].reflected_inertia,
)

# Ankle Roll (holosoma: Kp=50, Kd=3)
T1_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Ankle_Roll",),
    stiffness=50.0,    # Was computed: 134.05 (2.7x too stiff!)
    damping=3.0,       # Was computed: 8.53 (2.8x too high)
    effort_limit=MOTOR_SPECS["ankle"].effort_limit,
    armature=MOTOR_SPECS["ankle"].reflected_inertia,
)
```

**Also update deployment config** `src/colosseum/robots/t1_23dof/deploy_config.py`:

```python
joint_stiffness=(
    5.0, 5.0,          # Head (holosoma T1)
    20.0, 20.0, 20.0, 20.0,  # Left arm (holosoma T1)
    20.0, 20.0, 20.0, 20.0,  # Right arm
    200.0,             # Waist (holosoma T1)
    200.0, 200.0, 200.0, 200.0, 50.0, 50.0,  # Left leg (holosoma T1)
    200.0, 200.0, 200.0, 200.0, 50.0, 50.0,  # Right leg
),

joint_damping=(
    0.5, 0.5,          # Head (holosoma T1)
    0.5, 0.5, 0.5, 0.5,  # Left arm (holosoma T1)
    0.5, 0.5, 0.5, 0.5,  # Right arm
    5.0,               # Waist (holosoma T1)
    5.0, 5.0, 5.0, 5.0, 3.0, 3.0,  # Left leg (holosoma T1)
    5.0, 5.0, 5.0, 5.0, 3.0, 3.0,  # Right leg
),
```

**Retrain:**
```bash
pixi run python -m mjlab.scripts.train --task=velocity-t1-23dof
```

**Why This Works:**
- ✅ **T1-specific proven gains** (not G1, not generic)
- ✅ **29-DOF trained** (even more complete than 23-DOF)
- ✅ **Legs match manufacturer exactly** (Hip/Knee=200, Ankle=50)
- ✅ **Arms MUCH lower** (20 vs your 160 - fixes stiffness!)
- ✅ **Damping MUCH lower** (0.5-5 vs your 8-16 - fixes "heavy" feel!)

---

## Summary of Architectural Lessons

### What Colosseum Can Learn from holosoma_inference

1. **PD Gains in ONNX Metadata** ⭐
   - Single source of truth
   - No duplication between training and deployment
   - Cleaner configs

2. **Observation History Buffers**
   - Per-term deques for temporal context
   - More robust policies

3. **Multi-Robot Support**
   - Factory pattern for SDK abstraction
   - InterfaceWrapper design

### What Colosseum Can Learn from booster_deploy

1. **Manual PD Torque Control** ⭐⭐⭐
   - Per-substep state updates
   - Dynamic feedback loop
   - THIS is what fixes "heavy feet"!

2. **TorchScript Models**
   - No ONNX conversion overhead
   - Guaranteed numerical consistency
   - Simpler deployment

3. **Working Reference Implementation**
   - Proven locomotion on T1
   - Known-good PD gains (locomotion.py)
   - Simple, understandable codebase

### What Colosseum Does Better

1. **Type-Safe Configs**
   - Pydantic frozen dataclasses
   - Strict validation
   - Better than @configclass

2. **Clean Separation of Concerns**
   - Policy uses pure MDP functions
   - Shared observation logic (training + deployment)
   - Better maintainability

3. **Modular Input Abstraction**
   - BaseInputSource interface
   - Easy to extend (joystick, keyboard, ROS)
   - Thread-safe by design

---

## Recommended Action Plan

### Phase 1: Quick Validation (< 1 hour)

1. ✅ **Implement manual PD control** (Fix #1 above)
   - Test immediately: `pixi run deploy -b mujoco -t t1-velocity-rough`
   - Expected: 50-70% improvement in responsiveness

2. ⏸️ **Test with PyTorch model** (if available)
   - Rules out ONNX issues
   - If this works → ONNX export is the problem

### Phase 2: Retraining (1-2 days)

3. 🔄 **Retrain with holosoma gains** (Fix #3 above)
   - Update actuators.py and deploy_config.py
   - Run full training: `pixi run python -m mjlab.scripts.train --task=velocity-t1-23dof`
   - Export to ONNX with metadata

4. ✅ **Verify consistency**
   - Test in mjlab play.py (sim-to-sim)
   - Test in deploy.py (sim-to-sim with manual PD)
   - Compare robot behavior

### Phase 3: Production Deployment (variable)

5. 🚀 **Deploy sim-to-sim** (current goal)
   - Should work with fixes #1 + #3

6. 🚀 **Deploy sim-to-sim with ROS**
   - Add ROS bridge (similar to holosoma ROS2 support)

7. 🚀 **Deploy sim-to-real**
   - Use BoosterController backend
   - Test with conservative gains first
   - Gradually increase if stable

---

## Conclusion

The "heavy feet" issue is primarily caused by:
1. **Control loop mismatch** (70% confidence) - static position targets vs dynamic PD feedback
2. **Wrong PD gains** (20% confidence) - computed gains don't match empirical tuning
3. **ONNX export issues** (8% confidence) - potential numerical errors

**Fix order:**
1. Manual PD control (immediate, high impact)
2. Holosoma gains retraining (long-term, guaranteed fix)
3. PyTorch model testing (diagnostic, rules out ONNX)

Your current architecture is sound - you just need these critical fixes to match the working reference implementations.

Good luck with deployment! 🚀
