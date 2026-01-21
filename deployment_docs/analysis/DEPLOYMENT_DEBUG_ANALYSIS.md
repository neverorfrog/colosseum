# Deployment Debug Analysis: "Heavy Feet" Issue

## Problem Summary

**Symptom**: Robot stands but barely moves when commanded forward velocity. Feet seem "very heavy".

**Root Causes** (Multiple Issues Identified):

1. **PRIMARY: PD Gains Completely Wrong** (70% confidence) 🔴 **NEW FINDING**
   - **Manufacturer's training gains** (ground truth): Arms=20, Hip/Knee=200, Ankle=50
   - **Your computed gains**: Arms=160.61 (8x too high!), Ankle=134 (2.7x too high)
   - **Your damping**: 2-17x higher than manufacturer's (creates overdamped motion)
   - Policy trained with wrong gains → learned wrong movement patterns

2. **SECONDARY: Control Loop Mismatch** (20% confidence)
   - **Reference (working)**: Manual PD torque control with per-step state updates
   - **Your implementation**: MuJoCo position actuators with static targets
   - Could compound the gains issue

3. **TERTIARY: Potential ONNX Export Issues** (8% confidence)
   - Multiple tensor conversions could introduce numerical errors
   - Less likely given consistent (not erratic) behavior

**CRITICAL**: After finding manufacturer's actual training config, PD gains are now the PRIMARY issue!

---

## Critical Finding: PD Gains Discrepancy

### 0. Manufacturer's Actual Training Gains (12-DOF LEGS ONLY)

**CRITICAL**: Manufacturer trains **12-DOF legs only**, NOT full 23-DOF body!

**Source**: `external/booster_gym/envs/T1.yaml` (manufacturer's IsaacGym training suite)

**Lines 91-93** (12-DOF locomotion training config - LEGS ONLY):
```yaml
control:
  stiffness: {"Hip": 200., "Knee": 200., "Ankle": 50.} # [N*m/rad]
  damping: {"Hip": 5., "Knee": 5., "Ankle": 1.} # [N*m*s/rad]
  action_scale: 1.
  decimation: 10
```

**Lines 208-215** (Domain randomization during training):
```yaml
randomization:
  dof_stiffness:
    range: [0.95, 1.05]  # ±5% randomization
    operation: "scaling"
    distribution: "uniform"
  dof_damping:
    range: [0.95, 1.05]  # ±5% randomization
    operation: "scaling"
    distribution: "uniform"
```

**Key Observations**:
- ⚠️ **LEGS ONLY (12-DOF)**: Hip_Pitch, Hip_Roll, Hip_Yaw, Knee, Ankle_Pitch, Ankle_Roll × 2 legs
- ⚠️ **NO upper body**: No Neck, Arms, or Waist in manufacturer's training
- ✅ **Uniform gains per joint group**: Hip=200, Knee=200, Ankle=50 (simple, not computed from motor specs)
- ✅ **Domain randomization**: Gains varied ±5% during training for robustness
- ✅ **action_scale = 1.0**: Actions directly scale joint position targets
- ✅ **decimation = 10**: 10 physics steps per policy step (dt=0.002s → 0.02s control freq)

**Implication for 23-DOF Training**:
Since you're training **23-DOF full body** (not 12-DOF), you need gains for:
- **Legs (12-DOF)**: Can use manufacturer's training gains
- **Upper body (11-DOF)**: Must use different source (manufacturer never trained these!)

### 1. Your Training Configuration (Computed from motor specs)

**Source**: `src/colosseum/robots/t1_23dof/actuators.py`

Training uses **computed gains** from motor specifications using natural frequency method:

```python
# Computed using: Kp = I × ω_n², Kd = 2 × ζ × I × ω_n
# Where: ω_n = 10-15 Hz, ζ = 2.0 (overdamped)

Neck (15Hz):     Kp = 15.99,  Kd = 0.68
Arms (12Hz):     Kp = 160.61, Kd = 8.52
Waist (10Hz):    Kp = 188.76, Kd = 12.02
Hip Pitch:       Kp = 206.83, Kd = 13.17
Hip Roll:        Kp = 188.76, Kd = 12.02
Hip Yaw:         Kp = 188.76, Kd = 12.02
Knee:            Kp = 251.09, Kd = 15.98
Ankle Pitch:     Kp = 134.05, Kd = 8.53
Ankle Roll:      Kp = 134.05, Kd = 8.53
```

**Your current deployment** (`deploy_config.py`): **MATCHES TRAINING** ✓

### 2. Working Reference Deployment (booster_deploy)

**Source**: `external/booster_deploy/tasks/locomotion/locomotion.py` (T1WalkControllerCfg, lines 215-230)

Reference uses **manually-tuned LOW gains** that override the base config:

```python
# T1WalkControllerCfg (WORKING configuration)
Neck:            Kp = 4.0,    Kd = 1.0
Arms:            Kp = 50.0,   Kd = 1.0
Waist:           Kp = 200.0,  Kd = 5.0
Hip Pitch:       Kp = 200.0,  Kd = 5.0
Hip Roll:        Kp = 200.0,  Kd = 5.0
Hip Yaw:         Kp = 200.0,  Kd = 5.0
Knee:            Kp = 200.0,  Kd = 5.0
Ankle Pitch:     Kp = 50.0,   Kd = 2.0
Ankle Roll:      Kp = 50.0,   Kd = 2.0
```

### 3. Comparison: All Four Configurations

**Joint-by-joint comparison** (Kp values):

| Joint | **Mfr Training (12-DOF)** | **Holosoma T1 (29-DOF)** | **locomotion.py** | **Your Computed** | **Your/Holosoma** |
|-------|---------------------------|--------------------------|-------------------|-------------------|-------------------|
| **HEAD** |
| Head_yaw | N/A | **5** | 4.0 | 15.99 | **3.2x** |
| Head_pitch | N/A | **5** | 4.0 | 15.99 | **3.2x** |
| **ARMS** |
| Shoulder_Pitch | N/A | **20** | 50.0 | 160.61 | **8.0x** ❌ |
| Shoulder_Roll | N/A | **20** | 50.0 | 160.61 | **8.0x** ❌ |
| Elbow_Pitch | N/A | **20** | 50.0 | 160.61 | **8.0x** ❌ |
| Elbow_Yaw | N/A | **20** | 50.0 | 160.61 | **8.0x** ❌ |
| **WRISTS** |
| Wrist_Pitch | N/A | **20** | N/A (23-DOF) | N/A | N/A |
| Wrist_Yaw | N/A | **20** | N/A (23-DOF) | N/A | N/A |
| Hand_Roll | N/A | **20** | N/A (23-DOF) | N/A | N/A |
| **TORSO** |
| Waist | N/A | **200** | 200.0 | 188.76 | 0.94x ✓ |
| **LEGS** |
| Hip_Pitch | **200** | **200** | 200.0 | 206.83 | 1.03x ✓ |
| Hip_Roll | **200** | **200** | 200.0 | 188.76 | 0.94x ✓ |
| Hip_Yaw | **200** | **200** | 200.0 | 188.76 | 0.94x ✓ |
| Knee | **200** | **200** | 200.0 | 251.09 | 1.26x |
| Ankle_Pitch | **50** | **50** | 50.0 | 134.05 | **2.7x** |
| Ankle_Roll | **50** | **50** | 50.0 | 134.05 | **2.7x** |

**Joint-by-joint comparison** (Kd values):

| Joint | **Mfr Training (12-DOF)** | **Holosoma T1 (29-DOF)** | **locomotion.py** | **Your Computed** | **Your/Holosoma** |
|-------|---------------------------|--------------------------|-------------------|-------------------|-------------------|
| **HEAD** |
| Head | N/A | **0.5** | 1.0 | 0.68 | 1.4x |
| **ARMS** |
| Shoulders/Elbows | N/A | **0.5** | 1.0 | 8.52 | **17x** ❌ |
| **WRISTS** |
| Wrists/Hands | N/A | **0.5** | N/A | N/A | N/A |
| **TORSO** |
| Waist | N/A | **5.0** | 5.0 | 12.02 | **2.4x** |
| **LEGS** |
| Hip Pitch/Roll/Yaw | **5.0** | **5.0** | 5.0 | 12.02-13.17 | **2.4-2.6x** ❌ |
| Knee | **5.0** | **5.0** | 5.0 | 15.98 | **3.2x** ❌ |
| Ankle Pitch | **1.0** | **3.0** | 2.0 | 8.53 | **2.8x** ❌ |
| Ankle Roll | **1.0** | **3.0** | 2.0 | 8.53 | **2.8x** ❌ |

**CRITICAL FINDINGS**:

1. **HOLOSOMA SUCCESSFULLY TRAINED T1 29-DOF** ✅ **BEST REFERENCE!**
   - T1-specific full body training (29-DOF including wrists!)
   - Arms: Kp=20, Kd=0.5 (vs your computed 160.61/8.52 - **8x too stiff!**)
   - Legs: Hip/Knee=200, Ankle=50, Kd=5/3 (matches manufacturer's 12-DOF training)
   - Head: Kp=5, Kd=0.5 (very low, compliant)
   - **This is proven to work for full-body T1 training!**

2. **MANUFACTURER ONLY TRAINED LEGS (12-DOF)** ⚠️
   - No neck, arms, or waist in manufacturer's training suite
   - **You're training 23-DOF** → need gains for 11 upper body DOF from elsewhere!
   - But manufacturer's leg gains match holosoma's exactly!

3. **YOUR GAINS ARE COMPLETELY WRONG** ❌
   - **Arms**: 8x too stiff (160.61 vs holosoma's 20)
   - **Damping**: 2-17x too high everywhere (8-16 vs holosoma's 0.5-5)
   - **Ankles**: 2.7x too stiff (134 vs 50)
   - Natural frequency method doesn't work for simulation!

4. **THREE PROVEN REFERENCES CONVERGE**:
   - **Holosoma T1 29-DOF**: Arms=20, Hip/Knee=200, Ankle=50
   - **Manufacturer 12-DOF**: Hip/Knee=200, Ankle=50 (legs only)
   - **locomotion.py deploy**: Similar but neck/arms slightly higher
   - All use LOW damping (0.5-5) vs your computed (8-16)

5. **Natural Frequency Method vs Empirical Tuning**
   - Your approach: Compute from motor specs (theoretically correct)
   - All working configs: Simple empirical values (works in practice)
   - **Gap suggests**: Motor specs are for hardware, not simulation

**Key Observation**:
- ✅ **HOLOSOMA T1 29-DOF IS PERFECT REFERENCE** - T1-specific, full body, proven working!
- ❌ **Your arms 8x too stiff** - this is THE problem
- ❌ **Your damping 2-17x too high** - creates "heavy" motion
- ⚠️ **Manufacturer only trained legs** - but holosoma has full body!

### Why This Causes "Heavy Feet"

**Root Cause**: Training/deployment gain mismatch + wrong gains for 23-DOF

1. **During training**: Your policy trained with computed gains (Kp=160 for arms, Kp=189-251 for legs, Kd=8-16!)
2. **Policy learned to compensate**: Network outputs actions assuming high stiffness + heavy damping
3. **If you deploy with LOWER gains**: Actions are too small → robot barely moves ("heavy feet")
4. **If you deploy with SAME gains**: Should work IF control loop is correct

**But there's a BIGGER problem for 23-DOF training**:
- **Manufacturer only trained 12-DOF legs** (Hip/Knee=200, Ankle=50, Damping=5/5/1)
- **Your 23-DOF training needs upper body gains** (11-DOF) that manufacturer never trained!
- **Your computed gains are wrong**:
  - Arms: 3.2x too stiff vs locomotion.py (160 vs 50)
  - Damping: 2-8x too high everywhere (8-16 vs 1-5)
  - Ankles: 2.7x too stiff (134 vs 50)
- **This means your policy learned wrong movement patterns** for both legs AND upper body

**Three possible scenarios**:
1. **Best case**: Domain randomization made policy robust → works with corrected gains
2. **Likely case**: Damping way too high → robot feels "heavy" and sluggish
3. **Worst case**: Policy fundamentally expects wrong gains → needs retraining (most likely!)

---

## Additional Differences

### Default Pose Mismatch

**Training** (`constants.py` HOME_QPOS):
```python
Left_Elbow_Yaw: -0.4   # Bent inward
Right_Elbow_Yaw: 0.4   # Bent outward
Legs: -0.2, 0.0, 0.0, 0.4, -0.2, 0.0  # Slightly bent
```

**Reference Deployment** (`booster.py` T1_23DOF_CFG):
```python
Left_Elbow_Yaw: 0.0    # Straight
Right_Elbow_Yaw: 0.0   # Straight
Legs: 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Completely straight
```

**Impact**:
- Observations use `joint_pos_rel = joint_pos - default_pos`
- Different default poses → **wrong relative positions** in observations
- **This could cause incorrect policy behavior!**

### MuJoCo Controller Implementation Differences

#### Reference Implementation (`external/booster_deploy/booster_deploy/controllers/mujoco_controller.py`)

**Actuators**:
- Loads XML directly with **existing actuators from MJCF**
- XML contains pre-defined actuators

**Control Application** (lines 204-223):
```python
# Manual PD control with per-step recomputation
for i in range(self.decimation):
    self.mj_data.ctrl = np.clip(
        kp * (dof_targets - dof_pos) - kd * dof_vel,
        -ctrl_limit,
        ctrl_limit,
    )
    mujoco.mj_step(self.mj_model, self.mj_data)
    dof_pos = self.mj_data.qpos[7:]  # Update for next step
    dof_vel = self.mj_data.qvel[6:]
```

**State Update** (lines 138-161):
- Uses `qpos[:3]` and `qpos[3:7]` for base position/quat
- Uses **`qvel[:3]` and `qvel[3:6]`** directly for base velocities (world frame then transformed)

#### Your Implementation (`src/colosseum/deploy/backends/mujoco.py`)

**Actuators**:
- **Programmatically creates actuators** via `_add_position_actuators_from_cfg()` (lines 17-52)
- Clears XML actuators and adds new ones

**Control Application** (lines 126-149):
```python
# Uses MuJoCo position actuators (decimation loop)
if getattr(self.mj_model, "na", 0) > 0:
    for _ in range(self.decimation):
        self.mj_data.ctrl = dof_targets  # Position targets
        mujoco.mj_step(self.mj_model, self.mj_data)
```

**State Update** (lines 90-114):
- Uses `sensor("orientation")` for quaternion
- Uses **`sensor("imu_lin_vel")` and `sensor("imu_ang_vel")`** for velocities (body frame directly)

**Critical Differences Summary**:

| Aspect | Reference (booster_deploy) | Your Implementation | Impact |
|--------|---------------------------|---------------------|--------|
| **Actuator Source** | XML file (pre-defined) | Programmatically created | Different actuator params? |
| **Control Type** | Manual PD torque control | MuJoCo position actuators | Fundamentally different! |
| **State Updates** | **qpos/qvel updated EACH substep** | qpos/qvel NOT updated in loop | **CRITICAL** |
| **Velocity Source** | qvel (world frame) | IMU sensors (body frame) | Could cause mismatch |

### The Critical Control Loop Difference (HIGH PRIORITY!)

This difference could be **as important as the PD gains issue**!

#### Reference Implementation (booster_deploy)

**Lines 204-223**: Manual PD control with **dynamic state feedback**

```python
def ctrl_step(self, dof_targets: torch.Tensor):
    dof_pos = self.mj_data.qpos[7:]  # Get initial state
    dof_vel = self.mj_data.qvel[6:]
    kp = self.robot.joint_stiffness.numpy()
    kd = self.robot.joint_damping.numpy()

    for i in range(self.decimation):  # 4 physics steps
        # Compute PD torque based on CURRENT state
        torque = kp * (dof_targets - dof_pos) - kd * dof_vel
        torque = np.clip(torque, -effort_limit, effort_limit)

        self.mj_data.ctrl = torque  # Apply torques
        mujoco.mj_step(self.mj_model, self.mj_data)

        # CRITICAL: Update state for next substep!
        dof_pos = self.mj_data.qpos[7:]  # Fresh state
        dof_vel = self.mj_data.qvel[6:]  # Fresh velocities
```

**Key behavior**:
- PD torques are **recomputed each physics substep** (4x per policy step)
- Uses **real-time state feedback** from current simulation state
- Torque adjusts dynamically as robot moves toward target
- More responsive and accurate control

#### Your Implementation (colosseum)

**Lines 126-149**: MuJoCo position actuators with **static targets**

```python
def ctrl_step(self, dof_targets: torch.Tensor):
    dof_targets = dof_targets.cpu().numpy()

    if getattr(self.mj_model, "na", 0) > 0:
        # Position actuator mode
        for _ in range(self.decimation):  # 4 physics steps
            self.mj_data.ctrl = dof_targets  # Same target each time!
            mujoco.mj_step(self.mj_model, self.mj_data)
            # NO state update here!
```

**Key behavior**:
- Position targets are **set once** at the start
- MuJoCo's internal PD controller handles torque computation
- **No dynamic adjustment** based on intermediate states
- Less responsive, could explain "sluggish" behavior

### Why This Matters

**Problem**: Static position targets vs dynamic torque control

1. **Reference approach** (dynamic torque):
   ```
   Step 1: target=0.5, pos=0.0, vel=0.0 → torque = Kp*(0.5-0.0) = 100
   Step 2: target=0.5, pos=0.1, vel=0.3 → torque = Kp*(0.5-0.1) - Kd*0.3 = 70
   Step 3: target=0.5, pos=0.3, vel=0.5 → torque = Kp*(0.5-0.3) - Kd*0.5 = 40
   Step 4: target=0.5, pos=0.45, vel=0.4 → torque = Kp*(0.5-0.45) - Kd*0.4 = 10
   ```
   → Torque **decreases** as robot approaches target (smooth, controlled)

2. **Your approach** (static position):
   ```
   Step 1-4: target=0.5 (unchanged)
   MuJoCo internally: Computes PD but might not update between steps correctly
   ```
   → MuJoCo's internal controller handles this, but behavior may differ

**Symptoms matching your issue**:
- Robot struggles to move (torques not adjusting properly)
- "Heavy feet" (actuators fighting against inertia without dynamic feedback)
- Less responsive to commands

### Additional State Update Issues

#### Reference: Fresh State Each Substep

```python
# After each physics step (lines 222-223)
dof_pos = self.mj_data.qpos[7:]  # Get updated positions
dof_vel = self.mj_data.qvel[6:]  # Get updated velocities
```

→ Next PD computation uses **actual current state**

#### Your Implementation: No Intermediate Updates

```python
# Inside decimation loop
for _ in range(self.decimation):
    self.mj_data.ctrl = dof_targets  # Same target
    mujoco.mj_step(self.mj_model, self.mj_data)
    # qpos/qvel not read here!
```

→ State only updated once at the **end** of the decimation loop (in `update_state()`)

**Impact**:
- Your policy makes decisions based on **4 physics steps old** state
- Reference policy gets fresher state information
- Could cause delayed/incorrect responses

---

## Model Loading and ONNX Export Issues

### Critical Difference: ONNX vs PyTorch Model Formats

This is a **potential secondary issue** that could compound the PD gains problem or cause issues independently.

#### Working Setup (play.py + booster_deploy)

**Source**: `external/mjlab/src/mjlab/scripts/play.py` (lines 186-190) and `external/booster_deploy/tasks/locomotion/locomotion.py` (lines 29-30)

```python
# play.py: Uses RSL-RL runner to get inference policy
runner = OnPolicyRunner(env, asdict(agent_cfg), log_dir=str(log_dir), device=device)
runner.load(str(resume_path), map_location=device)
policy = runner.get_inference_policy(device=device)  # Returns PyTorch model

# booster_deploy: Loads TorchScript directly
self._model: torch.jit.ScriptModule = torch.jit.load(
    policy_path, map_location="cpu")
self._model.eval()
```

**Model Format**: PyTorch/TorchScript (`.pt` or `.pth` files)
- Native PyTorch execution
- No format conversion
- Direct tensor operations
- Full PyTorch operator support

#### Your Deployment Setup

**Source**: `src/colosseum/deploy/core/policy.py` (lines 78-93)

```python
def _load_artifact(self, model_path: Path) -> "_PolicyModule":
    """Resolve a model artifact based on file extension."""
    try:
        import onnxruntime as ort
    except ImportError as err:
        raise RuntimeError(
            "onnxruntime is required to load ONNX checkpoints."
        ) from err

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else [
        "CPUExecutionProvider"
    ]
    session = ort.InferenceSession(str(model_path), providers=providers)
    return _OnnxPolicyWrapper(session)
```

**Model Format**: ONNX (`.onnx` files)
- ONNX Runtime execution (different backend)
- Multiple tensor conversions (PyTorch → NumPy → ONNX → NumPy → PyTorch)
- Potential numerical precision differences
- Limited operator support (some PyTorch ops may convert differently)

### ONNX-Specific Issues

#### 1. Numerical Precision and Type Conversions

**Issue**: Multiple data type conversions introduce numerical errors

```python
# From policy.py _OnnxPolicyWrapper.__call__ (lines 110-114)
def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    ort_inputs = {name: obs.detach().cpu().numpy() for name in self._input_names}  # Torch → NumPy
    outputs = self.session.run(self._output_names, ort_inputs)                     # ONNX inference
    result = torch.from_numpy(outputs[0])                                           # NumPy → Torch
    return result.to(obs.device)
```

**Conversion Chain**:
1. `torch.Tensor` (float32) → `numpy.ndarray` (potential float32/float64 mismatch)
2. ONNX Runtime inference (uses its own numerics)
3. `numpy.ndarray` → `torch.Tensor` (back conversion)

**Potential Problems**:
- Float precision differences (especially with operations like `exp`, `tanh`, layer normalization)
- Accumulation of small errors over time
- Different rounding behavior
- Different handling of edge cases (NaN, Inf)

**Impact**:
- Actions may differ slightly from training
- Over multiple timesteps, small errors accumulate
- Robot behavior diverges from expected

#### 2. Observation Normalizer Export Issues

**Source**: `src/colosseum/utils/train/export.py` (lines 27-39, 99)

```python
def _actor_normalizer(policy: Any) -> Any | None:
  if getattr(policy, "actor_obs_normalization", False):
    return getattr(policy, "actor_obs_normalizer", None)
  return None

# In export_policy():
normalizer = hooks.normalizer_fn(policy) if hooks.normalizer_fn else None
hooks.exporter_fn(policy, str(output_dir), normalizer, filename)
```

**Critical Questions**:
1. **Does your training use observation normalization?**
   - Check training config for `actor_obs_normalization = True`
   - If True, observations are normalized using running mean/std

2. **Is the normalizer correctly exported to ONNX?**
   - The normalizer should be part of the ONNX graph
   - If missing, ONNX model receives unnormalized observations
   - PyTorch model (in play.py) receives normalized observations
   - **Result**: Completely different inputs → wrong actions!

3. **Where is normalization applied?**
   - **In the model**: Normalizer is part of the exported graph (correct)
   - **Outside the model**: Normalizer applied separately (could be missing in ONNX export!)

**How to Check**:
```bash
# Check if your policy uses normalization
pixi run python -c "
from mjlab.tasks.registry import load_rl_cfg
cfg = load_rl_cfg('velocity-t1-23dof')
print(f'actor_obs_normalization: {cfg.policy.actor_obs_normalization}')
"
```

#### 3. Operator Compatibility and Different Implementations

ONNX doesn't support all PyTorch operations. Some operations get converted to ONNX equivalents that may behave differently:

**Potentially Problematic Operations**:
- Layer normalization (different epsilon handling)
- Batch normalization (running stats frozen vs learned)
- Custom activation functions
- In-place operations (converted to copies)
- Dynamic shapes (ONNX uses static shapes)

**Example**: If your policy uses `torch.nn.LayerNorm`:
```python
# PyTorch (training)
layer_norm = nn.LayerNorm(normalized_shape, eps=1e-5)

# ONNX export may use different epsilon or numerical implementation
# Result: Slightly different outputs
```

#### 4. ONNX Export Configuration Issues

**Source**: ONNX export happens via `mjlab.tasks.velocity.rl.export_velocity_policy_as_onnx()`

**Potential Issues**:
- **Opset version**: Older ONNX opsets may not support all operations
- **Dynamic axes**: If not specified correctly, ONNX expects fixed batch size
- **Input/output names**: Mismatch between expected and actual names
- **Constant folding**: ONNX may optimize differently than PyTorch

**Check your export**:
```bash
# Inspect ONNX model
pixi run python -c "
import onnx
model = onnx.load('path/to/policy.onnx')
print('Inputs:', [i.name for i in model.graph.input])
print('Outputs:', [o.name for o in model.graph.output])
print('Opset version:', model.opset_import[0].version)
"
```

### Debugging ONNX Issues

#### Step 1: Compare PyTorch vs ONNX Outputs

Create `test_onnx_vs_pytorch.py`:

```python
#!/usr/bin/env python3
"""Compare PyTorch and ONNX model outputs to detect export issues."""

import torch
import onnxruntime as ort
import numpy as np
from pathlib import Path

def compare_models(pt_path: str, onnx_path: str, num_tests: int = 10):
    """Compare PyTorch and ONNX model outputs with random inputs."""

    # Load PyTorch model
    print(f"Loading PyTorch model: {pt_path}")
    pt_model = torch.jit.load(pt_path, map_location="cpu")
    pt_model.eval()

    # Load ONNX model
    print(f"Loading ONNX model: {onnx_path}")
    onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = onnx_session.get_inputs()[0].name

    # Get observation size from ONNX model
    input_shape = onnx_session.get_inputs()[0].shape
    obs_size = input_shape[1] if len(input_shape) > 1 else input_shape[0]
    print(f"Observation size: {obs_size}")

    # Test with multiple random observations
    max_diffs = []
    mean_diffs = []

    for i in range(num_tests):
        # Generate random observation (typical range: -10 to 10)
        test_obs = torch.randn(1, obs_size) * 3.0

        # PyTorch inference
        with torch.no_grad():
            pt_output = pt_model(test_obs).numpy()

        # ONNX inference
        onnx_output = onnx_session.run(None, {input_name: test_obs.numpy()})[0]

        # Compare
        diff = np.abs(pt_output - onnx_output)
        max_diff = diff.max()
        mean_diff = diff.mean()
        max_diffs.append(max_diff)
        mean_diffs.append(mean_diff)

        if i == 0:
            print(f"\nTest {i+1}:")
            print(f"  PyTorch output: {pt_output.flatten()[:5]} ...")
            print(f"  ONNX output:    {onnx_output.flatten()[:5]} ...")
            print(f"  Max diff:  {max_diff:.8f}")
            print(f"  Mean diff: {mean_diff:.8f}")

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Summary over {num_tests} tests:")
    print(f"  Max difference:  {np.max(max_diffs):.8f} (worst case)")
    print(f"  Mean difference: {np.mean(mean_diffs):.8f} (average)")
    print(f"  Std difference:  {np.std(mean_diffs):.8f}")
    print(f"{'='*60}\n")

    # Interpretation
    max_worst = np.max(max_diffs)
    if max_worst < 1e-5:
        print("✓ EXCELLENT: Differences are negligible (< 0.00001)")
        print("  ONNX export is highly accurate")
    elif max_worst < 1e-3:
        print("✓ GOOD: Differences are small (< 0.001)")
        print("  ONNX export is acceptable, unlikely to cause issues")
    elif max_worst < 1e-1:
        print("⚠ WARNING: Differences are noticeable (< 0.1)")
        print("  ONNX export may cause slight behavior differences")
    else:
        print("✗ CRITICAL: Differences are LARGE (>= 0.1)")
        print("  ONNX export is INCORRECT - this WILL cause problems!")
        print("  Policy behavior will be significantly different")

    return max_worst

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python test_onnx_vs_pytorch.py <pytorch_model.pt> <onnx_model.onnx>")
        sys.exit(1)

    pt_path = sys.argv[1]
    onnx_path = sys.argv[2]

    if not Path(pt_path).exists():
        print(f"Error: PyTorch model not found: {pt_path}")
        sys.exit(1)
    if not Path(onnx_path).exists():
        print(f"Error: ONNX model not found: {onnx_path}")
        sys.exit(1)

    compare_models(pt_path, onnx_path, num_tests=100)
```

**Usage**:
```bash
# Find your trained checkpoint
PT_MODEL="logs/rsl_rl/velocity_t1_23dof/model_5000.pt"
ONNX_MODEL="src/colosseum/tasks/velocity/deploy/t1_23dof/models/policy.onnx"

pixi run python test_onnx_vs_pytorch.py $PT_MODEL $ONNX_MODEL
```

#### Step 2: Check Observation Normalization

Create `check_normalization.py`:

```python
#!/usr/bin/env python3
"""Check if training uses observation normalization."""

import torch
from pathlib import Path

def check_normalization(checkpoint_path: str):
    """Inspect checkpoint for observation normalizer."""

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Check for normalizer in checkpoint
    keys = checkpoint.keys()
    print(f"\nCheckpoint keys: {list(keys)}")

    if "actor_obs_normalizer" in checkpoint:
        print("\n✓ Found actor_obs_normalizer in checkpoint")
        normalizer = checkpoint["actor_obs_normalizer"]
        print(f"  Type: {type(normalizer)}")

        # Try to extract statistics
        if hasattr(normalizer, "mean"):
            print(f"  Mean shape: {normalizer.mean.shape}")
            print(f"  Std shape: {normalizer.std.shape}")
            print(f"  Mean (first 5): {normalizer.mean[:5]}")
            print(f"  Std (first 5): {normalizer.std[:5]}")

        print("\n⚠ WARNING: Training uses observation normalization!")
        print("  ONNX export MUST include the normalizer")
        print("  If ONNX model doesn't normalize, it will receive wrong inputs")

    else:
        print("\n✓ No actor_obs_normalizer found")
        print("  Training does NOT use observation normalization")
        print("  ONNX export should work correctly")

    # Check policy architecture
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        print(f"\nModel has {len(model_state)} parameters")
        print("First few parameter names:")
        for i, key in enumerate(list(model_state.keys())[:10]):
            print(f"  {key}: {model_state[key].shape}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python check_normalization.py <checkpoint.pt>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    check_normalization(checkpoint_path)
```

#### Step 3: Test Deployment with PyTorch Model

Modify `src/colosseum/deploy/core/policy.py` to support `.pt` files:

```python
def _load_artifact(self, model_path: Path) -> "_PolicyModule":
    """Load model from .onnx or .pt file."""

    # Check file extension
    if model_path.suffix in [".pt", ".pth"]:
        print(f"[Policy] Loading TorchScript model: {model_path}")
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        return model

    elif model_path.suffix == ".onnx":
        print(f"[Policy] Loading ONNX model: {model_path}")
        try:
            import onnxruntime as ort
        except ImportError as err:
            raise RuntimeError(
                "onnxruntime is required to load ONNX checkpoints. "
                "Install it via `pip install onnxruntime`."
            ) from err

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else [
            "CPUExecutionProvider"
        ]
        session = ort.InferenceSession(str(model_path), providers=providers)
        return _OnnxPolicyWrapper(session)

    else:
        raise ValueError(
            f"Unsupported model format: {model_path.suffix}. "
            f"Supported formats: .pt, .pth (TorchScript), .onnx (ONNX)"
        )
```

Then test with PyTorch model:
```bash
# Use the same checkpoint that works with play.py
pixi run deploy -b mujoco -t t1-velocity-rough --checkpoint-path logs/rsl_rl/velocity_t1_23dof/model_5000.pt
```

#### Step 4: Inspect ONNX Model Details

Create `inspect_onnx.py`:

```python
#!/usr/bin/env python3
"""Inspect ONNX model structure and configuration."""

import onnx
import onnxruntime as ort
from pathlib import Path

def inspect_onnx(model_path: str):
    """Detailed inspection of ONNX model."""

    print(f"Inspecting ONNX model: {model_path}\n")

    # Load with onnx
    model = onnx.load(model_path)

    # Basic info
    print("="*60)
    print("MODEL STRUCTURE")
    print("="*60)
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"IR version: {model.ir_version}")
    print(f"Opset version: {model.opset_import[0].version}")
    print(f"Graph name: {model.graph.name}")

    # Inputs
    print(f"\n{'='*60}")
    print("INPUTS")
    print("="*60)
    for i, input_tensor in enumerate(model.graph.input):
        shape = [dim.dim_value if dim.dim_value > 0 else "dynamic"
                 for dim in input_tensor.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)
        print(f"  {i}: {input_tensor.name}")
        print(f"      Shape: {shape}")
        print(f"      Type:  {dtype}")

    # Outputs
    print(f"\n{'='*60}")
    print("OUTPUTS")
    print("="*60)
    for i, output_tensor in enumerate(model.graph.output):
        shape = [dim.dim_value if dim.dim_value > 0 else "dynamic"
                 for dim in output_tensor.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(output_tensor.type.tensor_type.elem_type)
        print(f"  {i}: {output_tensor.name}")
        print(f"      Shape: {shape}")
        print(f"      Type:  {dtype}")

    # Nodes (operations)
    print(f"\n{'='*60}")
    print("OPERATIONS")
    print("="*60)
    node_types = {}
    for node in model.graph.node:
        node_types[node.op_type] = node_types.get(node.op_type, 0) + 1

    print(f"Total nodes: {len(model.graph.node)}")
    print("Node type counts:")
    for op_type, count in sorted(node_types.items(), key=lambda x: -x[1]):
        print(f"  {op_type:20s}: {count:3d}")

    # Check for normalization layers
    has_layer_norm = "LayerNormalization" in node_types
    has_batch_norm = "BatchNormalization" in node_types

    if has_layer_norm or has_batch_norm:
        print(f"\n⚠ Model contains normalization layers:")
        if has_layer_norm:
            print(f"  - LayerNormalization: {node_types['LayerNormalization']} instances")
        if has_batch_norm:
            print(f"  - BatchNormalization: {node_types['BatchNormalization']} instances")
        print("  Ensure these match PyTorch implementation exactly!")

    # Runtime info
    print(f"\n{'='*60}")
    print("RUNTIME INFO")
    print("="*60)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Used providers: {session.get_providers()}")

    # Metadata
    if model.metadata_props:
        print(f"\n{'='*60}")
        print("METADATA")
        print("="*60)
        for prop in model.metadata_props:
            print(f"  {prop.key}: {prop.value}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inspect_onnx.py <model.onnx>")
        sys.exit(1)

    model_path = sys.argv[1]
    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    inspect_onnx(model_path)
```

### When is ONNX the Problem?

ONNX is likely the issue if:

1. **PyTorch model works but ONNX doesn't**
   - Same config, same gains, only model format differs
   - PyTorch deployment succeeds, ONNX fails

2. **Large numerical differences** (> 0.01) in comparison test
   - Actions differ significantly between PyTorch and ONNX
   - Small errors compound over time

3. **Observation normalization is used** but not exported correctly
   - Training uses `actor_obs_normalization = True`
   - ONNX model doesn't include normalizer

4. **Erratic or unstable behavior** rather than just "slow/heavy"
   - Robot makes jerky movements
   - Actions seem random or oscillatory
   - This suggests wrong action values, not just wrong gains

### When is ONNX NOT the Problem?

ONNX is likely fine if:

1. **Consistent "heavy feet" behavior**
   - Robot consistently struggles to move
   - Behavior is predictable (not erratic)
   - This pattern matches PD gains mismatch more than ONNX issues

2. **Small numerical differences** (< 0.001) in comparison test
   - PyTorch and ONNX produce nearly identical outputs
   - Differences are negligible

3. **No observation normalization** in training
   - `actor_obs_normalization = False`
   - No normalizer in checkpoint

---

## Recommended Fixes

### Option 0A: Retrain with Hybrid Gains (HIGHEST PRIORITY 🔴 **RECOMMENDED**)

**Goal**: Use manufacturer's leg training gains + reference deployment upper body gains

**Rationale for 23-DOF Training**:
- **Legs (12-DOF)**: Use manufacturer's proven training gains from booster_gym
- **Upper body (11-DOF)**: Use reference deployment gains from locomotion.py (manufacturer never trained these!)
- This combines the best of both worlds for 23-DOF full body training

**Modify** `src/colosseum/robots/t1_23dof/actuators.py` with hybrid approach:

```python
# HYBRID APPROACH for 23-DOF:
# - Upper body (11-DOF): Reference deployment gains from locomotion.py
# - Legs (12-DOF): Manufacturer's training gains from booster_gym

# ===== UPPER BODY (11-DOF) - FROM LOCOMOTION.PY =====

# Neck (from locomotion.py - manufacturer never trained this!)
T1_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
    joint_names_expr=("AAHead_yaw", "Head_pitch"),
    stiffness=4.0,     # locomotion.py (was computed 15.99)
    damping=1.0,       # locomotion.py (was computed 0.68)
    effort_limit=MOTOR_SPECS["neck"].effort_limit,
    armature=MOTOR_SPECS["neck"].reflected_inertia,
)

# Arms (from locomotion.py - manufacturer never trained these!)
T1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Shoulder.*", ".*Elbow.*"),
    stiffness=50.0,    # locomotion.py (was computed 160.61!)
    damping=1.0,       # locomotion.py (was computed 8.52!)
    effort_limit=MOTOR_SPECS["arm"].effort_limit,
    armature=MOTOR_SPECS["arm"].reflected_inertia,
)

# Waist (from locomotion.py - manufacturer never trained this!)
T1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    joint_names_expr=("Waist",),
    stiffness=200.0,   # locomotion.py (was computed 188.76) ✓
    damping=5.0,       # locomotion.py (was computed 12.02)
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)

# ===== LEGS (12-DOF) - FROM MANUFACTURER'S TRAINING =====

# Hip Pitch (from booster_gym training: Hip=200, damping=5)
_hip_pitch_stiffness = 200.0  # Manufacturer training (was computed 206.83)
_hip_pitch_damping = 5.0      # Manufacturer training (was computed 13.17)
T1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Hip_Pitch",),
    stiffness=_hip_pitch_stiffness,
    damping=_hip_pitch_damping,
    effort_limit=MOTOR_SPECS["hip_pitch"].effort_limit,
    armature=MOTOR_SPECS["hip_pitch"].reflected_inertia,
)

# Hip Roll (from booster_gym training: Hip=200, damping=5)
_hip_roll_stiffness = 200.0   # Manufacturer training (was computed 188.76)
_hip_roll_damping = 5.0       # Manufacturer training (was computed 12.02)
T1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Hip_Roll",),
    stiffness=_hip_roll_stiffness,
    damping=_hip_roll_damping,
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)

# Hip Yaw (from booster_gym training: Hip=200, damping=5)
_hip_yaw_stiffness = 200.0    # Manufacturer training (was computed 188.76)
_hip_yaw_damping = 5.0        # Manufacturer training (was computed 12.02)
T1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Hip_Yaw",),
    stiffness=_hip_yaw_stiffness,
    damping=_hip_yaw_damping,
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)

# Knee (from booster_gym training: Knee=200, damping=5)
_knee_stiffness = 200.0       # Manufacturer training (was computed 251.09)
_knee_damping = 5.0           # Manufacturer training (was computed 15.98)
T1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Knee_Pitch",),
    stiffness=_knee_stiffness,
    damping=_knee_damping,
    effort_limit=MOTOR_SPECS["knee"].effort_limit,
    armature=MOTOR_SPECS["knee"].reflected_inertia,
)

# Ankle Pitch (from booster_gym training: Ankle=50, damping=1 or 3?)
# NOTE: booster_gym shows damping=1, but locomotion.py uses damping=2
# Using damping=3 as compromise (matches booster_gym deploy config)
_ankle_stiffness = 50.0       # Manufacturer training (was computed 134.05!)
_ankle_damping = 3.0          # Compromise between training(1) and deploy(2)
T1_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Ankle_Pitch",),
    stiffness=_ankle_stiffness,
    damping=_ankle_damping,
    effort_limit=MOTOR_SPECS["ankle"].effort_limit,
    armature=MOTOR_SPECS["ankle"].reflected_inertia,
)

# Ankle Roll (from booster_gym training: Ankle=50, damping=3)
T1_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Ankle_Roll",),
    stiffness=_ankle_stiffness,
    damping=_ankle_damping,
    effort_limit=MOTOR_SPECS["ankle"].effort_limit,
    armature=MOTOR_SPECS["ankle"].reflected_inertia,
)
```

**Also update deployment config** `src/colosseum/robots/t1_23dof/deploy_config.py`:

HYBRID approach (manufacturer legs + locomotion.py upper body):

```python
joint_stiffness=(
    4.0, 4.0,          # Neck (from locomotion.py - not in manufacturer training)
    50.0, 50.0, 50.0, 50.0,  # Left arm (from locomotion.py - not in manufacturer training)
    50.0, 50.0, 50.0, 50.0,  # Right arm
    200.0,             # Waist (from locomotion.py - not in manufacturer training)
    200.0, 200.0, 200.0, 200.0, 50.0, 50.0,  # Left leg (from manufacturer training!)
    200.0, 200.0, 200.0, 200.0, 50.0, 50.0,  # Right leg (from manufacturer training!)
),

joint_damping=(
    1.0, 1.0,          # Neck (from locomotion.py)
    1.0, 1.0, 1.0, 1.0,  # Left arm (from locomotion.py)
    1.0, 1.0, 1.0, 1.0,  # Right arm
    5.0,               # Waist (from locomotion.py)
    5.0, 5.0, 5.0, 5.0, 3.0, 3.0,  # Left leg (from manufacturer training: Hip=5, Ankle=1-3)
    5.0, 5.0, 5.0, 5.0, 3.0, 3.0,  # Right leg
),
```

**Then retrain the policy**:
```bash
pixi run python -m mjlab.scripts.train --task=velocity-t1-23dof
```

**Why this hybrid approach**:
- ✅ **Legs**: Uses manufacturer's proven 12-DOF training gains (Hip/Knee=200, Ankle=50)
- ✅ **Upper body**: Uses locomotion.py deployment gains (manufacturer never trained these!)
- ✅ **Best of both worlds**: Manufacturer's leg expertise + proven deployment upper body
- ✅ **Deployment will match training exactly** (guaranteed consistency)
- ✅ **Empirically tuned** (not theoretically computed)

**Pros**:
- Uses manufacturer's actual leg training gains (proven for 12-DOF)
- Uses proven deployment upper body gains (only available reference)
- Training/deployment consistency guaranteed
- Follows industry best practices (empirical over theoretical)

**Cons**:
- Requires full retraining (time-consuming)
- Mixing two sources (but best available for 23-DOF)
- Ankle damping unclear (manufacturer training=1, deployment=2, compromise=3?)

**Priority**: **RECOMMENDED** - Best approach for 23-DOF training

---

### Option 0B: Retrain with All Locomotion.py Gains (SIMPLER ALTERNATIVE)

**Goal**: Use ALL gains from locomotion.py for consistency (simpler than hybrid)

**Rationale**:
- **Simpler**: Single source (locomotion.py) for all 23-DOF gains
- **Proven**: This is a working 23-DOF deployment configuration
- **Consistent**: No mixing of sources
- **Trade-off**: Not using manufacturer's training gains for legs (but close enough)

**Code**: Use the same code as Option 0A, but replace leg gains with locomotion.py values:

```python
# All gains from locomotion.py (T1WalkControllerCfg)
# Legs: Use locomotion.py instead of manufacturer training

# Ankle Pitch/Roll (from locomotion.py instead of manufacturer)
_ankle_stiffness = 50.0       # locomotion.py (manufacturer=50 too, same!)
_ankle_damping = 2.0          # locomotion.py (manufacturer=1-3, close)
```

**Deployment config**: Same as Option 0A, but with ankle damping=2.0 instead of 3.0

**Pros**:
- **Simplest approach** - single source for all gains
- Proven working 23-DOF configuration
- Still close to manufacturer's leg gains (only ankle damping differs: 2 vs 1-3)
- No ambiguity about mixing sources

**Cons**:
- Not using manufacturer's exact training gains for legs
- Ankle damping 2.0 vs manufacturer's 1.0 (but deployment uses 2.0, so matches!)

**Priority**: **ALTERNATIVE** - Choose this if you want simplicity over "perfect" leg matching

---

### **Which Option 0 to Choose?**

| Criteria | Option 0A (Hybrid) | Option 0B (All locomotion.py) |
|----------|-------------------|-------------------------------|
| **Leg gains match manufacturer training** | ✅ Exact match | ⚠️ Close (ankle damping 2 vs 1) |
| **Upper body gains** | ✅ locomotion.py | ✅ locomotion.py (same) |
| **Simplicity** | ⚠️ Mixing sources | ✅ Single source |
| **Training/deployment consistency** | ✅ Yes | ✅ Yes |
| **Proven working** | ✅ Both parts proven | ✅ Whole config proven |

**Recommendation**:
- **Option 0A (Hybrid)** if you want to be closest to manufacturer's leg training
- **Option 0B (All locomotion.py)** if you want simplicity (and differences are minimal anyway!)

**Both are good choices** - pick based on your preference. Option 0B is simpler and probably good enough.

---

### Option 0C: Retrain with Holosoma T1 29-DOF Gains (🔥 **BEST - HIGHLY RECOMMENDED** 🔥)

**Goal**: Use holosoma's proven T1 29-DOF full-body training gains

**Rationale**:
- ✅ **T1-specific** (not G1, not generic)
- ✅ **Full 29-DOF body trained** (includes wrists - more than your 23-DOF!)
- ✅ **Proven working** (holosoma successfully trained T1 with these exact gains)
- ✅ **Legs match manufacturer** (Hip/Knee=200, Ankle=50)
- ✅ **Much simpler than computed gains** (Arms=20 vs your 160.61!)
- ✅ **Very low damping** (0.5-5 vs your 8-16)

**Source**: `external/holosoma/src/holosoma/holosoma/config_values/robot.py` lines 1026-1061

**Modify** `src/colosseum/robots/t1_23dof/actuators.py`:

```python
# HOLOSOMA T1 29-DOF GAINS (proven working for full-body T1 training)
# Source: external/holosoma config_values/robot.py t1_29dof_waist_wrist

# Head (holosoma: Kp=5, Kd=0.5)
T1_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
    joint_names_expr=("AAHead_yaw", "Head_pitch"),
    stiffness=5.0,     # Holosoma T1 (was computed 15.99)
    damping=0.5,       # Holosoma T1 (was computed 0.68)
    effort_limit=MOTOR_SPECS["neck"].effort_limit,
    armature=MOTOR_SPECS["neck"].reflected_inertia,
)

# Arms (holosoma: Kp=20, Kd=0.5 for ALL arm joints)
T1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Shoulder.*", ".*Elbow.*"),
    stiffness=20.0,    # Holosoma T1 (was computed 160.61!)
    damping=0.5,       # Holosoma T1 (was computed 8.52!)
    effort_limit=MOTOR_SPECS["arm"].effort_limit,
    armature=MOTOR_SPECS["arm"].reflected_inertia,
)

# Waist (holosoma: Kp=200, Kd=5)
T1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    joint_names_expr=("Waist",),
    stiffness=200.0,   # Holosoma T1 (was computed 188.76) ✓
    damping=5.0,       # Holosoma T1 (was computed 12.02)
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)

# Hip Pitch (holosoma: Kp=200, Kd=5)
_hip_pitch_stiffness = 200.0  # Holosoma T1 (was computed 206.83)
_hip_pitch_damping = 5.0      # Holosoma T1 (was computed 13.17)
T1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Hip_Pitch",),
    stiffness=_hip_pitch_stiffness,
    damping=_hip_pitch_damping,
    effort_limit=MOTOR_SPECS["hip_pitch"].effort_limit,
    armature=MOTOR_SPECS["hip_pitch"].reflected_inertia,
)

# Hip Roll (holosoma: Kp=200, Kd=5)
_hip_roll_stiffness = 200.0   # Holosoma T1 (was computed 188.76)
_hip_roll_damping = 5.0       # Holosoma T1 (was computed 12.02)
T1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Hip_Roll",),
    stiffness=_hip_roll_stiffness,
    damping=_hip_roll_damping,
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)

# Hip Yaw (holosoma: Kp=200, Kd=5)
_hip_yaw_stiffness = 200.0    # Holosoma T1 (was computed 188.76)
_hip_yaw_damping = 5.0        # Holosoma T1 (was computed 12.02)
T1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Hip_Yaw",),
    stiffness=_hip_yaw_stiffness,
    damping=_hip_yaw_damping,
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)

# Knee (holosoma: Kp=200, Kd=5)
_knee_stiffness = 200.0       # Holosoma T1 (was computed 251.09)
_knee_damping = 5.0           # Holosoma T1 (was computed 15.98)
T1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Knee_Pitch",),
    stiffness=_knee_stiffness,
    damping=_knee_damping,
    effort_limit=MOTOR_SPECS["knee"].effort_limit,
    armature=MOTOR_SPECS["knee"].reflected_inertia,
)

# Ankle Pitch (holosoma: Kp=50, Kd=3)
_ankle_stiffness = 50.0       # Holosoma T1 (was computed 134.05!)
_ankle_damping = 3.0          # Holosoma T1 (was computed 8.53)
T1_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Ankle_Pitch",),
    stiffness=_ankle_stiffness,
    damping=_ankle_damping,
    effort_limit=MOTOR_SPECS["ankle"].effort_limit,
    armature=MOTOR_SPECS["ankle"].reflected_inertia,
)

# Ankle Roll (holosoma: Kp=50, Kd=3)
T1_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Ankle_Roll",),
    stiffness=_ankle_stiffness,
    damping=_ankle_damping,
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

**Then retrain the policy**:
```bash
pixi run python -m mjlab.scripts.train --task=velocity-t1-23dof
```

**Why this is the BEST approach**:
- ✅ **T1-specific proven gains** (not G1, not generic deployment)
- ✅ **29-DOF trained** (even more complete than your 23-DOF)
- ✅ **Legs match manufacturer exactly** (Hip/Knee=200, Ankle=50)
- ✅ **Arms are MUCH lower** (20 vs your 160 - THE critical fix!)
- ✅ **Damping is MUCH lower** (0.5-5 vs your 8-16 - fixes "heavy feet"!)
- ✅ **Single authoritative source** (holosoma's working T1 config)
- ✅ **action_scale=0.25** (matches holosoma's locomotion setting)

**Pros**:
- **THE BEST REFERENCE** for T1 full-body training
- Proven working configuration (holosoma trained successfully)
- T1-specific (not adapted from G1 or generic)
- Matches manufacturer's leg gains exactly
- Dramatically reduces your computed gains (especially arms: 20 vs 160!)
- Very low damping (fixes "heavy" feeling)

**Cons**:
- Requires full retraining (time-consuming)
- Arms might be MORE compliant than locomotion.py (20 vs 50)
  - But holosoma proved this works for full-body training!

**Priority**: 🔥 **HIGHLY RECOMMENDED** - This is the gold standard for T1 23-DOF training

---

### **Updated: Which Option 0 to Choose?**

| Criteria | Option 0A (Hybrid) | Option 0B (All locomotion.py) | **Option 0C (Holosoma T1)** 🔥 |
|----------|-------------------|-------------------------------|--------------------------------|
| **T1-specific** | ⚠️ Partial (legs only) | ❌ Generic deployment | ✅ **YES - T1 29-DOF!** |
| **Leg gains match manufacturer** | ✅ Exact match | ⚠️ Close | ✅ **Exact match** |
| **Upper body reference** | ⚠️ locomotion.py | ⚠️ locomotion.py | ✅ **T1-specific!** |
| **Proven full-body training** | ❌ No (hybrid) | ⚠️ Deployment only | ✅ **YES - 29-DOF trained!** |
| **Arms** | locomotion (50) | locomotion (50) | **holosoma (20)** - more compliant |
| **Damping** | Mixed sources | locomotion (1-5) | **holosoma (0.5-5)** - very low |
| **Simplicity** | ⚠️ Mixing sources | ✅ Single source | ✅ **Single source** |

**NEW Recommendation**:
- 🔥 **Option 0C (Holosoma T1)** - **BEST CHOICE!** T1-specific, proven 29-DOF training, matches manufacturer legs
- **Option 0B (All locomotion.py)** - Good alternative if you want slightly stiffer arms (50 vs 20)
- **Option 0A (Hybrid)** - Only if you don't trust holosoma's approach

**Why Option 0C is best**:
1. **Only T1-specific full-body training reference** available
2. **Proven to work** (holosoma successfully trained 29-DOF T1)
3. **Manufacturer leg gains** + **holosoma upper body gains** (perfect combination!)
4. **Dramatically fixes your issues**: Arms 20 vs your 160 (8x!), Damping 0.5-5 vs your 8-16 (2-17x!)

---

### Option A: Fix Control Loop (Try this first while retraining!)

**Goal**: Match the reference implementation's manual PD control with per-step state updates

Modify `src/colosseum/deploy/backends/mujoco.py` to use manual torque control:

```python
def ctrl_step(self, dof_targets: torch.Tensor):
    """Apply control using manual PD torque computation (matches booster_deploy)."""
    dof_targets = dof_targets.cpu().numpy()

    if self.vel_command is not None:
        self.update_command()

    # Get initial state
    dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
    dof_vel = self.mj_data.qvel.astype(np.float32)[6:]

    # Get PD gains
    kp = np.asarray(self.robot.cfg.joint_stiffness, dtype=np.float32)
    kd = np.asarray(self.robot.cfg.joint_damping, dtype=np.float32)
    effort = np.asarray(self.robot.cfg.effort_limit, dtype=np.float32)

    # Manual PD control with per-step state updates
    for _ in range(self.decimation):
        # Compute PD torque based on CURRENT state
        tau = kp * (dof_targets - dof_pos) - kd * dof_vel
        tau = np.clip(tau, -effort, effort)

        # Apply torques (not positions!)
        self.mj_data.ctrl[:] = tau

        # Step physics
        mujoco.mj_step(self.mj_model, self.mj_data)

        # CRITICAL: Update state for next substep
        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
```

**Why this matters**:
- **MuJoCo position actuators** apply PD internally but may not update correctly between substeps
- **Manual torque control** gives you explicit control over PD computation timing
- **Per-step state updates** provide real-time feedback for responsive control
- This **exactly matches** the working reference implementation

**When to use this**:
- If fixing PD gains alone doesn't work
- If you want to match the reference implementation exactly
- As a diagnostic to see if control method is the issue

**Pros**:
- Matches proven working implementation exactly
- Gives explicit control over actuator behavior
- More responsive and accurate

**Cons**:
- Requires code modification (not just config)
- More complex than using built-in position actuators
- Need to remove programmatic actuator creation (or use fallback path)

**Note**: You may need to modify actuator setup to ensure no XML actuators exist, or use the fallback path in your current code (lines 137-149).

### Option B: Quick Fix - Match Reference Gains

Modify `src/colosseum/robots/t1_23dof/deploy_config.py` to match the working reference:

```python
joint_stiffness=(
    4.0, 4.0,          # Neck (was 15.99)
    50.0, 50.0, 50.0, 50.0,  # Left arm (was 160.61)
    50.0, 50.0, 50.0, 50.0,  # Right arm
    200.0,             # Waist (was 188.76)
    200.0, 200.0, 200.0, 200.0, 50.0, 50.0,  # Left leg
    200.0, 200.0, 200.0, 200.0, 50.0, 50.0,  # Right leg
),

joint_damping=(
    1.0, 1.0,          # Neck (was 0.68)
    1.0, 1.0, 1.0, 1.0,  # Left arm (was 8.52)
    1.0, 1.0, 1.0, 1.0,  # Right arm
    5.0,               # Waist (was 12.02)
    5.0, 5.0, 5.0, 5.0, 2.0, 2.0,  # Left leg
    5.0, 5.0, 5.0, 5.0, 2.0, 2.0,  # Right leg
),

default_joint_pos=(
    0.0, 0.0,          # Neck
    0.0, -1.4, 0.0, 0.0,  # Left arm (straighten elbow: was -0.4)
    0.0, 1.4, 0.0, 0.0,   # Right arm (straighten elbow: was 0.4)
    0.0,               # Waist
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Left leg (straighten: was -0.2, 0.4, -0.2)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Right leg
),
```

**Pros**:
- Should work immediately with existing policy
- Matches proven working configuration
- No retraining needed

**Cons**:
- Policy was trained with different gains (but might still work due to domain randomization)

**Priority**: Try this SECOND (after Option A control loop fix)

### Option C: Retrain with Reference Gains (Long-term solution)

Modify `src/colosseum/robots/t1_23dof/actuators.py` to match reference gains:

```python
# Replace computed gains with reference tuned values
T1_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
    joint_names_expr=("AAHead_yaw", "Head_pitch"),
    stiffness=4.0,   # Instead of computed 15.99
    damping=1.0,     # Instead of computed 0.68
    effort_limit=MOTOR_SPECS["neck"].effort_limit,
    armature=MOTOR_SPECS["neck"].reflected_inertia,
)

T1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Shoulder.*", ".*Elbow.*"),
    stiffness=50.0,  # Instead of computed 160.61
    damping=1.0,     # Instead of computed 8.52
    effort_limit=MOTOR_SPECS["arm"].effort_limit,
    armature=MOTOR_SPECS["arm"].reflected_inertia,
)

# ... (similar for other joints)
```

Then retrain the policy.

**Pros**:
- Training and deployment perfectly aligned
- Guaranteed consistency
- Follows best practices

**Cons**:
- Requires full retraining (time-consuming)

### Option D: Test with PyTorch Model (Quick ONNX Debug)

**Goal**: Rule out ONNX export issues

1. **Modify policy loader** to support `.pt` files (see debugging section above)
2. **Test deployment** with the same PyTorch checkpoint used in play.py
3. **Compare results**:
   - If `.pt` works but `.onnx` doesn't → ONNX export is the problem
   - If both fail → PD gains or other config issue

**Pros**:
- Quick test (< 5 minutes to modify and test)
- Definitively identifies if ONNX is the issue
- Uses proven working checkpoint

**Cons**:
- Requires code modification
- Doesn't fix the underlying issue, just identifies it

### Option E: Comprehensive Approach (Recommended)

**Priority order for maximum efficiency**:

1. **Fix control loop FIRST** (Option A) - **CRITICAL**
   - This could be THE root cause
   - Reference uses manual PD with per-step updates
   - Your implementation uses static position targets
   - Takes 10-15 minutes to modify
   - Test immediately

2. **If still not working, fix PD gains** (Option B)
   - Training gains may be too high
   - Match reference manual tuned gains
   - Takes 5 minutes to modify config
   - Test immediately

3. **If still not working, test ONNX** (Option D)
   - Takes 10 minutes to add PyTorch support
   - Tests with same checkpoint as play.py
   - Identifies if ONNX is compounding the problem

4. **If ONNX is the issue, debug export**:
   - Run comparison scripts (Step 1-4 from ONNX debugging)
   - Check observation normalization
   - Fix export configuration

5. **Long-term: Retrain properly** (Option C)
   - Use reference gains for training
   - Use manual torque control in training too
   - Ensure training/deployment consistency
   - Verify ONNX export is correct

---

## Next Steps (Recommended Workflow)

### Phase 1: Quick Fixes (< 45 minutes)

1. ✓ **Identified issues**:
   - **Control loop mismatch** (NEW - could be primary!)
   - PD gains mismatch (likely compound issue)
   - ONNX export (potential secondary)

2. **Apply control loop fix FIRST** (Option A) - **HIGHEST PRIORITY**
   - Modify `src/colosseum/deploy/backends/mujoco.py`
   - Implement manual PD torque control with per-step state updates
   - Match reference booster_deploy implementation exactly
   - Test: `pixi run deploy -b mujoco -t t1-velocity-rough`

3. **If robot still doesn't move properly, apply PD gains fix** (Option B)
   - Modify `src/colosseum/robots/t1_23dof/deploy_config.py`
   - Match reference booster_deploy gains
   - Test: `pixi run deploy -b mujoco -t t1-velocity-rough`

4. **If still broken, test ONNX** (Option D):
   - Add PyTorch model support to policy loader
   - Test with `.pt` checkpoint from play.py
   - This isolates whether ONNX is the issue

### Phase 2: ONNX Debugging (if needed, 1-2 hours)

5. **Run ONNX comparison tests**:
   ```bash
   # Create test scripts from examples above
   python test_onnx_vs_pytorch.py model.pt policy.onnx
   python check_normalization.py model.pt
   python inspect_onnx.py policy.onnx
   ```

6. **Analyze results**:
   - Large differences (> 0.01)? → ONNX export is broken
   - Missing normalizer? → Re-export with correct settings
   - Unsupported ops? → Check ONNX opset version

7. **Fix ONNX export**:
   - Update export configuration in mjlab
   - Re-export policy with correct settings
   - Verify with comparison tests

### Phase 3: Long-term Solutions (hours to days)

8. **Retrain with correct configuration** (Option C)
   - Modify `actuators.py` with tuned gains
   - Run full training
   - Export to ONNX with verification

9. **Verify consistency**:
   - Test in mjlab play.py
   - Test in deploy.py (both .pt and .onnx)
   - Compare robot behavior
   - Document final configuration

### Phase 4: Documentation

10. **Update deployment docs**:
   - Document correct PD gains
   - Document ONNX export procedure
   - Add troubleshooting guide
   - Share lessons learned

---

## Diagnostic Decision Tree

```
START: Robot has "heavy feet", barely moves
│
├─ Test 1: Fix control loop (Option A) - HIGHEST PRIORITY
│  │
│  ├─ SUCCESS: Robot walks properly!
│  │  └─ Root cause: Control loop mismatch (static vs dynamic torques)
│  │     Action: Keep manual PD control implementation
│  │            Retrain with same control method (long-term)
│  │
│  └─ FAIL: Still doesn't work
│     │
│     └─ Test 2: Fix PD gains (Option B)
│        │
│        ├─ SUCCESS: Robot walks properly
│        │  └─ Root cause: PD gains mismatch + control loop issues
│        │     Action: Retrain with reference gains (long-term)
│        │
│        └─ FAIL: Still doesn't work
│           │
│           └─ Test 3: Try PyTorch model (Option D)
│              │
│              ├─ SUCCESS: Robot walks with .pt model
│              │  └─ Root cause: ONNX export issue
│              │     Action: Debug ONNX (Phase 2)
│              │            Fix export configuration
│              │
│              └─ FAIL: Still doesn't work with .pt
│                 └─ Root cause: Observation/other mismatch
│                    Action: Debug observation computation
│                           Check base velocity sources
│                           Compare training vs deployment observations
│                           Verify joint ordering
```

---

## Open Questions

### PD Gains Related

1. **Why does the reference deployment override base gains?**
   - The base `T1_23DOF_CFG` has lower gains (4.0, 80.0)
   - The walking config `T1WalkControllerCfg` overrides some to higher (200.0 for legs)
   - This suggests the gains were manually tuned for walking stability
   - Were they tuned empirically or derived from sim-to-real transfer requirements?

2. **What gains were actually used during training?**
   - Check training logs/config to verify actual PD gains
   - If play.py works, what gains does it use?
   - Does training match the computed gains or the reference gains?

3. **Why are the computed gains so much higher?**
   - Natural frequency method: ω_n = 10-15 Hz, ζ = 2.0
   - This is theoretically correct for motor specs
   - But empirical tuning suggests lower gains work better
   - Is there a domain gap (sim vs real) that requires lower gains?

### ONNX Related

4. **Does training use observation normalization?**
   - Check `actor_obs_normalization` in training config
   - If True, is the normalizer correctly exported to ONNX?
   - Does the ONNX model include the normalizer in its graph?

5. **How accurate is the ONNX export?**
   - What are the numerical differences between PyTorch and ONNX?
   - Are they within acceptable tolerance (< 0.001)?
   - Do small errors accumulate over many timesteps?

6. **What checkpoint does play.py use?**
   - Find the exact `.pt` file used by play.py
   - Is it the same policy you're trying to export to ONNX?
   - Can we test deployment with that same `.pt` file?

### Observation Related

7. **Are there other observation mismatches?**
   - Base velocity: qvel (world frame) vs IMU sensors (body frame)?
   - Projected gravity: computed differently between systems?
   - Joint ordering: verified correct with real2sim/sim2real maps?

8. **What is the observation size?**
   - Training expects: base_lin_vel(3) + base_ang_vel(3) + gravity(3) + joint_pos(23) + joint_vel(23) + last_action(23) + vel_cmd(3) = 81
   - Does deployment match exactly?
   - Are all components in the same order?

---

## Summary

**Most Likely Issues** (in priority order after discovering manufacturer's config):

1. **PD Gains Completely Wrong (70% confidence) 🔴 CRITICAL NEW FINDING!**
   - 🔥 **HOLOSOMA T1 29-DOF** provides the BEST reference (proven T1 full-body training!)
   - **Your training** (23-DOF): Arms=160.61 vs holosoma's 20 (**8x too stiff!**), Damping=8-16 vs holosoma's 0.5-5 (**2-17x too high!**)
   - **Manufacturer trains 12-DOF legs only** (but holosoma has full 29-DOF T1!)
   - **Symptom**: Policy learned with wrong actuator dynamics, especially arms and damping
   - **Why critical**: Arms 8x too stiff + damping 17x too high = "heavy feet" and poor movement
   - **Fix**: Retrain with holosoma T1 29-DOF gains (Option 0C) **← DO THIS!**

2. **Control Loop Mismatch (20% confidence)**
   - **Reference**: Manual PD torque control with per-step state updates
   - **Your implementation**: MuJoCo position actuators with static targets
   - **Symptom**: Could compound the gains issue
   - **Fix**: Try manual control (Option A) while retraining

3. **ONNX Export Issues (8% confidence)**
   - Multiple tensor conversions
   - Potential missing normalizer
   - Symptom: Would cause erratic behavior, not just "heavy feet"
   - Less likely given consistent "sluggish" behavior
   - **Fix**: Test with PyTorch model (Option D) to isolate

4. **Observation Mismatch (2% confidence)**
   - Different velocity sources (qvel vs IMU)
   - Wrong default pose
   - Symptom: Would cause completely wrong actions
   - Least likely given robot does respond to commands (just poorly)

**UPDATED Recommended Action** (after manufacturer config discovery):

**HIGHEST PRIORITY**:
- 🔥 **Option 0C (Holosoma T1 29-DOF)**: Retrain with proven T1 full-body gains **← BEST CHOICE!**
  - **T1-specific 29-DOF** training gains (proven working by holosoma)
  - Arms: Kp=20 (vs your 160!) Legs: Hip/Knee=200, Ankle=50 (matches manufacturer)
  - Damping: 0.5-5 (vs your 8-16 - **THIS fixes "heavy feet"!**)
  - **ROOT CAUSE FIX**: Arms 8x too stiff, damping 2-17x too high
  - **Start retraining NOW with holosoma's gains**

**Good Alternatives**:
- **Option 0B (All locomotion.py)**: Simpler, single source, slightly stiffer arms (50 vs 20)
- **Option 0A (Hybrid)**: Manufacturer legs + locomotion upper body (mixing sources)

**WHILE RETRAINING** (quick tests to verify diagnosis):
1. **Option A**: Try manual torque control (10 min) - may help with current policy
2. **Option B**: Try manufacturer's gains with current policy (5 min) - probably won't work but worth trying
3. **Option D**: Test with PyTorch model (10 min) - rule out ONNX

**LONG-TERM** (after retraining completes):
- Verify new policy works in both training (play.py) and deployment (deploy.py)
- Document final configuration for real robot deployment
- Share lessons learned about empirical vs computed gains
