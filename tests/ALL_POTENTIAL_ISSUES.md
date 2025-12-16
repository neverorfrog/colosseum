# Complete Deployment Issue Checklist

This document catalogs **ALL** potential causes for the T1 deployment instability, including timing, ONNX conversion, observation computation, and configuration mismatches.

---

## Issue Categories

1. **[Control Loop Timing](#1-control-loop-timing)** ⭐ **HIGH CONFIDENCE**
2. **[ONNX Model Conversion](#2-onnx-model-conversion)** ⚠️ **NEW INVESTIGATION**
3. **[Observation Computation](#3-observation-computation)** ⚠️ **NEW INVESTIGATION**
4. **[Action Scaling](#4-action-scaling)**
5. **[Physics Solver Settings](#5-physics-solver-settings)**
6. **[PD Gains](#6-pd-gains)**
7. **[Joint Order Remapping](#7-joint-order-remapping)**
8. **[Actuator Configuration](#8-actuator-configuration)**
9. **[Ground Contact / Init Height](#9-ground-contact--init-height)**
10. **[Decimation / Physics Timestep](#10-decimation--physics-timestep)**

---

## 1. Control Loop Timing

### Description
MujocoController uses fixed 20ms sleep BEFORE state update, causing policy to see stale observations.

### Symptoms
- Robot falls and oscillates
- Works with play.py but fails with MujocoController
- Even dummy policy (returning default_qpos) fails

### Root Cause
```python
# MujocoController.run() line 118
sleep(self.cfg.physics_dt * self.cfg.mujoco.decimation)  # 20ms BEFORE
update_state()  # Reads OLD state
policy_step()   # Inference on 20ms old data
ctrl_step()     # Physics happens HERE
```

**Total latency:** 20ms (sleep) + 0.5ms (update) + 1.5ms (policy) + 0.5ms (ctrl) = **22.5ms**

### Test
```bash
pixi run python tests/test_sleep_timing.py
```

### Expected Outcome
```
COMPARISON SUMMARY
Fixed Sleep    →  20.45ms   0.1234m  FALLING  ← Current problem
Minimal Sleep  →   1.23ms   0.0012m  STABLE   ← Fix works!
Adaptive       →  20.15ms   0.0008m  STABLE   ← Best solution
```

### Fix
Implement adaptive timing (Priority 1). See [mujoco.py:107-124](../src/colosseum/deploy/backends/mujoco.py#L107-L124).

**Confidence:** 85%

---

## 2. ONNX Model Conversion

### Description
Policy exported to ONNX may have numerical differences or incorrect input/output handling compared to PyTorch.

### Potential Issues

#### 2.1 ONNX Export Configuration

**Problem:** ONNX export might use different opset version, dynamic axes, or optimization settings.

**File:** [src/colosseum/deploy/utils/export.py](../src/colosseum/deploy/utils/export.py)

Export uses: `mjlab.tasks.velocity.rl.export_velocity_policy_as_onnx`

**Check:**
```python
# In mjlab/tasks/velocity/rl/exporter.py
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=?,           # What version?
    input_names=?,             # Correct names?
    output_names=?,            # Correct names?
    dynamic_axes=?,            # Any dynamic axes?
    export_params=True,        # Weights exported?
    do_constant_folding=True,  # Optimizations?
)
```

#### 2.2 Observation Normalizer

**Problem:** ONNX export might not include observation normalization that training uses.

**File:** [src/colosseum/deploy/utils/export.py:26-29](../src/colosseum/deploy/utils/export.py#L26-L29)

```python
def _actor_normalizer(policy: Any) -> Any | None:
  if getattr(policy, "actor_obs_normalization", False):
    return getattr(policy, "actor_obs_normalizer", None)  # ← Included in ONNX?
  return None
```

**Check:** Does the exported ONNX model include the normalizer, or is it applied separately?

#### 2.3 Numerical Precision

**Problem:** ONNX Runtime might use different floating-point precision (FP32 vs FP16).

**File:** [src/colosseum/deploy/core/policy.py:88-91](../src/colosseum/deploy/core/policy.py#L88-L91)

```python
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else [
    "CPUExecutionProvider"
]
session = ort.InferenceSession(str(model_path), providers=providers)
```

**Check:** Are session options set? Is precision consistent with training?

#### 2.4 Input/Output Tensor Handling

**Problem:** ONNX wrapper converts to numpy and back, potentially losing gradients or changing tensor properties.

**File:** [src/colosseum/deploy/core/policy.py:110-114](../src/colosseum/deploy/core/policy.py#L110-L114)

```python
def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    ort_inputs = {name: obs.detach().cpu().numpy() for name in self._input_names}  # ← To numpy
    outputs = self.session.run(self._output_names, ort_inputs)
    result = torch.from_numpy(outputs[0])  # ← Back to torch
    return result.to(obs.device)
```

**Potential issues:**
- Device mismatch (CPU vs CUDA)
- Dtype changes (float32 vs float64)
- Memory layout (contiguous vs strided)

### Test: Compare PyTorch vs ONNX Inference

Create `tests/test_onnx_consistency.py`:

```python
"""Test ONNX model outputs match PyTorch model outputs."""

import torch
import numpy as np
from pathlib import Path
from colosseum.deploy.core.registry import TASK_REGISTRY
from colosseum.deploy.backends.mujoco import MujocoController


def test_onnx_vs_pytorch():
    """Compare ONNX and PyTorch inference on same inputs."""

    # Load both models
    cfg = TASK_REGISTRY.get_config("t1-velocity-flat")

    # 1. Load ONNX model (deployment)
    controller = MujocoController(cfg)
    controller.update_state()
    controller.start()

    onnx_policy = controller.policy

    # 2. Load PyTorch model (training)
    # TODO: Load original .pt checkpoint
    # pytorch_policy = ...

    # 3. Generate test observations
    test_obs = torch.randn(1, 81)  # 81-dim velocity observation

    # 4. Run both models
    onnx_output = onnx_policy._model(test_obs)
    # pytorch_output = pytorch_policy(test_obs)

    # 5. Compare outputs
    # diff = (onnx_output - pytorch_output).abs().max()
    # assert diff < 1e-5, f"ONNX differs from PyTorch by {diff}"

    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"ONNX output range: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")
    print(f"ONNX output mean: {onnx_output.mean():.4f}")
    print(f"ONNX output std:  {onnx_output.std():.4f}")


def test_onnx_metadata():
    """Check ONNX model metadata."""
    import onnx

    onnx_path = Path("src/colosseum/tasks/velocity/deploy/t1_23dof/models/policy_flat.onnx")
    model = onnx.load(str(onnx_path))

    print(f"\nONNX Model Info:")
    print(f"  IR Version: {model.ir_version}")
    print(f"  Opset: {model.opset_import[0].version}")
    print(f"  Producer: {model.producer_name}")

    print(f"\nInputs:")
    for inp in model.graph.input:
        print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")

    print(f"\nOutputs:")
    for out in model.graph.output:
        print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")

    print(f"\nMetadata:")
    for prop in model.metadata_props:
        print(f"  {prop.key}: {prop.value}")


if __name__ == "__main__":
    test_onnx_metadata()
    test_onnx_vs_pytorch()
```

### Expected Outcome
- ✅ **ONNX == PyTorch:** Outputs match within 1e-5 tolerance → ONNX conversion is correct
- ⚠️ **ONNX != PyTorch:** Significant difference → ONNX export is broken

### Holosoma Comparison
Holosoma also uses ONNX export. Check their export configuration:
- File: `external/holosoma/src/holosoma/utils/onnx_export.py`
- Opset version, dynamic axes, provider settings

**Confidence:** 30% (needs investigation)

---

## 3. Observation Computation

### Description
Observations computed in deployment may differ from training due to wrong ordering, incorrect calculations, or missing components.

### Potential Issues

#### 3.1 Projected Gravity Calculation

**Problem:** `compute_projected_gravity` might use wrong quaternion convention or rotation direction.

**File:** [src/colosseum/mdp/observations.py:12-54](../src/colosseum/mdp/observations.py#L12-L54)

```python
def compute_projected_gravity(
    root_quat_w: torch.Tensor,
    gravity_w: torch.Tensor | None = None,
) -> torch.Tensor:
    if gravity_w is None:
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)

    # Uses isaaclab.math.quat_apply_inverse
    return lab_math.quat_apply_inverse(root_quat_w, gravity_w)
```

**Potential issues:**
- Quaternion convention: (w,x,y,z) vs (x,y,z,w)
- Rotation direction: forward vs inverse
- Gravity direction: -Z vs +Z

**Check Training:**
```python
# In mjlab velocity task, how is projected gravity computed?
# Does it match colosseum.mdp.observations.compute_projected_gravity?
```

#### 3.2 Observation Ordering

**Problem:** Deployment observation order might not match training.

**Training order** (from [velocity_env_cfg.py](../src/colosseum/tasks/velocity/config/t1_23dof/env_cfgs.py)):
```python
observations["policy"].terms = {
    "base_lin_vel": ...,      # (3,)
    "base_ang_vel": ...,      # (3,)
    "projected_gravity": ..., # (3,)
    "joint_pos_rel": ...,     # (23,) relative to default
    "joint_vel": ...,         # (23,)
    "last_action": ...,       # (23,)
    "velocity_command": ...,  # (3,)
}
# Total: 3+3+3+23+23+23+3 = 81
```

**Deployment order** (from [policy.py:77-88](../src/colosseum/tasks/velocity/deploy/t1_23dof/policy.py#L77-L88)):
```python
obs = torch.cat([
    base_lin_vel,       # (3,)
    base_ang_vel,       # (3,)
    projected_gravity,  # (3,)
    joint_pos_rel,      # (23,)
    joint_vel,          # (23,)
    last_action,        # (23,)
    vel_cmd,            # (3,)
], dim=-1)
```

**Check:** Does this EXACTLY match training? Are there any hidden terms?

#### 3.3 Joint Position Relative Calculation

**Problem:** "Relative to default" might be computed differently.

**Deployment** (from [policy.py:66-68](../src/colosseum/tasks/velocity/deploy/t1_23dof/policy.py#L66-L68)):
```python
joint_pos = self.robot.data.joint_pos[real2sim_map]
default_pos = self.robot.default_joint_pos[real2sim_map]
joint_pos_rel = joint_pos - default_pos
```

**Training:** How is joint_pos_rel computed in mjlab? Check `mjlab.tasks.velocity.mdp.observations`.

#### 3.4 Joint Order Remapping

**Problem:** real2sim_map might be incorrect, causing wrong joint order in observations.

**File:** [policy.py:45-46, 66-71](../src/colosseum/tasks/velocity/deploy/t1_23dof/policy.py#L45-L46)

```python
# Get joint mapping (real robot order → simulation order)
real2sim_map = self.robot.data.real2sim_joint_indexes

# Used for observations
joint_pos = self.robot.data.joint_pos[real2sim_map]
joint_vel = self.robot.data.joint_vel[real2sim_map]
```

**Check:** Is real2sim_map computed correctly? Print and verify.

#### 3.5 Velocity Command Scaling

**Problem:** Velocity commands might have different ranges/scales between training and deployment.

**Deployment** (from [policy.py:48-55](../src/colosseum/tasks/velocity/deploy/t1_23dof/policy.py#L48-L55)):
```python
vel_cmd = torch.tensor([
    self.vel_command.lin_vel_x,    # Range?
    self.vel_command.lin_vel_y,    # Range?
    self.vel_command.ang_vel_yaw,  # Range?
], dtype=torch.float32)
```

**Training:** What are the velocity command ranges in training? Are they normalized?

### Test: Observation Validation

Create `tests/test_observations.py`:

```python
"""Test observation computation matches training."""

import torch
from colosseum.deploy.core.registry import TASK_REGISTRY
from colosseum.deploy.backends.mujoco import MujocoController
from colosseum.tasks.velocity.mdp.observation_spec import VELOCITY_OBS_SPEC


def test_observation_structure():
    """Verify observation structure matches spec."""
    cfg = TASK_REGISTRY.get_config("t1-velocity-flat")
    controller = MujocoController(cfg)
    controller.update_state()
    controller.start()

    obs = controller.policy.compute_observation()

    print(f"Observation shape: {obs.shape}")
    print(f"Expected shape: (1, {VELOCITY_OBS_SPEC.total_size(23)})")

    # Validate structure
    VELOCITY_OBS_SPEC.validate_observation(obs.squeeze(0), num_joints=23)

    # Print components
    print(f"\nObservation breakdown:")
    offset = 0
    for name, size in VELOCITY_OBS_SPEC.ORDER:
        size_resolved = size if isinstance(size, int) else size(23)
        component = obs[0, offset:offset+size_resolved]
        print(f"  {name:<20} ({size_resolved:2d}): mean={component.mean():7.4f}, "
              f"std={component.std():7.4f}, range=[{component.min():7.4f}, {component.max():7.4f}]")
        offset += size_resolved


def test_projected_gravity_correctness():
    """Test projected gravity computation."""
    from colosseum.mdp.observations import compute_projected_gravity

    # Test case 1: Identity quaternion (upright)
    quat_upright = torch.tensor([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
    gravity = compute_projected_gravity(quat_upright)

    print(f"\nProjected Gravity Tests:")
    print(f"  Upright robot: {gravity.tolist()}")
    print(f"  Expected:      [0.0, 0.0, -1.0]")

    assert torch.allclose(gravity, torch.tensor([0.0, 0.0, -1.0]), atol=1e-5), \
        "Upright gravity should be [0, 0, -1]"

    # Test case 2: 90° pitch forward (nose down)
    quat_pitch = torch.tensor([0.7071, 0.7071, 0.0, 0.0])  # 90° about X-axis
    gravity = compute_projected_gravity(quat_pitch)

    print(f"  90° pitch:     {gravity.tolist()}")
    print(f"  Expected:      [0.0, 1.0, 0.0]")  # Gravity now points "forward" in base frame

    # Test case 3: 90° roll (on side)
    quat_roll = torch.tensor([0.7071, 0.0, 0.7071, 0.0])  # 90° about Y-axis
    gravity = compute_projected_gravity(quat_roll)

    print(f"  90° roll:      {gravity.tolist()}")
    print(f"  Expected:      [-1.0, 0.0, 0.0]")  # Gravity points "left" in base frame


def test_joint_order_remapping():
    """Verify joint order remapping is correct."""
    cfg = TASK_REGISTRY.get_config("t1-velocity-flat")
    controller = MujocoController(cfg)

    robot = controller.robot

    print(f"\nJoint Order Remapping:")
    print(f"  Real hardware order (23 joints):")
    for i, name in enumerate(robot.cfg.joint_names):
        print(f"    [{i:2d}] {name}")

    print(f"\n  Simulation order (23 joints):")
    for i, name in enumerate(robot.cfg.sim_joint_names):
        print(f"    [{i:2d}] {name}")

    print(f"\n  Remapping: real2sim_joint_indexes:")
    print(f"    {robot.data.real2sim_joint_indexes}")

    print(f"\n  Remapping: sim2real_joint_indexes:")
    print(f"    {robot.data.sim2real_joint_indexes}")

    # Verify bidirectional mapping
    for i in range(23):
        real_idx = i
        sim_idx = robot.data.real2sim_joint_indexes[real_idx]
        back_to_real = robot.data.sim2real_joint_indexes[sim_idx]

        assert back_to_real == real_idx, \
            f"Mapping broken at index {i}: {real_idx} → {sim_idx} → {back_to_real}"

    print(f"\n  ✓ Bidirectional mapping verified")


if __name__ == "__main__":
    test_observation_structure()
    test_projected_gravity_correctness()
    test_joint_order_remapping()
```

### Expected Outcome
- ✅ **All tests pass:** Observations computed correctly
- ⚠️ **Projected gravity wrong:** Quaternion convention issue
- ⚠️ **Joint order wrong:** Remapping broken
- ⚠️ **Observation order wrong:** Training/deployment mismatch

**Confidence:** 40% (needs investigation)

---

## 4. Action Scaling

### Description
Action scaling formula in deployment might differ from training.

### Training
Uniform 0.25 scaling: `action * 0.25 + default_pos`

### Deployment
Per-joint scaling: `action * (0.25 * effort_limit / stiffness) + default_pos`

**File:** [src/colosseum/deploy/core/policy.py:55-59](../src/colosseum/deploy/core/policy.py#L55-L59)

### Test
```bash
pixi run python tests/test_deployment_diagnostics.py
```

Look for:
```
⚠️  Action scaling is PER-JOINT (varies by 45.2%)
```

### Expected Outcome
- ⚠️ Per-joint scaling differs from uniform 0.25 → **Mismatch with training**

**Confidence:** 40%

---

## 5. Physics Solver Settings

### Description
MuJoCo solver iterations may use defaults instead of training values.

### Training
- Iterations: 10
- LS Iterations: 20

### Deployment
- Iterations: 50 (MuJoCo default)
- LS Iterations: 10 (MuJoCo default)

### Test
```bash
pixi run python tests/test_deployment_diagnostics.py
```

Look for:
```
⚠️  Iterations mismatch: 50 vs 10
```

### Fix
```python
# In mujoco.py after model compilation
self.mj_model.opt.iterations = 10
self.mj_model.opt.ls_iterations = 20
```

**Confidence:** 15%

---

## 6. PD Gains

### Description
Deployment PD gains might not match training.

### Status
✅ **VERIFIED CORRECT** - Deployment config matches motor specifications exactly.

**Confidence:** 5% (unlikely to be the issue)

---

## 7. Joint Order Remapping

### Description
Joint order conversion between hardware and simulation might be incorrect.

### Implementation
**File:** [src/colosseum/deploy/core/robot.py](../src/colosseum/deploy/core/robot.py)

Computes bidirectional mappings:
- `real2sim_joint_indexes`: hardware → simulation order
- `sim2real_joint_indexes`: simulation → hardware order

### Test
```bash
pixi run python tests/test_observations.py
```

### Expected Outcome
- ✅ **Mapping verified:** Bidirectional mapping correct
- ⚠️ **Mapping broken:** Indices don't match joint names

**Confidence:** 20%

---

## 8. Actuator Configuration

### Description
MuJoCo actuators might not match training configuration.

### Status
✅ **VERIFIED** - Uses same `create_position_actuator()` function as training.

**Confidence:** 10%

---

## 9. Ground Contact / Init Height

### Description
Robot initialization height might cause feet to float above ground.

### Status
✅ **FIXED** - Init height raised from 0.665m to 0.70m to ensure ground contact.

**Confidence:** 5%

---

## 10. Decimation / Physics Timestep

### Description
Policy frequency or physics timestep might differ from training.

### Status
✅ **VERIFIED MATCHING**
- Physics: 0.005s (200Hz)
- Decimation: 4x
- Policy: 0.02s (50Hz)

**Confidence:** 5%

---

## Summary: Test Priority Order

### 🚀 Run First (High Confidence)
1. **Sleep Timing Test** (85% confidence)
   ```bash
   pixi run python tests/test_sleep_timing.py
   ```
   Expected: Fixed sleep = FALLING, Minimal/Adaptive = STABLE

---

### 🔍 Run Second (Medium Confidence)
2. **Observation Validation** (40% confidence)
   ```bash
   pixi run python tests/test_observations.py
   ```
   Check: Projected gravity, joint order, observation structure

3. **ONNX Consistency** (30% confidence)
   ```bash
   pixi run python tests/test_onnx_consistency.py
   ```
   Compare: ONNX vs PyTorch inference outputs

---

### 📊 Run Third (Full Analysis)
4. **Full Diagnostics** (catches all issues)
   ```bash
   pixi run python tests/test_deployment_diagnostics.py
   ```
   Check: Action scaling, solver settings, actuators

---

## Quick Checklist

Use this when running tests:

- [ ] **Timing:** Fixed sleep causes falling? → [Test 1](#1-control-loop-timing)
- [ ] **ONNX:** Model export correct? → [Test 2](#2-onnx-model-conversion)
- [ ] **Observations:** Computed correctly? → [Test 3](#3-observation-computation)
  - [ ] Projected gravity direction
  - [ ] Joint order remapping
  - [ ] Observation ordering
  - [ ] Relative joint positions
- [ ] **Action Scaling:** Uniform or per-joint? → [Test 4](#4-action-scaling)
- [ ] **Solver:** Iterations match training? → [Test 5](#5-physics-solver-settings)
- [ ] **PD Gains:** Match motor specs? → [Test 6](#6-pd-gains) ✅ Already verified
- [ ] **Actuators:** Created correctly? → [Test 8](#8-actuator-configuration) ✅ Already verified
- [ ] **Init Height:** Feet on ground? → [Test 9](#9-ground-contact--init-height) ✅ Already fixed

---

## Expected Timeline

- **Timing test:** 20 seconds
- **Observation tests:** 10 seconds
- **ONNX test:** 30 seconds (need to create)
- **Full diagnostics:** 40 seconds
- **Total:** ~2 minutes to run all tests

---

## Next Steps After Testing

1. **If timing confirmed:** Implement adaptive timing fix (Priority 1)
2. **If ONNX issues found:** Re-export with correct settings or use TorchScript
3. **If observation issues:** Fix projected gravity or joint order
4. **If multiple issues:** Address in priority order (timing first)

Good luck debugging! 🐛🔍
