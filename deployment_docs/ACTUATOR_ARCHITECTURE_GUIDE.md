# MuJoCo Actuator Architecture Guide for Colosseum Deployment

**Decision Guide: Position vs Motor Actuators for Clean, Working Deployment**

**Last Updated:** 2026-01-21

---

## Executive Summary

Colosseum deployment currently uses a **hybrid approach** that's neither clean nor optimal:
- ✅ **Has:** Position actuators (created programmatically)
- ❌ **But:** Manually computes PD torques in Python (redundant!)
- 🤔 **Result:** Confusion and potential bugs

**This document provides two clean architectures:**
1. **Option A (RECOMMENDED):** Pure motor actuators + manual PD (matches booster_deploy, best sim-to-real)
2. **Option B:** Pure position actuators (simplest, let MuJoCo handle PD)

---

## Table of Contents

1. [Current Problem](#current-problem)
2. [MuJoCo Actuator Types Explained](#mujoco-actuator-types-explained)
3. [Three System Comparison](#three-system-comparison)
4. [Option A: Motor Actuators (RECOMMENDED)](#option-a-motor-actuators-recommended)
5. [Option B: Position Actuators](#option-b-position-actuators)
6. [Decision Matrix](#decision-matrix)
7. [Implementation Guide](#implementation-guide)
8. [Migration Path](#migration-path)

---

## Current Problem

### What Colosseum Does Now (Broken Hybrid)

**Step 1:** Creates position actuators programmatically
```python
# src/colosseum/deploy/backends/mujoco.py:_add_position_actuators_from_cfg()
actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
actuator.gainprm[0] = kp           # Position actuator setup
actuator.biasprm[1] = -kp
actuator.biasprm[2] = -kd
```

**Step 2:** Then manually computes PD torques in `ctrl_step()`
```python
def ctrl_step(self, dof_targets: torch.Tensor):
    for _ in range(self.decimation):
        # Manually compute PD torques (redundant with position actuators!)
        tau = kp * (dof_targets - dof_pos) - kd * dof_vel
        tau = np.clip(tau, -effort, effort)

        # Pass torques to position actuators (bypasses their PD control!)
        self.mj_data.ctrl[:] = tau  # Confusing!
```

**Why This Is Wrong:**
- Position actuators expect **position targets** in `ctrl[]`
- Code computes **torques** and puts them in `ctrl[]`
- Actuators try to apply PD on torques (nonsensical!)
- Works accidentally because `forcelimited=True` clamps output
- State updates **NOT per-substep** (static targets)

**Result:** "Heavy feet" + architectural confusion!

---

## MuJoCo Actuator Types Explained

### Position Actuator (Built-in PD Servo)

**What it does:** MuJoCo implements a PD controller for you

**XML Definition:**
```xml
<position name="joint_servo" joint="joint_name" kp="100" kv="10" forcerange="-50 50"/>
```

**Programmatic Equivalent:**
```python
actuator = spec.add_actuator()
actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
actuator.gainprm[0] = kp
actuator.biasprm[1] = -kp
actuator.biasprm[2] = -kd
```

**How it works:**
```
User: ctrl[i] = target_position (e.g., 0.5 radians)
      ↓
MuJoCo (every physics step @ 200Hz):
      τ = kp * (ctrl[i] - q) - kd * qd
      τ_clamped = clip(τ, -forcerange, +forcerange)
      ↓
Apply τ_clamped to joint
```

**Control flow in your code:**
```python
# Set target position
mj_data.ctrl[i] = target_position

# MuJoCo automatically does:
# - Read current state (q, qd)
# - Compute τ = kp * (target - q) - kd * qd
# - Clamp and apply
```

**Pros:**
- ✅ Simple interface (just set target positions)
- ✅ MuJoCo handles PD (native C++, optimized)
- ✅ No Python overhead

**Cons:**
- ❌ PD computed with state from **start of mj_step()**, not real-time
- ❌ Less control over PD computation
- ❌ Doesn't match real hardware SDK (which updates state continuously)

---

### Motor Actuator (Direct Torque)

**What it does:** Pass torques directly to joints (you compute PD in Python)

**XML Definition:**
```xml
<motor name="joint_motor" joint="joint_name" ctrlrange="-50 50" ctrllimited="true"/>
```

**Programmatic Equivalent:**
```python
actuator = spec.add_actuator()
actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
actuator.gainprm[0] = 1.0  # Unity gain (passthrough)
```

**How it works:**
```
User: Compute τ = kp * (target - q) - kd * qd in Python
      ↓
      ctrl[i] = τ
      ↓
MuJoCo (every physics step):
      τ_out = ctrl[i]  (direct passthrough)
      τ_clamped = clip(τ_out, ctrlrange)
      ↓
Apply τ_clamped to joint
```

**Control flow in your code:**
```python
# You compute PD manually
tau = kp * (target_position - dof_pos) - kd * dof_vel
tau_clamped = np.clip(tau, -effort_limit, +effort_limit)

# Pass torques to MuJoCo
mj_data.ctrl[i] = tau_clamped

# MuJoCo just applies them (no additional PD)
```

**Pros:**
- ✅ Full control over PD computation
- ✅ Can update state **per-substep** (dynamic feedback!)
- ✅ Matches real hardware SDK behavior
- ✅ Better sim-to-real transfer

**Cons:**
- ❌ More code (manual PD implementation)
- ❌ Python overhead (but negligible for 23 DOF)

---

## Three System Comparison

### 1. Colosseum Deployment (Current - BROKEN)

**Actuator Type:** Position (programmatic)

**Control Flow:**
```python
# Create position actuators
_add_position_actuators_from_cfg(spec, robot_cfg)

# But then compute PD manually (redundant!)
def ctrl_step(self, dof_targets):
    for _ in range(decimation):
        tau = kp * (dof_targets - dof_pos) - kd * dof_vel  # Manual PD
        self.mj_data.ctrl = tau  # Pass torques to position actuators!
        mujoco.mj_step(model, data)
        # No per-substep state update! ❌
```

**Problems:**
- Position actuators expect positions, get torques
- No per-substep state updates
- Confusing hybrid approach

---

### 2. booster_deploy (WORKING - Reference)

**Actuator Type:** Motor (XML-defined)

**Control Flow:**
```python
# XML defines motor actuators
# <motor name="Left_Hip_Pitch" joint="Left_Hip_Pitch" ctrlrange="-45 45"/>

def ctrl_step(self, dof_targets):
    dof_pos = self.mj_data.qpos[7:]
    dof_vel = self.mj_data.qvel[6:]
    kp = self.robot.joint_stiffness
    kd = self.robot.joint_damping

    for i in range(decimation):
        # Manual PD with CURRENT state ⭐
        tau = kp * (dof_targets - dof_pos) - kd * dof_vel
        tau = np.clip(tau, -effort_limit, +effort_limit)

        self.mj_data.ctrl = tau
        mujoco.mj_step(model, data)

        # CRITICAL: Update state for next substep ⭐
        dof_pos = self.mj_data.qpos[7:]
        dof_vel = self.mj_data.qvel[6:]
```

**Why It Works:**
- ✅ Motor actuators accept torques (clean interface)
- ✅ Per-substep state updates (dynamic feedback)
- ✅ Matches real hardware SDK (PD with fresh sensor data)

---

### 3. holosoma Training (GPU-Optimized)

**Actuator Type:** Motor (XML-defined)

**Control Flow:**
```python
# XML: <motor name="Left_Hip_Pitch" ... />

# PyTorch PD computation (batched, GPU)
def apply_actions(actions: torch.Tensor):  # [4096 envs, 29 actions]
    torques = kp * (actions + default_pos - dof_pos) - kd * dof_vel
    torques = torch.clamp(torques, -effort_limit, +effort_limit)

    # Zero-copy GPU write to MuJoCo Warp
    ctrl_tensor[:] = torques  # [4096, 29] GPU tensor

    # MuJoCo Warp steps all 4096 envs in parallel
```

**Why It's Different:**
- GPU-batched computation across thousands of environments
- Zero-copy tensor operations
- Domain randomization of PD gains per environment
- Not applicable to single-environment deployment

---

## Option A: Motor Actuators (RECOMMENDED)

**Why This Is Best for Deployment:**
- ✅ Matches booster_deploy (proven working reference)
- ✅ Matches real hardware SDK (better sim-to-real)
- ✅ Full control over PD computation
- ✅ Clean architecture (explicit is better than implicit)
- ✅ Per-substep state updates (dynamic feedback)

### Implementation

**Step 1: Create Motor Actuators (Programmatic)**

```python
def _add_motor_actuators_from_cfg(spec: mujoco.MjSpec, robot_cfg: RobotConfig) -> None:
    """Add one motor actuator per joint for direct torque control.

    This approach:
    - Gives full control over PD computation
    - Matches booster_deploy implementation
    - Better sim-to-real transfer (matches hardware SDK)
    """
    if len(spec.actuators) > 0:
        return  # Actuators already present in XML

    for i, joint_name in enumerate(robot_cfg.sim_joint_names):
        actuator = spec.add_actuator(name=f"{joint_name}_motor", target=joint_name)

        # Configure as motor actuator (direct torque)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_NONE

        # Unity gain (passthrough)
        actuator.gainprm[0] = 1.0

        # Torque limits (motor accepts torques in ctrl[])
        effort_limit = float(robot_cfg.effort_limit[i])
        actuator.ctrllimited = True
        actuator.ctrlrange[:] = np.array([-effort_limit, effort_limit], dtype=np.float64)

        # Set joint armature (reflected inertia) - CRITICAL for stability!
        armature = float(robot_cfg.joint_armature[i])
        spec.joint(joint_name).armature = armature
```

**Step 2: Manual PD Control with Per-Substep Updates**

```python
def ctrl_step(self, dof_targets: torch.Tensor) -> None:
    """Apply joint targets using manual PD torque control.

    This implementation:
    - Matches booster_deploy exactly
    - Updates state every substep (dynamic feedback)
    - Computes PD with fresh sensor data
    """
    dof_targets = dof_targets.cpu().numpy()

    # Update velocity commands if needed
    if self.vel_command is not None:
        self.update_command()

    # Get initial state
    dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
    dof_vel = self.mj_data.qvel.astype(np.float32)[6:]

    # Get PD gains and limits from config
    kp = np.asarray(self.robot.cfg.joint_stiffness, dtype=np.float32)
    kd = np.asarray(self.robot.cfg.joint_damping, dtype=np.float32)
    effort_limit = np.asarray(self.robot.cfg.effort_limit, dtype=np.float32)

    # Manual PD control with per-substep state updates
    for _ in range(self.cfg.mujoco.decimation):
        # Compute PD torque based on CURRENT state ⭐
        tau = kp * (dof_targets - dof_pos) - kd * dof_vel
        tau = np.clip(tau, -effort_limit, effort_limit)

        # Apply torques (motor actuators pass through directly)
        self.mj_data.ctrl[:] = tau

        # Step physics
        mujoco.mj_step(self.mj_model, self.mj_data)

        # CRITICAL: Update state for next substep ⭐
        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
```

**Step 3: Update MujocoController.__init__()**

```python
def __init__(self, cfg: ControllerConfig):
    super().__init__(cfg)

    # Load MJCF
    spec = mujoco.MjSpec.from_file(self.robot.cfg.mjcf_path)

    # Clear any existing actuators (use programmatic definition)
    spec.actuators.clear()

    # Add ground plane
    ground = spec.worldbody.add_geom()
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.friction[:] = [1.0, 0.005, 0.0001]

    # Add motor actuators (NOT position!) ⭐
    _add_motor_actuators_from_cfg(spec, self.robot.cfg)

    # Compile
    self.mj_model = spec.compile()
    self.mj_model.opt.timestep = self.cfg.physics_dt
    self.mj_data = mujoco.MjData(self.mj_model)

    # Set initial pose
    self.mj_data.qpos[:] = np.concatenate([
        cfg.mujoco.init_pos,           # (0.0, 0.0, 0.70)
        cfg.mujoco.init_quat,          # (1.0, 0.0, 0.0, 0.0)
        self.robot.default_joint_pos,  # (23,) standing pose
    ])
```

### Pros and Cons

**Pros:**
- ✅ Clean, explicit architecture
- ✅ Matches working reference (booster_deploy)
- ✅ Per-substep state updates (responsive control)
- ✅ Better sim-to-real transfer
- ✅ Full control over PD computation
- ✅ Easy to debug (PD logic in Python)

**Cons:**
- ⚠️ More code than position actuators
- ⚠️ Python overhead (negligible for 23 DOF @ 50Hz)
- ⚠️ Must implement PD manually

---

## Option B: Position Actuators

**When This Makes Sense:**
- You trust MuJoCo's PD implementation
- You want simplest possible code
- You don't care about matching hardware SDK exactly
- You're only doing sim-to-sim testing (no real robot deployment)

### Implementation

**Step 1: Keep Position Actuators (Current)**

```python
# src/colosseum/deploy/backends/mujoco.py
# Keep _add_position_actuators_from_cfg() as-is
```

**Step 2: Simplify ctrl_step() - Let MuJoCo Do PD**

```python
def ctrl_step(self, dof_targets: torch.Tensor) -> None:
    """Apply joint targets using MuJoCo position actuators.

    SIMPLE: Position actuators handle PD control internally.
    Policy just provides target positions.
    """
    dof_targets = dof_targets.cpu().numpy()

    # Update velocity commands if needed
    if self.vel_command is not None:
        self.update_command()

    # Decimation loop (4 physics steps @ 200Hz per policy step @ 50Hz)
    for _ in range(self.cfg.mujoco.decimation):
        # Set target positions (MuJoCo computes PD internally)
        self.mj_data.ctrl[:] = dof_targets

        # MuJoCo automatically:
        # 1. Reads current state (q, qd)
        # 2. Computes τ = kp * (ctrl - q) - kd * qd
        # 3. Clamps to forcerange
        # 4. Applies torque
        mujoco.mj_step(self.mj_model, self.mj_data)
```

**That's it! Much simpler.**

### Pros and Cons

**Pros:**
- ✅ Simplest possible implementation
- ✅ MuJoCo handles PD (native C++, optimized)
- ✅ Fewer lines of code
- ✅ Less Python overhead

**Cons:**
- ❌ PD computed once at start of mj_step() (not per-substep)
- ❌ Less responsive (static targets for 4 substeps)
- ❌ Doesn't match hardware SDK
- ❌ Worse sim-to-real transfer
- ❌ Less control over PD computation

---

## Decision Matrix

| Criterion | Motor Actuators (Option A) | Position Actuators (Option B) |
|-----------|---------------------------|-------------------------------|
| **Code Complexity** | Medium (manual PD) | Low (MuJoCo handles PD) |
| **Sim-to-Real Transfer** | ✅ Excellent (matches SDK) | ⚠️ Poor (different control flow) |
| **Responsiveness** | ✅ High (per-substep updates) | ⚠️ Low (static targets) |
| **Debugging** | ✅ Easy (PD in Python) | ⚠️ Harder (PD in MuJoCo C++) |
| **Performance** | ✅ Good (minimal Python overhead) | ✅ Excellent (native C++) |
| **Matches booster_deploy** | ✅ Yes (identical) | ❌ No |
| **Matches holosoma training** | ✅ Yes (same actuator type) | ❌ No |
| **Best For** | Deployment → real robot | Pure sim-to-sim only |

### Recommendation

**Use Motor Actuators (Option A)** if:
- ✅ You plan to deploy to real hardware eventually
- ✅ You want to match the working booster_deploy reference
- ✅ You want maximum control and transparency
- ✅ You value sim-to-real consistency

**Use Position Actuators (Option B)** if:
- You only care about sim-to-sim testing
- You want absolute minimum code complexity
- You trust MuJoCo's PD implementation
- You don't plan real robot deployment

**My Strong Recommendation: Option A (Motor Actuators)**
- Fixes "heavy feet" issue (per-substep updates)
- Matches proven working reference
- Better architecture for your goal (sim2sim → sim2real)

---

## Implementation Guide

### Complete Code Changes for Option A

**File:** `src/colosseum/deploy/backends/mujoco.py`

**Replace:**
```python
# REMOVE this function entirely
def _add_position_actuators_from_cfg(spec, robot_cfg):
    ...
```

**Add:**
```python
def _add_motor_actuators_from_cfg(spec: mujoco.MjSpec, robot_cfg: RobotConfig) -> None:
    """Add motor actuators for direct torque control (matches booster_deploy)."""
    if len(spec.actuators) > 0:
        return  # Actuators already in XML

    for i, joint_name in enumerate(robot_cfg.sim_joint_names):
        actuator = spec.add_actuator(name=f"{joint_name}_motor", target=joint_name)

        # Motor actuator (direct torque passthrough)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
        actuator.gainprm[0] = 1.0  # Unity gain

        # Torque limits
        effort_limit = float(robot_cfg.effort_limit[i])
        actuator.ctrllimited = True
        actuator.ctrlrange[:] = np.array([-effort_limit, effort_limit], dtype=np.float64)

        # Joint armature
        armature = float(robot_cfg.joint_armature[i])
        spec.joint(joint_name).armature = armature
```

**Update ctrl_step():**
```python
def ctrl_step(self, dof_targets: torch.Tensor) -> None:
    """Manual PD control with per-substep state updates (matches booster_deploy)."""
    dof_targets = dof_targets.cpu().numpy()

    if self.vel_command is not None:
        self.update_command()

    # Get initial state
    dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
    dof_vel = self.mj_data.qvel.astype(np.float32)[6:]

    # Get PD parameters
    kp = np.asarray(self.robot.cfg.joint_stiffness, dtype=np.float32)
    kd = np.asarray(self.robot.cfg.joint_damping, dtype=np.float32)
    effort_limit = np.asarray(self.robot.cfg.effort_limit, dtype=np.float32)

    # Manual PD with per-substep updates
    for _ in range(self.cfg.mujoco.decimation):
        # Compute torque from CURRENT state
        tau = kp * (dof_targets - dof_pos) - kd * dof_vel
        tau = np.clip(tau, -effort_limit, effort_limit)

        # Apply torques
        self.mj_data.ctrl[:] = tau

        # Step physics
        mujoco.mj_step(self.mj_model, self.mj_data)

        # Update state for next substep
        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
```

**Update __init__():**
```python
def __init__(self, cfg: ControllerConfig):
    super().__init__(cfg)

    spec = mujoco.MjSpec.from_file(self.robot.cfg.mjcf_path)
    spec.actuators.clear()

    # Add ground
    ground = spec.worldbody.add_geom()
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.friction[:] = [1.0, 0.005, 0.0001]

    # Add motor actuators (NOT position!)
    _add_motor_actuators_from_cfg(spec, self.robot.cfg)  # ⭐ Changed!

    self.mj_model = spec.compile()
    self.mj_model.opt.timestep = self.cfg.physics_dt
    self.mj_data = mujoco.MjData(self.mj_model)

    # Set initial pose
    self.mj_data.qpos[:] = np.concatenate([
        cfg.mujoco.init_pos,
        cfg.mujoco.init_quat,
        self.robot.default_joint_pos,
    ])
```

### Complete Code Changes for Option B

**File:** `src/colosseum/deploy/backends/mujoco.py`

**Keep _add_position_actuators_from_cfg() as-is**

**Simplify ctrl_step():**
```python
def ctrl_step(self, dof_targets: torch.Tensor) -> None:
    """Position actuator control (MuJoCo handles PD internally)."""
    dof_targets = dof_targets.cpu().numpy()

    if self.vel_command is not None:
        self.update_command()

    # Simple: Just set target positions
    for _ in range(self.cfg.mujoco.decimation):
        self.mj_data.ctrl[:] = dof_targets
        mujoco.mj_step(self.mj_model, self.mj_data)
```

**No other changes needed!**

---

## Migration Path

### Step 1: Backup Current Implementation

```bash
cd /home/neverorfrog/code/spqr/colosseum
git checkout -b feature/motor-actuators
git add src/colosseum/deploy/backends/mujoco.py
git commit -m "Backup: Before actuator refactoring"
```

### Step 2: Choose Your Option

**For Option A (Motor Actuators - RECOMMENDED):**
```bash
# Apply changes from "Implementation Guide - Option A"
# Test immediately
pixi run deploy -b mujoco -t t1-velocity-rough
```

**For Option B (Position Actuators):**
```bash
# Apply changes from "Implementation Guide - Option B"
# Test immediately
pixi run deploy -b mujoco -t t1-velocity-rough
```

### Step 3: Verify Behavior

**Expected Results with Option A:**
- ✅ Robot feels more responsive
- ✅ "Heavy feet" reduced significantly (50-70%)
- ✅ Matches booster_deploy behavior
- ✅ Same control flow as real hardware SDK

**Expected Results with Option B:**
- ✅ Simpler code
- ⚠️ Still feels "heavy" (no per-substep updates)
- ⚠️ Different from booster_deploy

### Step 4: Combine with PD Gain Fix

**After verifying actuator changes work, retrain with holosoma gains:**
```bash
# Update actuators.py with holosoma T1 29-DOF gains
# (See DEPLOYMENT_ARCHITECTURE_COMPARISON.md)

pixi run python -m mjlab.scripts.train --task=velocity-t1-23dof
```

---

## FAQ

### Q: Why does booster_deploy work without defining actuators programmatically?

**A:** booster_deploy loads XML files that **already contain motor actuators** (from `booster_assets` package). The actuators are pre-defined in the robot MJCF.

### Q: Why doesn't Colosseum's T1_23dof.xml have actuators?

**A:** Design choice for clean separation:
- **XML:** Mechanical structure only (joints, bodies, geoms)
- **Python:** Control configuration (actuators, gains, limits)

This makes sense when gains are config-driven. But requires programmatic actuator creation.

### Q: Can I use XML-defined actuators instead of programmatic?

**A:** Yes! Add to `T1_23dof.xml`:
```xml
<actuator>
  <motor name="AAHead_yaw_motor" joint="AAHead_yaw" ctrlrange="-7 7" ctrllimited="true"/>
  <motor name="Head_pitch_motor" joint="Head_pitch" ctrlrange="-7 7" ctrllimited="true"/>
  <!-- ... 23 more actuators ... -->
</actuator>
```

**But:** Gains would be in XML, duplicating your config. Programmatic is cleaner for deployment.

### Q: What about holosoma's approach (XML actuators)?

**A:** holosoma uses XML actuators because:
- Training has **static gains** per robot (not runtime configurable)
- GPU training needs standalone MJCFs (no programmatic modification)
- Gains are embedded in ONNX at export time

For deployment, you could adopt the same: store gains in ONNX metadata, load at runtime.

### Q: Will Option A slow down deployment?

**A:** No! Python PD computation for 23 DOF @ 50Hz is negligible (<0.1ms per step). The per-substep state updates are worth the tiny overhead.

---

## Summary

**Current State:**
- ❌ Hybrid approach (position actuators + manual PD)
- ❌ No per-substep state updates
- ❌ Confusing architecture

**Recommended Fix:**
- ✅ **Option A: Motor actuators + manual PD** (matches booster_deploy)
- ✅ Per-substep state updates
- ✅ Clean, explicit architecture
- ✅ Better sim-to-real transfer

**Alternative:**
- Option B: Position actuators (simplest, but less responsive)

**Action Item:**
1. Choose Option A (motor actuators) - RECOMMENDED
2. Apply code changes from Implementation Guide
3. Test: `pixi run deploy -b mujoco -t t1-velocity-rough`
4. Combine with holosoma PD gains (from DEPLOYMENT_ARCHITECTURE_COMPARISON.md)
5. Retrain policy

This will give you a **clean, working, elegant architecture** that matches the proven booster_deploy reference! 🚀
