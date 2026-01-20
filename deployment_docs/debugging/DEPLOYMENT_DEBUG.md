# Deployment Debug Analysis

## Summary
Comparing working play script with deployment controller to identify issues.

## Key Findings

### 1. **DEFAULT JOINT POSITIONS MISMATCH** ⚠️ CRITICAL
- **Training (constants.py)**:
  ```python
  HOME_QPOS = {
      "Left_Hip_Pitch": -0.2,
      "Left_Knee_Pitch": 0.4,
      "Left_Ankle_Pitch": -0.2,
      # ... (bent knees for standing)
  }
  ```
- **Deployment (deploy_config.py)**:
  ```python
  default_joint_pos = (
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Left leg STRAIGHT
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Right leg STRAIGHT
      # ...
  }
  ```
- **Impact**: Joint position observations are `joint_pos - default_pos`. Different defaults = wrong observations!
- **Fix**: Update deployment `default_joint_pos` to match `HOME_QPOS` (in real robot order)

### 2. **PD GAINS MISMATCH** ⚠️ HIGH PRIORITY
- **Training actuators** (computed from motor specs with ω_n=10Hz, ζ=2.0):
  - Neck (15Hz): Kp=15.99, Kd=0.68
  - Arms (12Hz): Kp=160.61, Kd=8.52
  - Waist: Kp=188.76, Kd=12.02
  - Hip Pitch: Kp=206.83, Kd=13.17
  - Hip Roll/Yaw: Kp=188.76, Kd=12.02
  - Knee: Kp=251.09, Kd=15.98
  - Ankle Pitch: Kp=134.05, Kd=8.53
  - Ankle Roll: Kp=134.05, Kd=8.53

- **Deployment config** (deploy_config.py): MATCHES! ✅

### 3. **JOINT ORDERING** ⚠️ NEEDS VERIFICATION
- **Training**: MuJoCo assigns indices alphabetically when compiling XML
- **Deployment**: `sim_joint_names` in deploy_config.py assumes alphabetical ordering
- **Current sim_joint_names**:
  ```python
  ("AAHead_yaw", "Left_Shoulder_Pitch", "Right_Shoulder_Pitch", "Waist", ...)
  ```
- **Question**: Is this actually alphabetical? Need to verify against compiled MuJoCo model.

### 4. **ACTION SCALING** ✅ CORRECT
- Training: `ACTION_SCALE = 0.25` (uniform for all joints)
- Deployment: `action_scale = 0.25` in config.py
- **Status**: MATCHES

### 5. **OBSERVATION COMPUTATION**
Comparing training wrappers vs deployment policy:

#### Training (observations.py + wrappers):
```python
# From ManagerBasedRlEnv with Entity abstraction
base_lin_vel = asset.data.root_lin_vel_b  # (num_envs, 3)
base_ang_vel = asset.data.root_ang_vel_b  # (num_envs, 3)
projected_gravity = asset.data.projected_gravity_b  # (num_envs, 3)
joint_pos_rel = asset.data.joint_pos - asset.data.default_joint_pos
joint_vel = asset.data.joint_vel
last_action = actions (stored)
velocity_commands = commands["twist"]
```

#### Deployment (policy.py):
```python
# From MuJoCo sensors + manual computation
base_lin_vel = mj_data.sensor("imu_lin_vel").data
base_ang_vel = mj_data.sensor("imu_ang_vel").data
projected_gravity = compute_projected_gravity(root_quat_w)
joint_pos_rel = joint_pos[real2sim_map] - default_pos[real2sim_map]  # ⚠️ depends on correct default_pos
joint_vel = joint_vel[real2sim_map]
last_action = stored
velocity_commands = vel_command object
```

**Issue**: If `default_joint_pos` is wrong, `joint_pos_rel` will be completely incorrect!

### 6. **ACTUATOR TYPE** ✅ CORRECT
- Training: `BuiltinPositionActuatorCfg` (MuJoCo applies internal PD)
- Deployment: `create_position_actuator` (MuJoCo applies internal PD)
- **Status**: BOTH USE POSITION ACTUATORS

## Root Cause Analysis

The most likely culprits in order of severity:

### 1. **DEFAULT POSE MISMATCH** (99% likely)
The deployment uses straight legs (all zeros) while training used bent knees. This means:
- `joint_pos_rel` observations are shifted by ~0.2-0.4 radians for leg joints
- Policy receives completely different inputs than it was trained on
- Robot behavior will be unpredictable/unstable

### 2. **JOINT ORDERING** (50% likely if sim_joint_names is wrong)
If `sim_joint_names` doesn't match MuJoCo's alphabetical ordering:
- `real2sim_joint_indexes` mapping will be incorrect
- Observations will have joints in wrong order
- Actions will be sent to wrong joints

### 3. **ONNX CONVERSION** (20% likely)
- Less likely if play.py works with .pt checkpoint
- Could be normalization issues or export bugs
- Would need to compare .pt vs .onnx outputs

## Recommended Fix Priority

1. **IMMEDIATE**: Fix default_joint_pos in deploy_config.py
   - Match HOME_QPOS from constants.py (convert to real robot order)
   - Current: All zeros
   - Correct: Bent knees (-0.2, 0.4, -0.2 for hip/knee/ankle)

2. **VERIFY**: Check sim_joint_names ordering
   - Load compiled MuJoCo model
   - Print actual joint names in order
   - Verify alphabetical sorting matches deploy_config.py

3. **TEST**: Compare ONNX vs PT inference
   - Load both models
   - Feed same observation
   - Verify outputs match

## Next Steps
1. Create verification script to check MuJoCo joint ordering
2. Update deploy_config.py with correct default_joint_pos
3. Test deployment again
4. If still broken, verify ONNX export
