# Deployment Stability Fix

## Problem Summary

When starting the MuJoCo simulation:
1. **Legs shaking** - oscillating motion even at rest
2. **Falls forward on velocity command** - robot collapses instead of moving feet

## Diagnostic Results

Running the control loop stability test revealed:

### Control Loop Oscillations
```
Joint      Final Pos Error      Max |Vel|      Status
-----------------------------------------------------------
j0 (Head_yaw)     0.0070          2.81 rad/s    ❌ OSCILLATING
j1 (Head_pitch)   0.0091          1.30 rad/s    ❌ OSCILLATING
j2 (L_Shoulder)   0.0153          0.64 rad/s    ✓ Stable
```

Head and arm joints are oscillating at 2.8 and 1.3 rad/s when trying to reach their targets.

### Sensor Output Issues
```
Orientation: [0.999974, -0.000554, 0.000369, 0.002197]  ✓ Good
Lin velocity (X,Y,Z): [0.040, 0.013, -5.704] m/s        ❌ CRITICAL
  - Z velocity should be ~0, but is -5.7 m/s (falling!)
  - Std dev in Z: 0.567 (massive noise)
  
Angular velocity: [-0.008, -0.347, -0.193] rad/s        ❌ High
```

**The robot is falling even at rest, with Z-velocity of -5.7 m/s!**

## Root Cause Analysis

The ankle joints had **insufficient damping** to stabilize the vertical oscillations:

- **Ankle Pitch Kd**: 8.53 Nm·s/rad (too low)
- **Ankle Roll Kd**: 8.53 Nm·s/rad (too low)

The ankle joints couldn't dissipate energy fast enough, causing:
1. Continuous falling motion (negative Z velocity)
2. Head/arm oscillations trying to compensate for instability
3. Feet not providing stable support
4. Policy unable to execute walking commands (no stable base)

## Applied Fixes

### 1. Doubled Ankle Damping
```
Left_Ankle_Pitch  Kd: 8.53  → 17.06  (2x)
Left_Ankle_Roll   Kd: 8.53  → 17.06  (2x)
Right_Ankle_Pitch Kd: 8.53  → 17.06  (2x)
Right_Ankle_Roll  Kd: 8.53  → 17.06  (2x)
```

Higher damping absorbs oscillation energy faster, stabilizing the stance.

### 2. Increased Knee Stiffness (1.5x)
```
Left_Knee_Pitch   Kp: 251.09 → 376.64
Right_Knee_Pitch  Kp: 251.09 → 376.64
```

Stiffer knees better resist gravity and support the robot's weight.

### 3. Increased Ankle Pitch Stiffness (1.5x)
```
Left_Ankle_Pitch  Kp: 134.05 → 201.08
Right_Ankle_Pitch Kp: 134.05 → 201.08
```

Stiffer ankle pitch maintains foot position more firmly.

## Expected Improvements

After these changes:
- ✅ Vertical velocity should stabilize to ~0 m/s
- ✅ Robot should stand without shaking
- ✅ Legs should support the body weight properly
- ✅ When given forward velocity commands, feet should actually move
- ✅ Policy should have stable base to execute walking

## If Issues Persist

If the robot still shows instability after these changes:

1. **Increase hip damping further** (currently 13.17 for hip pitch)
   ```
   Left_Hip_Pitch Kd: 13.17 → 20.0
   Right_Hip_Pitch Kd: 13.17 → 20.0
   ```

2. **Increase hip stiffness** (currently 206.83)
   ```
   Left_Hip_Pitch Kp: 206.83 → 300+
   Right_Hip_Pitch Kp: 206.83 → 300+
   ```

3. **Verify default stance** - check that knees are bent enough (currently -0.2 rad)
   - May need to increase to -0.3 or -0.4 rad

4. **Check ground friction** - high friction prevents sliding but increases oscillation
   - Current: [1.0, 0.005, 0.0001]
   - Try: [0.8, 0.003, 0.0001]

5. **Lower physics timestep** - smaller steps = more stability
   - Current: 0.002s (500Hz)
   - Try: 0.001s (1000Hz) but increases compute

## File Modified

`src/colosseum/robots/t1_23dof/deploy_config.py`
- Lines 103-108: Updated joint_stiffness (knee and ankle pitch)
- Lines 115-120: Updated joint_damping (ankle pitch and roll)

## Testing

To verify the fix works:
```bash
cd /home/neverorfrog/code/spqr/colosseum
python tests/debug_deployment.py
```

Look for:
- Control loop stability: all joints should be stable now (Max |Vel| < 0.5)
- Sensor outputs: Z velocity should be near 0, not -5.7
