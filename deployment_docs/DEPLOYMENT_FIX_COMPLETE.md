# T1 Deployment Stability - Complete Fix Summary

## Problem Statement

When starting the MuJoCo simulation:
1. **Legs shaking** - oscillations at 2.8 rad/s even at rest
2. **Falls forward on velocity command** - robot collapses instead of walking
3. **Falling motion** - Z velocity at -5.7 m/s (continuously falling)

## Root Cause

**Multi-layered instability:**

1. **Primary Issue**: Robot COM not properly balanced → falling motion (-5.7 m/s)
2. **Secondary Issue**: Ankle damping too low → leg oscillations (2.8 rad/s)
3. **Initialization Issue**: Height mismatch between training (0.665m) and deployment (0.66m)

## Fixes Applied

### Fix 1: PD Gains Tuning

**File**: `src/colosseum/robots/t1_23dof/deploy_config.py`

#### Ankle Damping (2x increase)
```python
# Lines 117-118
# BEFORE: Left_Ankle_Pitch Kd: 8.53
# AFTER:  Left_Ankle_Pitch Kd: 17.06 (doubled)

# BEFORE: Left_Ankle_Roll Kd: 8.53
# AFTER:  Left_Ankle_Roll Kd: 17.06 (doubled)
```
**Rationale**: Higher damping absorbs oscillation energy faster

#### Knee Pitch Stiffness (1.5x increase)
```python
# Lines 103-104
# BEFORE: Left_Knee_Pitch Kp: 251.09
# AFTER:  Left_Knee_Pitch Kp: 376.64 (1.5x)
```
**Rationale**: Stiffer knees better support robot weight against gravity

#### Ankle Pitch Stiffness (1.5x increase)
```python
# Lines 103-104
# BEFORE: Left_Ankle_Pitch Kp: 134.05
# AFTER:  Left_Ankle_Pitch Kp: 201.08 (1.5x)
```
**Rationale**: Stiffer ankle pitch maintains foot position

### Fix 2: Initialization Height Synchronization

**File**: `src/colosseum/tasks/velocity/deploy/t1_23dof/config.py`

```python
# Lines 55 & 108
# BEFORE: init_pos=(0.0, 0.0, 0.66)
# AFTER:  init_pos=(0.0, 0.0, 0.665)
```

**Rationale**: 
- Training default height: 0.665m
- Old deployment height: 0.66m  
- Mismatch caused improper foot-ground contact
- Feet not properly in contact → falling motion

## How to Verify Fixes Work

### Test 1: Run Diagnostic
```bash
cd /home/neverorfrog/code/spqr/colosseum
python tests/debug_deployment.py
```

**Expected results:**
- ✅ Control loop stability: Head/arm max velocity < 1.0 rad/s (was 2.8+)
- ✅ Z velocity noise std < 0.3 (was 0.567)
- ✅ Vertical velocity near 0 (was -5.7)

### Test 2: Check Ground Contact
```bash
python tests/debug_ground_contact.py
```

**Expected results:**
- ✅ Feet in contact with ground (z < 0.01m)
- ✅ Contact forces increasing over time
- ✅ Z velocity stabilizing to near 0
- ✅ COM stabilizing in vertical

### Test 3: Manual Simulation Test
```bash
python src/colosseum/deploy/core/launcher.py --task t1-velocity-rough
```

**Expected behavior:**
1. Robot initializes and stands still without shaking
2. Robot maintains upright position for 5+ seconds
3. Small velocity command (e.g., `0.1 0 0`) causes feet to move forward
4. Robot doesn't collapse when walking

## Files Modified

```
src/colosseum/robots/t1_23dof/deploy_config.py
├─ Line 103-108: Updated joint_stiffness (knee, ankle pitch)
└─ Line 117-118: Updated joint_damping (ankle pitch, ankle roll)

src/colosseum/tasks/velocity/deploy/t1_23dof/config.py
├─ Line 55: Updated init_pos (ROUGH config)
└─ Line 108: Updated init_pos (FLAT config)
```

## Remaining Diagnosis

If issues still persist after these fixes:

### Scenario 1: Still Falling
Check:
- Is ground geometry loaded correctly? (visual inspector in MuJoCo)
- Do feet have proper collision shapes?
- Is foot height computed correctly?

### Scenario 2: Still Oscillating
Check:
- Is the observation spec matching training?
- Are joint orders consistent between training and deployment?
- Is the policy model loading correctly?

### Scenario 3: Falls on Velocity Command
Check:
- Is policy inference working?
- Are observation computations correct?
- Is action scaling correct (0.25)?

## Next Steps

1. **Run diagnostics** to verify PD gains and control loop
2. **Check ground contact** to ensure feet touch ground properly
3. **Run manual simulation** and observe behavior
4. **Adjust further if needed** using contingency fixes (see DEPLOYMENT_STABILITY_FIX.md)

---

**Created**: Dec 14, 2025
**Status**: Applied fixes for:
- ✅ Ankle damping (2x)
- ✅ Knee/ankle pitch stiffness (1.5x)  
- ✅ Initialization height (training match)
