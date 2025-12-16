# Deployment Stability Fixes - Second Round

## Issue Status After First Fix

After applying ankle damping and knee stiffness fixes, diagnostics showed:
- ❌ Head/arm oscillations **got WORSE** (2.81→2.96, 1.30→1.71 rad/s)  
- ❌ Vertical velocity still **-5.7 m/s** (robot falling)
- ❌ Oscillations indicate **structural instability**, not just PD tuning

## Root Cause Analysis

The diagnostic revealed the robot is falling even before control kicks in:
1. **Initial falling velocity of -5.7 m/s** right from startup
2. **Angular velocity in Y axis**: -0.209 rad/s (tipping forward)
3. **This happens during initialization**, not from control loop

This indicates the **initial stance/COM is not balanced**, not a control gain issue.

## Applied Fixes (Round 2)

### 1. Verified Default Standing Pose
**Confirmed:**
- Knee pitch = 0.4 rad (23° bend) ✓ Correct
- Matches training HOME_QPOS exactly ✓

### 2. Updated Initialization Height
```python
# BEFORE: init_pos=(0.0, 0.0, 0.66)  # 660mm
# AFTER:  init_pos=(0.0, 0.0, 0.665) # 665mm (training default)
```

This ensures the robot starts at the same height as training, where the feet should contact the ground properly.

**Files updated:**
- `src/colosseum/tasks/velocity/deploy/t1_23dof/config.py`
  - `T1_23DOF_VELOCITY_ROUGH.mujoco.init_pos`: 0.66 → 0.665
  - `T1_23DOF_VELOCITY_FLAT.mujoco.init_pos`: 0.66 → 0.665

## Expected Improvements

With these fixes:
- ✅ Robot COM should be properly balanced over feet
- ✅ Initial falling motion should be eliminated
- ✅ Control loop can then stabilize the oscillations
- ✅ Velocity commands should result in movement, not collapse

## If Issues Still Persist

The remaining oscillations in head/arms might indicate:

1. **Ground contact not working properly**
   - Check if feet are in contact with ground in MuJoCo
   - May need to adjust ground friction further
   - Verify foot collision geometry is correct

2. **Numerical instability in the control loop**
   - The 2x increase in ankle damping actually made it worse
   - Might indicate the issue is elsewhere (ground contact, COM)
   - Consider reducing ankle damping back to original if ground contact fails

3. **Joint ordering or observation mismatch**
   - Verify that joint order in deployment matches MuJoCo
   - Check observation spec matches policy training

## Summary of All Changes

### deploy_config.py
```
Line 130-135: Clarified default_joint_pos comments (knees=0.4 for stable standing)
Line 103-108: Increased knee stiffness 1.5x (251→376.64)
Line 104: Increased ankle pitch stiffness 1.5x (134.05→201.08)
Line 117: Doubled ankle pitch damping (8.53→17.06)
Line 118: Doubled ankle roll damping (8.53→17.06)
```

### config.py  
```
Line 55: Updated init_pos from 0.66 to 0.665 (ROUGH config)
Line 108: Updated init_pos from 0.66 to 0.665 (FLAT config)
```

## Testing Checklist

Before running full deployment, verify:
- [ ] Robot initializes at 0.665m height
- [ ] No falling velocity (-5.7 m/s) at startup
- [ ] Feet in contact with ground (check in MuJoCo viewer)
- [ ] Head/arm oscillations reduced
- [ ] Robot stands still without shaking for 2+ seconds
- [ ] Small velocity command causes feet to move (not fall)
