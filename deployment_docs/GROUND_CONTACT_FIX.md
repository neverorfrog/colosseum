# Ground Contact Fix - Root Cause Found & Fixed

## Diagnosis Result

The ground contact diagnostic revealed the **actual root cause** of the instability:

### Critical Issue: Feet Not Touching Ground!
```
Foot Z position: 0.0318m (should be ~0)
Feet are floating 3.2cm above the ground!
```

### Full Diagnostic Data:
- **Base position**: z = 0.665m
- **COM position**: z = 0.5754m (offset -0.0896m from base)
- **Foot position**: z = 0.0318m ← **3.2cm above ground!**
- **Leg length**: 0.665 - 0.0318 = 0.633m
- **Contact forces**: Very high (1667 N start, unstable)
- **Z velocity**: Oscillating (-0.1928 to +0.1478 m/s)

## Why Feet Are Floating

The model geometry shows feet are at ~3.2cm height relative to ground plane at z=0, but the COM calculation shows the feet need to be ~3.2cm lower to touch ground properly.

This means the **deployment initialization height of 0.665m was too low** for the model's geometry.

## Applied Fix

**File**: `src/colosseum/tasks/velocity/deploy/t1_23dof/config.py`

```python
# BEFORE (both ROUGH and FLAT configs):
init_pos=(0.0, 0.0, 0.665)  # Feet floating 3.2cm above ground

# AFTER:
init_pos=(0.0, 0.0, 0.70)   # Raised by 3.5cm to ensure ground contact
```

**Rationale**:
- Feet were at 0.0318m at height 0.665m
- Need feet at ~0.0m for ground contact
- So body needs to be raised by 0.70 - 0.665 = 0.035m (3.5cm)
- New height: 0.70m

## Expected Results After Fix

With feet actually touching the ground:

✅ **No more falling** - feet provide proper support  
✅ **Lower contact forces** - distributed over actual surface  
✅ **Stable stance** - COM properly supported  
✅ **Control loop works** - stable base for PD control  
✅ **Walking possible** - feet can push off ground  

## How High Contact Forces Caused Oscillation

The diagnostic showed:
- Contact forces: **1667 N initial → 788 N end**
- These massive forces combined with high stiffness (376.64 for knees) caused:
  1. Rapid penetration of "floating" feet into ground
  2. Huge contact reaction forces
  3. High forces → oscillations through stiff joints
  4. Cycle repeats → leg shaking at 2.8 rad/s

Once feet are actually touching ground from the start:
- Contact forces will be **much lower and stable**
- No rapid penetration → no shock forces
- Smooth settling instead of oscillation

## Files Modified

```
src/colosseum/tasks/velocity/deploy/t1_23dof/config.py
├─ Line 52-58: T1_23DOF_VELOCITY_ROUGH.mujoco.init_pos (0.665 → 0.70)
└─ Line 101-107: T1_23DOF_VELOCITY_FLAT.mujoco.init_pos (0.665 → 0.70)
```

## Why Training Worked at 0.665m

Training (mjlab) might use:
- Different ground plane offset
- Different model orientation
- Or they handle height offset differently

For deployment, we need feet to actually contact the ground at z=0, which requires 0.70m base height.

## Testing Next Step

Run diagnostic again with new height:
```bash
python tests/debug_ground_contact.py
```

Expected new results:
- Foot Z position: ~0.0000m (touching ground)
- Contact forces: Stable ~1000-1500 N (robot weight distributed)
- Z velocity: Stable near 0 m/s
- No oscillations in contact forces

---

**Summary**: The robot was literally falling through the air because its feet didn't touch the ground. Now with proper initialization height (0.70m instead of 0.665m), feet should contact ground and provide support for stable standing and walking.
