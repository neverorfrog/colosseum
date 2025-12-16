#!/usr/bin/env python3
"""Quick validation of PD gain fixes for deployment stability."""

import numpy as np

print("=" * 80)
print("DEPLOYMENT PD GAIN FIXES - BEFORE vs AFTER")
print("=" * 80)

# Original gains
kp_orig = [
    15.99, 15.99,      # Head
    160.61, 160.61, 160.61, 160.61,  # Left arm
    160.61, 160.61, 160.61, 160.61,  # Right arm
    188.76,            # Waist
    206.83, 188.76, 188.76, 251.09, 134.05, 134.05,  # Left leg
    206.83, 188.76, 188.76, 251.09, 134.05, 134.05,  # Right leg
]

kd_orig = [
    0.68, 0.68,        # Head
    8.52, 8.52, 8.52, 8.52,  # Left arm
    8.52, 8.52, 8.52, 8.52,  # Right arm
    12.02,             # Waist
    13.17, 12.02, 12.02, 15.98, 8.53, 8.53,  # Left leg
    13.17, 12.02, 12.02, 15.98, 8.53, 8.53,  # Right leg
]

# New gains (fixed)
kp_new = [
    15.99, 15.99,      # Head
    160.61, 160.61, 160.61, 160.61,  # Left arm
    160.61, 160.61, 160.61, 160.61,  # Right arm
    188.76,            # Waist
    206.83, 188.76, 188.76, 376.64, 201.08, 134.05,  # Left leg
    206.83, 188.76, 188.76, 376.64, 201.08, 134.05,  # Right leg
]

kd_new = [
    0.68, 0.68,        # Head
    8.52, 8.52, 8.52, 8.52,  # Left arm
    8.52, 8.52, 8.52, 8.52,  # Right arm
    12.02,             # Waist
    13.17, 12.02, 12.02, 15.98, 17.06, 17.06,  # Left leg
    13.17, 12.02, 12.02, 15.98, 17.06, 17.06,  # Right leg
]

joint_names = [
    "AAHead_yaw",           # 0
    "Head_pitch",           # 1
    "L_Shoulder_Pitch",     # 2
    "L_Shoulder_Roll",      # 3
    "L_Elbow_Pitch",        # 4
    "L_Elbow_Yaw",          # 5
    "R_Shoulder_Pitch",     # 6
    "R_Shoulder_Roll",      # 7
    "R_Elbow_Pitch",        # 8
    "R_Elbow_Yaw",          # 9
    "Waist",                # 10
    "L_Hip_Pitch",          # 11
    "L_Hip_Roll",           # 12
    "L_Hip_Yaw",            # 13
    "L_Knee_Pitch",         # 14 ⬅️ MODIFIED
    "L_Ankle_Pitch",        # 15 ⬅️ MODIFIED
    "L_Ankle_Roll",         # 16
    "R_Hip_Pitch",          # 17
    "R_Hip_Roll",           # 18
    "R_Hip_Yaw",            # 19
    "R_Knee_Pitch",         # 20 ⬅️ MODIFIED
    "R_Ankle_Pitch",        # 21 ⬅️ MODIFIED
    "R_Ankle_Roll",         # 22
]

print("\n🔧 CHANGES MADE:\n")
print(f"{'Joint':<20} {'Kp (Orig)':<12} {'Kp (New)':<12} {'Change':<10} | "
      f"{'Kd (Orig)':<10} {'Kd (New)':<10} {'Change':<10}")
print("-" * 100)

for i, name in enumerate(joint_names):
    kp_change = kp_new[i] - kp_orig[i]
    kd_change = kd_new[i] - kd_orig[i]
    
    marker = " ⬅️ " if (kp_change != 0 or kd_change != 0) else ""
    
    if kp_change != 0 or kd_change != 0:
        print(f"{name:<20} {kp_orig[i]:<12.2f} {kp_new[i]:<12.2f} "
              f"{kp_change:+.2f}({kp_change/kp_orig[i]*100:+.0f}%) | "
              f"{kd_orig[i]:<10.2f} {kd_new[i]:<10.2f} "
              f"{kd_change:+.2f}({kd_change/kd_orig[i]*100:+.0f}%){marker}")

print("\n" + "=" * 100)
print("\n✅ RATIONALE FOR FIXES:\n")
print("""
DIAGNOSIS RESULTS:
- Ankle vertical velocity oscillating at -5.7 m/s (should be ~0)
- Head/arm joints oscillating (max velocity 2.8 rad/s at rest)
- Robot falling when given velocity commands

ROOT CAUSE:
The legs cannot maintain a stable stance. The ankle roll and pitch joints
had insufficient damping (Kd=8.53) to dissipate vibrations and maintain
vertical equilibrium. The knee joints also needed more stiffness to better
support the robot's weight.

FIXES APPLIED:
1. Ankle pitch damping: 8.53 → 17.06 (2x increase)
   - Reduces oscillation frequency and damps vertical motion faster
   
2. Ankle roll damping: 8.53 → 17.06 (2x increase)  
   - Stabilizes side-to-side movement
   
3. Knee pitch stiffness: 251.09 → 376.64 (1.5x increase)
   - Better resists gravity, more stable standing
   
4. Ankle pitch stiffness: 134.05 → 201.08 (1.5x increase)
   - Stiffer ankle to maintain foot position

EXPECTED RESULTS:
✓ Vertical velocity oscillations should be damped (close to 0)
✓ Robot should stand without shaking
✓ When given velocity command, robot should move feet (not just fall)
✓ Better stability foundation for walking policy
""")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("""
1. Run the simulation again with the updated gains
2. Check that:
   - Robot stands still without shaking
   - Vertical velocity stays near 0 m/s
   - When given forward command, feet actually move
3. If still unstable, may need to:
   - Increase leg hip stiffness further
   - Check default stance (knee bending angle)
   - Adjust ground friction parameters
""")
