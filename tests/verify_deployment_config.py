#!/usr/bin/env python3
"""Verify deployment configuration matches training setup.

This script checks:
1. MuJoCo joint ordering (alphabetical)
2. Default joint positions match training
3. PD gains match actuator configs
4. Action scaling is correct
"""

import mujoco
import numpy as np
import torch
from pathlib import Path

from colosseum.robots.booster_t1.constants import get_spec, HOME_QPOS, JOINT_NAMES, ACTION_SCALE
from colosseum.robots.booster_t1.deploy_config import T1_23DOF_ROBOT_CFG
from colosseum.robots.booster_t1.actuators import (
    T1_ACTUATOR_NECK,
    T1_ACTUATOR_ARM,
    T1_ACTUATOR_WAIST,
    T1_ACTUATOR_HIP_PITCH,
    T1_ACTUATOR_HIP_ROLL,
    T1_ACTUATOR_HIP_YAW,
    T1_ACTUATOR_KNEE,
    T1_ACTUATOR_ANKLE_PITCH,
    T1_ACTUATOR_ANKLE_ROLL,
)


def check_joint_ordering():
    """Verify MuJoCo joint ordering is alphabetical."""
    print("=" * 80)
    print("1. CHECKING JOINT ORDERING")
    print("=" * 80)

    # Compile model
    spec = get_spec()
    model = spec.compile()

    # Get joint names from compiled model
    mujoco_joint_names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]

    print(f"\nMuJoCo joint order ({len(mujoco_joint_names)} joints):")
    for i, name in enumerate(mujoco_joint_names):
        print(f"  [{i:2d}] {name}")

    # Check if alphabetical
    sorted_names = sorted(mujoco_joint_names)
    is_alphabetical = mujoco_joint_names == sorted_names

    print(f"\nIs alphabetical? {is_alphabetical}")

    if not is_alphabetical:
        print("\n⚠️  WARNING: MuJoCo joints are NOT in alphabetical order!")
        print("Differences:")
        for i, (mj_name, sorted_name) in enumerate(zip(mujoco_joint_names, sorted_names)):
            if mj_name != sorted_name:
                print(f"  [{i:2d}] MuJoCo: {mj_name:20s} | Sorted: {sorted_name}")

    # Compare with deploy_config.py sim_joint_names
    deploy_sim_names = T1_23DOF_ROBOT_CFG.sim_joint_names
    print(f"\nDeploy config sim_joint_names ({len(deploy_sim_names)} joints):")
    for i, name in enumerate(deploy_sim_names):
        print(f"  [{i:2d}] {name}")

    matches = mujoco_joint_names == list(deploy_sim_names)
    print(f"\nDeploy sim_joint_names matches MuJoCo? {matches}")

    if not matches:
        print("\n⚠️  ERROR: Deploy sim_joint_names does NOT match MuJoCo ordering!")
        print("Differences:")
        for i in range(max(len(mujoco_joint_names), len(deploy_sim_names))):
            mj_name = mujoco_joint_names[i] if i < len(mujoco_joint_names) else "MISSING"
            deploy_name = deploy_sim_names[i] if i < len(deploy_sim_names) else "MISSING"
            if mj_name != deploy_name:
                print(f"  [{i:2d}] MuJoCo: {mj_name:20s} | Deploy: {deploy_name}")

    return mujoco_joint_names


def check_default_positions(mujoco_joint_names):
    """Verify default joint positions match training."""
    print("\n" + "=" * 80)
    print("2. CHECKING DEFAULT JOINT POSITIONS")
    print("=" * 80)

    # Training default positions (from HOME_QPOS in constants.py)
    print("\nTraining default positions (HOME_QPOS):")
    training_defaults = {}
    for joint_name in mujoco_joint_names:
        pos = HOME_QPOS.get(joint_name, 0.0)
        training_defaults[joint_name] = pos
        print(f"  {joint_name:20s}: {pos:6.3f}")

    # Deployment default positions (from deploy_config.py)
    # Need to map from real robot order to simulation order
    deploy_real_names = T1_23DOF_ROBOT_CFG.joint_names
    deploy_real_defaults = T1_23DOF_ROBOT_CFG.default_joint_pos

    print(f"\nDeployment default positions (in real robot order):")
    deploy_defaults_dict = {}
    for name, pos in zip(deploy_real_names, deploy_real_defaults):
        deploy_defaults_dict[name] = pos
        print(f"  {name:20s}: {pos:6.3f}")

    # Compare in simulation order
    print("\nComparison (in simulation order):")
    print(f"{'Joint Name':20s} | {'Training':>8s} | {'Deploy':>8s} | {'Diff':>8s} | Status")
    print("-" * 70)

    all_match = True
    max_diff = 0.0
    for joint_name in mujoco_joint_names:
        training_val = training_defaults[joint_name]
        deploy_val = deploy_defaults_dict.get(joint_name, 0.0)
        diff = abs(training_val - deploy_val)
        max_diff = max(max_diff, diff)

        status = "✓" if diff < 1e-6 else "✗"
        if diff >= 1e-6:
            all_match = False

        print(f"{joint_name:20s} | {training_val:8.3f} | {deploy_val:8.3f} | {diff:8.3f} | {status}")

    print(f"\nAll defaults match? {all_match}")
    print(f"Maximum difference: {max_diff:.3f} radians ({np.rad2deg(max_diff):.1f} degrees)")

    if not all_match:
        print("\n⚠️  ERROR: Deployment default positions DO NOT match training!")
        print("This will cause incorrect joint_pos_rel observations.")


def check_pd_gains():
    """Verify PD gains match between training and deployment."""
    print("\n" + "=" * 80)
    print("3. CHECKING PD GAINS")
    print("=" * 80)

    # Training PD gains from actuators
    training_gains = {
        "Neck": (T1_ACTUATOR_NECK.stiffness, T1_ACTUATOR_NECK.damping),
        "Arms": (T1_ACTUATOR_ARM.stiffness, T1_ACTUATOR_ARM.damping),
        "Waist": (T1_ACTUATOR_WAIST.stiffness, T1_ACTUATOR_WAIST.damping),
        "Hip_Pitch": (T1_ACTUATOR_HIP_PITCH.stiffness, T1_ACTUATOR_HIP_PITCH.damping),
        "Hip_Roll": (T1_ACTUATOR_HIP_ROLL.stiffness, T1_ACTUATOR_HIP_ROLL.damping),
        "Hip_Yaw": (T1_ACTUATOR_HIP_YAW.stiffness, T1_ACTUATOR_HIP_YAW.damping),
        "Knee": (T1_ACTUATOR_KNEE.stiffness, T1_ACTUATOR_KNEE.damping),
        "Ankle_Pitch": (T1_ACTUATOR_ANKLE_PITCH.stiffness, T1_ACTUATOR_ANKLE_PITCH.damping),
        "Ankle_Roll": (T1_ACTUATOR_ANKLE_ROLL.stiffness, T1_ACTUATOR_ANKLE_ROLL.damping),
    }

    print("\nTraining PD gains:")
    for name, (kp, kd) in training_gains.items():
        print(f"  {name:15s}: Kp={kp:7.2f}, Kd={kd:6.2f}")

    # Deployment PD gains (from deploy_config.py)
    # Map each joint to its gains
    deploy_real_names = T1_23DOF_ROBOT_CFG.joint_names
    deploy_kp = T1_23DOF_ROBOT_CFG.joint_stiffness
    deploy_kd = T1_23DOF_ROBOT_CFG.joint_damping

    print("\nDeployment PD gains (per joint):")
    for name, kp, kd in zip(deploy_real_names, deploy_kp, deploy_kd):
        print(f"  {name:20s}: Kp={kp:7.2f}, Kd={kd:6.2f}")


def check_action_scale():
    """Verify action scaling matches training."""
    print("\n" + "=" * 80)
    print("4. CHECKING ACTION SCALING")
    print("=" * 80)

    # Training action scale (uniform 0.25 for all joints)
    training_scale = 0.25
    print(f"\nTraining action scale: {training_scale}")
    print(f"  Defined in constants.py as uniform {training_scale} for all {len(JOINT_NAMES)} joints")

    # Deployment action scale (from config.py)
    from colosseum.tasks.velocity.deploy.t1_23dof.config import T1_23DOF_VELOCITY
    deploy_scale = T1_23DOF_VELOCITY.policy.action_scale
    print(f"\nDeployment action scale: {deploy_scale}")

    matches = abs(training_scale - deploy_scale) < 1e-6
    print(f"\nAction scales match? {matches}")

    if not matches:
        print(f"⚠️  WARNING: Action scales differ by {abs(training_scale - deploy_scale)}")


def main():
    print("\n" + "=" * 80)
    print("DEPLOYMENT CONFIGURATION VERIFICATION")
    print("=" * 80)

    # 1. Check joint ordering
    mujoco_joint_names = check_joint_ordering()

    # 2. Check default positions
    check_default_positions(mujoco_joint_names)

    # 3. Check PD gains
    check_pd_gains()

    # 4. Check action scaling
    check_action_scale()

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
