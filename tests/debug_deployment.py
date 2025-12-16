#!/usr/bin/env python3
"""Diagnostic tool for deployment stability issues.

Tests:
1. PD gains computation and actual values
2. Joint feedback and control loop behavior
3. Physics stepping and sensor integration
4. Policy inference and action application
"""

import numpy as np
import torch
import mujoco
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from colosseum.robots.t1_23dof.deploy_config import T1_23DOF_ROBOT_CFG
from colosseum.robots.t1_23dof.constants import get_spec, HOME_QPOS, JOINT_NAMES
from colosseum.deploy.backends.mujoco import MujocoController
from colosseum.tasks.velocity.deploy.t1_23dof.config import T1_23DOF_VELOCITY_ROUGH
from mjlab.utils.spec import create_position_actuator


def test_pd_gains():
    """Test PD gain values."""
    print("=" * 80)
    print("1. CHECKING PD GAINS")
    print("=" * 80)

    robot_cfg = T1_23DOF_ROBOT_CFG
    joint_names = robot_cfg.sim_joint_names
    stiffness = robot_cfg.joint_stiffness
    damping = robot_cfg.joint_damping

    print(f"\n{len(joint_names)} joints, stiffness and damping values:\n")
    print(f"{'Joint':<25} {'Kp':<10} {'Kd':<10}")
    print("-" * 45)

    for name, kp, kd in zip(joint_names, stiffness, damping):
        print(f"{name:<25} {kp:<10.2f} {kd:<10.2f}")

    # Check for issues
    issues = []
    if any(kp == 0 for kp in stiffness):
        issues.append("WARNING: Some Kp values are 0!")
    if any(kd == 0 for kd in damping):
        issues.append("WARNING: Some Kd values are 0!")
    if any(kp > 500 for kp in stiffness):
        issues.append("WARNING: Some Kp values > 500 (may oscillate)!")
    if any(kd > 50 for kd in damping):
        issues.append("WARNING: Some Kd values > 50!")

    if issues:
        print("\n❌ ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ PD gains appear reasonable")

    return robot_cfg


def test_mujoco_integration():
    """Test MuJoCo model compilation and sensor setup."""
    print("\n" + "=" * 80)
    print("2. CHECKING MUJOCO INTEGRATION")
    print("=" * 80)

    cfg = T1_23DOF_VELOCITY_ROUGH
    robot_cfg = cfg.robot

    # Compile spec
    spec = mujoco.MjSpec.from_file(robot_cfg.mjcf_path)
    spec.actuators.clear()

    # Add position actuators
    for i, joint_name in enumerate(robot_cfg.sim_joint_names):
        create_position_actuator(
            spec,
            joint_name,
            stiffness=robot_cfg.joint_stiffness[i],
            damping=robot_cfg.joint_damping[i],
            effort_limit=robot_cfg.effort_limit[i],
        )

    model = spec.compile()

    print(f"\nModel timestep: {model.opt.timestep:.4f}s")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of sensors: {model.nsensor}")

    # Check for required sensors
    required_sensors = ["orientation", "imu_lin_vel", "imu_ang_vel"]
    print(f"\nRequired sensors:")
    for sensor_name in required_sensors:
        try:
            sensor_id = model.sensor(sensor_name).id
            print(f"  ✓ {sensor_name} (id={sensor_id})")
        except KeyError:
            print(f"  ❌ {sensor_name} NOT FOUND!")

    # Test data initialization
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    qpos = np.concatenate([
        np.array(cfg.mujoco.init_pos, dtype=np.float32),
        np.array(cfg.mujoco.init_quat, dtype=np.float32),
        np.array(robot_cfg.default_joint_pos, dtype=np.float32),
    ])

    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    print(f"\n✓ MuJoCo model compiled successfully")
    print(f"  Initial base position: {data.qpos[:3]}")
    print(f"  Initial base quaternion: {data.qpos[3:7]}")
    print(f"  Default joint positions:\n    {np.array(robot_cfg.default_joint_pos, dtype=np.float32)[:5]}...")

    return model, data


def test_control_loop_stability(model: mujoco.MjModel, data: mujoco.MjData):
    """Test control loop stability with fixed targets."""
    print("\n" + "=" * 80)
    print("3. TESTING CONTROL LOOP STABILITY")
    print("=" * 80)

    # Use default positions as targets
    robot_cfg = T1_23DOF_ROBOT_CFG
    dof_targets = np.array(robot_cfg.default_joint_pos, dtype=np.float32)

    print(f"\nRunning 200 steps with default joint targets...")
    print(f"Target positions: {dof_targets[:3]}... (showing first 3 joints)")

    # Logs
    positions = []
    velocities = []
    controls = []

    for step in range(200):
        # Apply control (PD control)
        dof_pos = data.qpos[7:].astype(np.float32)
        dof_vel = data.qvel[6:].astype(np.float32)
        kp = robot_cfg.joint_stiffness
        kd = robot_cfg.joint_damping

        # PD control: τ = Kp * (q_target - q) - Kd * dq
        ctrl = np.clip(
            kp * (dof_targets - dof_pos) - kd * dof_vel,
            model.actuator_forcerange[:, 0],
            model.actuator_forcerange[:, 1],
        )

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if step % 20 == 0:
            positions.append(dof_pos.copy())
            velocities.append(dof_vel.copy())
            controls.append(ctrl.copy())

    positions = np.array(positions)
    velocities = np.array(velocities)
    controls = np.array(controls)

    print(f"\n✓ Completed 200 control steps")

    # Analyze stability
    print("\nStability analysis (first 3 joints):")
    print(f"{'Joint':<10} {'Final Pos Error':<20} {'Max |Vel|':<15}")
    print("-" * 45)

    stable_joints = 0
    for i in range(3):
        pos_error = np.abs(positions[-1, i] - dof_targets[i])
        max_vel = np.max(np.abs(velocities[:, i]))
        is_stable = pos_error < 0.1 and max_vel < 1.0

        symbol = "✓" if is_stable else "❌"
        print(f"{symbol} j{i:<8} {pos_error:<20.4f} {max_vel:<15.4f}")
        if is_stable:
            stable_joints += 1

    if stable_joints == 3:
        print("\n✓ Control loop appears stable!")
    else:
        print(f"\n⚠️  WARNING: Only {stable_joints}/3 joints are stable")
        print("   This indicates high/low gains or sensor issues")

    return positions, velocities, controls


def test_sensor_outputs(model: mujoco.MjModel, data: mujoco.MjData):
    """Test sensor outputs during simulation."""
    print("\n" + "=" * 80)
    print("4. TESTING SENSOR OUTPUTS")
    print("=" * 80)

    robot_cfg = T1_23DOF_ROBOT_CFG
    dof_targets = np.array(robot_cfg.default_joint_pos, dtype=np.float32)

    print(f"\nRunning 100 steps and logging sensor data...")

    sensor_logs = {
        "orientation": [],
        "imu_lin_vel": [],
        "imu_ang_vel": [],
    }

    for step in range(100):
        # Control
        dof_pos = data.qpos[7:].astype(np.float32)
        dof_vel = data.qvel[6:].astype(np.float32)
        kp = robot_cfg.joint_stiffness
        kd = robot_cfg.joint_damping

        ctrl = np.clip(
            kp * (dof_targets - dof_pos) - kd * dof_vel,
            model.actuator_forcerange[:, 0],
            model.actuator_forcerange[:, 1],
        )

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if step % 10 == 0:
            try:
                orientation = data.sensor("orientation").data.copy()
                lin_vel = data.sensor("imu_lin_vel").data.copy()
                ang_vel = data.sensor("imu_ang_vel").data.copy()

                sensor_logs["orientation"].append(orientation)
                sensor_logs["imu_lin_vel"].append(lin_vel)
                sensor_logs["imu_ang_vel"].append(ang_vel)
            except Exception as e:
                print(f"ERROR reading sensors: {e}")
                break

    if sensor_logs["orientation"]:
        sensor_logs["orientation"] = np.array(sensor_logs["orientation"])
        sensor_logs["imu_lin_vel"] = np.array(sensor_logs["imu_lin_vel"])
        sensor_logs["imu_ang_vel"] = np.array(sensor_logs["imu_ang_vel"])

        print(f"\n✓ Sensor data collected")
        print(f"  Orientation (should be ~[1, 0, 0, 0]): {sensor_logs['orientation'][-1]}")
        print(f"  Lin vel (should be ~[0, 0, 0]): {sensor_logs['imu_lin_vel'][-1]}")
        print(f"  Ang vel (should be ~[0, 0, 0]): {sensor_logs['imu_ang_vel'][-1]}")

        # Check for stability
        orientation_std = np.std(sensor_logs['orientation'], axis=0)
        print(f"\n  Orientation std dev: {orientation_std}")
        if np.any(orientation_std > 0.1):
            print(f"  ⚠️  WARNING: High orientation noise (std > 0.1)")

        lin_vel_std = np.std(sensor_logs['imu_lin_vel'], axis=0)
        print(f"  Lin vel std dev: {lin_vel_std}")
        if np.any(lin_vel_std > 0.5):
            print(f"  ⚠️  WARNING: High velocity noise")

    return sensor_logs


if __name__ == "__main__":
    print("\n" + "🔍 DEPLOYMENT DIAGNOSTIC TOOL 🔍".center(80))
    print("\nTesting T1 23-DOF velocity tracking deployment\n")

    # Test 1: PD gains
    robot_cfg = test_pd_gains()

    # Test 2: MuJoCo integration
    model, data = test_mujoco_integration()

    # Test 3: Control loop stability
    positions, velocities, controls = test_control_loop_stability(model, data)

    # Test 4: Sensor outputs
    sensor_logs = test_sensor_outputs(model, data)

    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print("""
Possible causes of leg shaking:
1. PD gains too high → check gains output above (Kp > 500 is likely)
2. Control loop frequency mismatch → verify decimation matches physics
3. Sensor noise → check sensor outputs above
4. Joint feedback control not matching training → verify joint order
5. Ground contact forces → may need ground friction tuning

Next steps:
1. Check if gains shown above are similar to training config
2. Verify joint order matches MuJoCo compilation order
3. Check if control_loop_stability test shows oscillations
4. Examine sensor_outputs for noise or drift

Falling forward on velocity command likely means:
- Policy produces invalid joint targets → verify obs spec matches
- Joint targets don't translate to forward motion → check policy
- Robot COM too far forward → check default stance
    """)
