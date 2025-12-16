"""Diagnostic tests for T1 deployment stability investigation.

Run this script to collect diagnostic data about:
1. Control loop timing
2. Action scaling comparison
3. Actuator configuration
4. Physics solver settings
5. State update timing
"""

import time
import numpy as np
import torch
import mujoco

from colosseum.deploy.core.registry import TASK_REGISTRY
from colosseum.deploy.backends.mujoco import MujocoController


def test_control_loop_timing(task_name: str = "t1-velocity-flat", num_steps: int = 100):
    """Measure control loop timing and identify bottlenecks."""
    print("\n" + "=" * 80)
    print("TEST 1: Control Loop Timing Analysis")
    print("=" * 80)

    cfg = TASK_REGISTRY.get_config(task_name)
    controller = MujocoController(cfg)

    # Target timing
    target_dt = cfg.physics_dt * cfg.mujoco.decimation
    print(f"\nTarget control loop period: {target_dt*1000:.2f}ms ({1/target_dt:.1f}Hz)")
    print(f"Physics timestep: {cfg.physics_dt*1000:.2f}ms ({1/cfg.physics_dt:.0f}Hz)")
    print(f"Decimation: {cfg.mujoco.decimation}x")

    # Initialize
    controller.update_state()
    controller.start()

    # Timing measurements
    loop_times = []
    update_state_times = []
    policy_step_times = []
    ctrl_step_times = []
    sleep_times = []

    print(f"\nRunning {num_steps} control cycles...")

    for i in range(num_steps):
        loop_start = time.perf_counter()

        # 1. Sleep (as currently implemented)
        sleep_start = time.perf_counter()
        time.sleep(target_dt)
        sleep_time = time.perf_counter() - sleep_start
        sleep_times.append(sleep_time)

        # 2. Update state
        update_start = time.perf_counter()
        controller.update_state()
        update_time = time.perf_counter() - update_start
        update_state_times.append(update_time)

        # 3. Policy inference
        policy_start = time.perf_counter()
        dof_targets = controller.policy_step()
        policy_time = time.perf_counter() - policy_start
        policy_step_times.append(policy_time)

        # 4. Control step (physics stepping)
        ctrl_start = time.perf_counter()
        controller.ctrl_step(dof_targets)
        ctrl_time = time.perf_counter() - ctrl_start
        ctrl_step_times.append(ctrl_time)

        # Total loop time
        loop_time = time.perf_counter() - loop_start
        loop_times.append(loop_time)

        if i % 20 == 0:
            print(f"Step {i:3d}: {loop_time*1000:6.2f}ms (sleep:{sleep_time*1000:5.2f}, "
                  f"update:{update_time*1000:4.2f}, policy:{policy_time*1000:4.2f}, "
                  f"ctrl:{ctrl_time*1000:4.2f})")

    # Statistics
    def print_stats(name, times, target=None):
        times_ms = np.array(times) * 1000
        print(f"\n{name}:")
        print(f"  Mean:   {times_ms.mean():.3f}ms")
        print(f"  Std:    {times_ms.std():.3f}ms")
        print(f"  Min:    {times_ms.min():.3f}ms")
        print(f"  Max:    {times_ms.max():.3f}ms")
        if target is not None:
            print(f"  Target: {target*1000:.3f}ms")
            print(f"  Error:  {(times_ms.mean() - target*1000):.3f}ms ({(times_ms.mean()/target/1000*100 - 100):.1f}%)")

    print("\n" + "-" * 80)
    print("TIMING STATISTICS")
    print("-" * 80)
    print_stats("Total Loop Time", loop_times, target_dt)
    print_stats("Sleep Time", sleep_times, target_dt)
    print_stats("Update State Time", update_state_times)
    print_stats("Policy Step Time", policy_step_times)
    print_stats("Control Step Time", ctrl_step_times)

    # Calculate overhead and latency
    execution_overhead = np.mean(update_state_times) + np.mean(policy_step_times) + np.mean(ctrl_step_times)
    total_latency = np.mean(sleep_times) + execution_overhead

    print("\n" + "-" * 80)
    print("LATENCY ANALYSIS")
    print("-" * 80)
    print(f"Execution overhead (update+policy+ctrl): {execution_overhead*1000:.3f}ms")
    print(f"Total observation latency (sleep+overhead): {total_latency*1000:.3f}ms")
    print(f"Latency as % of target period: {(total_latency/target_dt*100):.1f}%")
    print(f"\n⚠️  ISSUE: Policy sees observations that are {total_latency*1000:.1f}ms old!")
    print(f"⚠️  This is {(total_latency/target_dt):.2f}x the target control period!")

    return {
        "loop_times": loop_times,
        "update_times": update_state_times,
        "policy_times": policy_step_times,
        "ctrl_times": ctrl_step_times,
        "sleep_times": sleep_times,
        "total_latency_ms": total_latency * 1000,
    }


def test_action_scaling(task_name: str = "t1-velocity-flat"):
    """Verify action scaling matches training configuration."""
    print("\n" + "=" * 80)
    print("TEST 2: Action Scaling Verification")
    print("=" * 80)

    cfg = TASK_REGISTRY.get_config(task_name)
    controller = MujocoController(cfg)

    # Initialize policy
    controller.update_state()
    controller.start()

    policy = controller.policy
    robot = controller.robot

    print(f"\nPolicy action scale configuration:")
    print(f"  config.action_scale: {policy.config.action_scale}")
    print(f"\nComputed per-joint action scaling:")
    print(f"  Formula: action_scale * effort_limit / stiffness")

    # Print per-joint scaling
    action_scale = policy.action_scale.numpy()
    effort_limit = robot.effort_limit.numpy()
    stiffness = robot.joint_stiffness.numpy()

    print(f"\n{'Joint Name':<25} {'Effort':>8} {'Stiffness':>10} {'Scale':>10} {'Range':>10}")
    print("-" * 80)

    for i, joint_name in enumerate(robot.cfg.joint_names):
        joint_range = action_scale[i]
        print(f"{joint_name:<25} {effort_limit[i]:>8.2f} {stiffness[i]:>10.2f} "
              f"{action_scale[i]:>10.4f} {joint_range:>10.4f}")

    # Check if scaling is uniform or per-joint
    scale_min = action_scale.min()
    scale_max = action_scale.max()
    scale_std = action_scale.std()

    print(f"\nScaling statistics:")
    print(f"  Min:  {scale_min:.6f}")
    print(f"  Max:  {scale_max:.6f}")
    print(f"  Mean: {action_scale.mean():.6f}")
    print(f"  Std:  {scale_std:.6f}")

    if scale_std < 1e-6:
        print(f"\n✓ Action scaling is UNIFORM: {scale_min:.6f}")
    else:
        print(f"\n⚠️  Action scaling is PER-JOINT (varies by {(scale_max-scale_min)/scale_min*100:.1f}%)")
        print(f"⚠️  This might differ from training's uniform 0.25 scaling!")

    # Compare with expected training value
    expected_training_scale = 0.25
    print(f"\nExpected training action scale: {expected_training_scale}")
    print(f"Deployment mean action scale:   {action_scale.mean():.6f}")

    if abs(action_scale.mean() - expected_training_scale) > 0.001:
        print(f"⚠️  WARNING: Mean scale differs from training by {abs(action_scale.mean() - expected_training_scale):.6f}")
    else:
        print(f"✓ Mean scale matches training (within tolerance)")

    return {
        "action_scale": action_scale,
        "is_uniform": scale_std < 1e-6,
        "mean": action_scale.mean(),
    }


def test_actuator_configuration(task_name: str = "t1-velocity-flat"):
    """Verify actuator configuration matches training."""
    print("\n" + "=" * 80)
    print("TEST 3: Actuator Configuration Verification")
    print("=" * 80)

    cfg = TASK_REGISTRY.get_config(task_name)
    controller = MujocoController(cfg)

    model = controller.mj_model
    robot = controller.robot

    print(f"\nMuJoCo Model Configuration:")
    print(f"  Number of actuators: {model.nu}")
    print(f"  Number of DOFs: {model.nv}")
    print(f"  Number of joints: {model.njnt}")

    print(f"\n{'Actuator':<25} {'Joint':<25} {'Kp':>10} {'Kd':>10} {'Effort':>10}")
    print("-" * 95)

    for i in range(model.nu):
        actuator = model.actuator(i)
        joint_id = actuator.trnid[0]
        joint_name = model.joint(joint_id).name if joint_id >= 0 else "N/A"

        kp = actuator.gainprm[0]
        kd = -actuator.biasprm[2]  # Note: MuJoCo uses negative for damping
        effort_min, effort_max = actuator.forcerange

        print(f"{actuator.name:<25} {joint_name:<25} {kp:>10.2f} {kd:>10.2f} {effort_max:>10.2f}")

    # Compare with robot config
    print(f"\nExpected from RobotConfig:")
    print(f"{'Joint Name':<25} {'Kp (expected)':>15} {'Kd (expected)':>15} {'Effort (expected)':>18}")
    print("-" * 95)

    for i, joint_name in enumerate(robot.cfg.sim_joint_names):
        kp_expected = robot.cfg.joint_stiffness[i]
        kd_expected = robot.cfg.joint_damping[i]
        effort_expected = robot.cfg.effort_limit[i]
        print(f"{joint_name:<25} {kp_expected:>15.2f} {kd_expected:>15.2f} {effort_expected:>18.2f}")

    # Verify actuators match config
    print("\nVerification:")
    all_match = True
    for i in range(model.nu):
        actuator = model.actuator(i)
        kp = actuator.gainprm[0]
        kd = -actuator.biasprm[2]

        kp_expected = robot.cfg.joint_stiffness[i]
        kd_expected = robot.cfg.joint_damping[i]

        if abs(kp - kp_expected) > 0.01 or abs(kd - kd_expected) > 0.01:
            print(f"  ✗ Actuator {i} ({actuator.name}): Kp={kp:.2f} (expected {kp_expected:.2f}), "
                  f"Kd={kd:.2f} (expected {kd_expected:.2f})")
            all_match = False

    if all_match:
        print("  ✓ All actuators match RobotConfig")

    return {"actuators_match": all_match}


def test_solver_settings(task_name: str = "t1-velocity-flat"):
    """Check MuJoCo solver settings."""
    print("\n" + "=" * 80)
    print("TEST 4: Physics Solver Configuration")
    print("=" * 80)

    cfg = TASK_REGISTRY.get_config(task_name)
    controller = MujocoController(cfg)

    model = controller.mj_model
    opt = model.opt

    print(f"\nMuJoCo Solver Settings:")
    print(f"  Timestep:            {opt.timestep:.6f}s ({1/opt.timestep:.0f}Hz)")
    print(f"  Iterations:          {opt.iterations}")
    print(f"  LS Iterations:       {opt.ls_iterations}")
    print(f"  Tolerance:           {opt.tolerance}")
    print(f"  LS Tolerance:        {opt.ls_tolerance}")
    print(f"  Solver:              {opt.solver}")
    print(f"  Integrator:          {opt.integrator}")
    print(f"  Cone:                {opt.cone}")
    print(f"  Jacobian:            {opt.jacobian}")

    print(f"\nExpected Training Settings:")
    print(f"  Timestep:            0.005000s (200Hz)")
    print(f"  Iterations:          10")
    print(f"  LS Iterations:       20")

    warnings = []
    if opt.timestep != 0.005:
        warnings.append(f"Timestep mismatch: {opt.timestep} vs 0.005")
    if opt.iterations != 10:
        warnings.append(f"Iterations mismatch: {opt.iterations} vs 10")
    if opt.ls_iterations != 20:
        warnings.append(f"LS iterations mismatch: {opt.ls_iterations} vs 20")

    if warnings:
        print(f"\n⚠️  Warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print(f"\n✓ All solver settings match training")

    return {
        "timestep": opt.timestep,
        "iterations": opt.iterations,
        "ls_iterations": opt.ls_iterations,
        "matches_training": len(warnings) == 0,
    }


def test_state_freshness(task_name: str = "t1-velocity-flat", num_steps: int = 50):
    """Test observation freshness by tracking state change timing."""
    print("\n" + "=" * 80)
    print("TEST 5: State Freshness Analysis")
    print("=" * 80)

    cfg = TASK_REGISTRY.get_config(task_name)
    controller = MujocoController(cfg)

    controller.update_state()
    controller.start()

    print(f"\nTracking base position changes over {num_steps} steps...")

    positions = []
    observation_times = []
    physics_times = []

    for i in range(num_steps):
        # Record observation time BEFORE physics step
        obs_time = time.perf_counter()
        controller.update_state()
        observation_times.append(obs_time)

        # Record position at observation time
        pos = controller.robot.data.root_pos_w.clone()
        positions.append(pos)

        # Policy and control step
        dof_targets = controller.policy_step()

        # Physics step happens HERE
        physics_time = time.perf_counter()
        controller.ctrl_step(dof_targets)
        physics_times.append(physics_time)

        time.sleep(cfg.physics_dt * cfg.mujoco.decimation)

    # Calculate time deltas
    obs_to_physics = [(physics_times[i] - observation_times[i]) * 1000
                      for i in range(len(physics_times))]

    print(f"\nTime from observation to physics step:")
    print(f"  Mean:   {np.mean(obs_to_physics):.3f}ms")
    print(f"  Std:    {np.std(obs_to_physics):.3f}ms")
    print(f"  Min:    {np.min(obs_to_physics):.3f}ms")
    print(f"  Max:    {np.max(obs_to_physics):.3f}ms")

    # Calculate position changes
    positions = torch.stack(positions)
    pos_deltas = torch.diff(positions, dim=0).norm(dim=1).numpy()

    print(f"\nBase position change per step:")
    print(f"  Mean:   {pos_deltas.mean()*1000:.3f}mm")
    print(f"  Std:    {pos_deltas.std()*1000:.3f}mm")
    print(f"  Max:    {pos_deltas.max()*1000:.3f}mm")

    print(f"\n⚠️  Key insight: Observations are captured BEFORE physics step,")
    print(f"    but they reflect state from {np.mean(obs_to_physics):.1f}ms ago + sleep time!")

    return {
        "obs_to_physics_ms": obs_to_physics,
        "position_deltas": pos_deltas,
    }


def run_all_diagnostics(task_name: str = "t1-velocity-flat"):
    """Run all diagnostic tests."""
    print("\n" + "=" * 80)
    print("T1 DEPLOYMENT DIAGNOSTICS - FULL SUITE")
    print("=" * 80)
    print(f"\nTask: {task_name}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    try:
        results["timing"] = test_control_loop_timing(task_name, num_steps=100)
    except Exception as e:
        print(f"\n✗ Timing test failed: {e}")
        results["timing"] = None

    try:
        results["action_scaling"] = test_action_scaling(task_name)
    except Exception as e:
        print(f"\n✗ Action scaling test failed: {e}")
        results["action_scaling"] = None

    try:
        results["actuators"] = test_actuator_configuration(task_name)
    except Exception as e:
        print(f"\n✗ Actuator test failed: {e}")
        results["actuators"] = None

    try:
        results["solver"] = test_solver_settings(task_name)
    except Exception as e:
        print(f"\n✗ Solver test failed: {e}")
        results["solver"] = None

    try:
        results["state_freshness"] = test_state_freshness(task_name, num_steps=50)
    except Exception as e:
        print(f"\n✗ State freshness test failed: {e}")
        results["state_freshness"] = None

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    if results["timing"]:
        print(f"\n✓ Timing Analysis:")
        print(f"  - Total latency: {results['timing']['total_latency_ms']:.1f}ms")
        print(f"  - Primary issue: Fixed sleep before state update")

    if results["action_scaling"]:
        if results["action_scaling"]["is_uniform"]:
            print(f"\n✓ Action Scaling:")
            print(f"  - Uniform scaling: {results['action_scaling']['mean']:.6f}")
        else:
            print(f"\n⚠️  Action Scaling:")
            print(f"  - Per-joint scaling (may differ from training)")

    if results["actuators"]:
        if results["actuators"]["actuators_match"]:
            print(f"\n✓ Actuators: Configuration matches RobotConfig")
        else:
            print(f"\n⚠️  Actuators: Configuration mismatch detected")

    if results["solver"]:
        if results["solver"]["matches_training"]:
            print(f"\n✓ Solver: Settings match training")
        else:
            print(f"\n⚠️  Solver: Settings differ from training")

    print("\n" + "=" * 80)
    print("END DIAGNOSTICS")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    task_name = "t1-velocity-flat"
    if len(sys.argv) > 1:
        task_name = sys.argv[1]

    # Run diagnostics
    results = run_all_diagnostics(task_name)
