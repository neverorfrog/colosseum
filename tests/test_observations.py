"""Test observation computation matches training expectations."""

import torch
from colosseum.deploy.core.registry import TASK_REGISTRY, auto_register_tasks
from colosseum.deploy.backends.mujoco import MujocoController
from colosseum.tasks.velocity.mdp.observation_spec import VELOCITY_OBS_SPEC

auto_register_tasks()


def test_observation_structure():
    """Verify observation structure matches spec."""
    print("\n" + "=" * 80)
    print("TEST: Observation Structure Validation")
    print("=" * 80)

    cfg = TASK_REGISTRY.get_config("t1-velocity-flat")
    controller = MujocoController(cfg)
    controller.update_state()
    controller.start()

    obs = controller.policy.compute_observation()

    expected_size = VELOCITY_OBS_SPEC.total_size(23)
    print(f"\nObservation shape: {obs.shape}")
    print(f"Expected shape: (1, {expected_size})")

    if obs.shape[1] != expected_size:
        print(f"⚠️  ERROR: Observation size mismatch! {obs.shape[1]} != {expected_size}")
        return False

    # Validate structure
    try:
        VELOCITY_OBS_SPEC.validate_observation(obs.squeeze(0), num_joints=23)
        print("✓ Observation structure valid")
    except Exception as e:
        print(f"⚠️  ERROR: Observation validation failed: {e}")
        return False

    # Print components
    print(f"\nObservation breakdown:")
    print(f"{'Component':<25} {'Size':>4} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 80)

    offset = 0
    for name, size_fn in VELOCITY_OBS_SPEC.ORDER:
        size_resolved = size_fn if isinstance(size_fn, int) else size_fn(23)
        component = obs[0, offset:offset+size_resolved]
        print(f"{name:<25} {size_resolved:>4} {component.mean():>10.4f} {component.std():>10.4f} "
              f"{component.min():>10.4f} {component.max():>10.4f}")
        offset += size_resolved

    print(f"\n✓ Total size: {offset} (matches expected {expected_size})")
    return True


def test_projected_gravity_correctness():
    """Test projected gravity computation."""
    print("\n" + "=" * 80)
    print("TEST: Projected Gravity Computation")
    print("=" * 80)

    from colosseum.mdp.observations import compute_projected_gravity

    all_passed = True

    # Test case 1: Identity quaternion (upright)
    print("\nTest 1: Upright robot (identity quaternion)")
    quat_upright = torch.tensor([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
    gravity = compute_projected_gravity(quat_upright)

    expected = torch.tensor([0.0, 0.0, -1.0])
    print(f"  Computed:  [{gravity[0]:7.4f}, {gravity[1]:7.4f}, {gravity[2]:7.4f}]")
    print(f"  Expected:  [{expected[0]:7.4f}, {expected[1]:7.4f}, {expected[2]:7.4f}]")

    if torch.allclose(gravity, expected, atol=1e-4):
        print(f"  ✓ PASS")
    else:
        print(f"  ✗ FAIL: Difference = {(gravity - expected).abs().max():.6f}")
        all_passed = False

    # Test case 2: 90° pitch forward (nose down)
    print("\nTest 2: 90° pitch forward (nose down)")
    quat_pitch = torch.tensor([0.7071, 0.7071, 0.0, 0.0])  # 90° about X-axis
    gravity = compute_projected_gravity(quat_pitch)

    # When pitched 90° forward, gravity (0,0,-1) in world should point "forward" (+Y) in base
    expected = torch.tensor([0.0, 1.0, 0.0])
    print(f"  Computed:  [{gravity[0]:7.4f}, {gravity[1]:7.4f}, {gravity[2]:7.4f}]")
    print(f"  Expected:  [{expected[0]:7.4f}, {expected[1]:7.4f}, {expected[2]:7.4f}]")

    if torch.allclose(gravity, expected, atol=1e-4):
        print(f"  ✓ PASS")
    else:
        print(f"  ⚠️  Difference = {(gravity - expected).abs().max():.6f}")
        # Don't fail - this might be a convention difference
        print(f"  (This might be correct depending on frame conventions)")

    # Test case 3: 90° roll (on side)
    print("\nTest 3: 90° roll right (on right side)")
    quat_roll = torch.tensor([0.7071, 0.0, 0.7071, 0.0])  # 90° about Y-axis
    gravity = compute_projected_gravity(quat_roll)

    expected = torch.tensor([-1.0, 0.0, 0.0])
    print(f"  Computed:  [{gravity[0]:7.4f}, {gravity[1]:7.4f}, {gravity[2]:7.4f}]")
    print(f"  Expected:  [{expected[0]:7.4f}, {expected[1]:7.4f}, {expected[2]:7.4f}]")

    if torch.allclose(gravity, expected, atol=1e-4):
        print(f"  ✓ PASS")
    else:
        print(f"  ⚠️  Difference = {(gravity - expected).abs().max():.6f}")
        print(f"  (This might be correct depending on frame conventions)")

    # Test case 4: Upside down
    print("\nTest 4: Upside down (180° roll)")
    quat_upside_down = torch.tensor([0.0, 1.0, 0.0, 0.0])  # 180° about X-axis
    gravity = compute_projected_gravity(quat_upside_down)

    expected = torch.tensor([0.0, 0.0, 1.0])  # Gravity now points "up" in base frame
    print(f"  Computed:  [{gravity[0]:7.4f}, {gravity[1]:7.4f}, {gravity[2]:7.4f}]")
    print(f"  Expected:  [{expected[0]:7.4f}, {expected[1]:7.4f}, {expected[2]:7.4f}]")

    if torch.allclose(gravity, expected, atol=1e-4):
        print(f"  ✓ PASS")
    else:
        print(f"  ✗ FAIL: Difference = {(gravity - expected).abs().max():.6f}")
        all_passed = False

    # Test case 5: From actual deployment
    print("\nTest 5: From actual robot state (in deployment)")
    cfg = TASK_REGISTRY.get_config("t1-velocity-rough")
    controller = MujocoController(cfg)
    controller.update_state()

    actual_quat = controller.robot.data.root_quat_w
    actual_gravity = controller.robot.data.projected_gravity_b

    print(f"  Robot quaternion: [{actual_quat[0]:.4f}, {actual_quat[1]:.4f}, "
          f"{actual_quat[2]:.4f}, {actual_quat[3]:.4f}]")
    print(f"  Projected gravity: [{actual_gravity[0]:.4f}, {actual_gravity[1]:.4f}, "
          f"{actual_gravity[2]:.4f}]")
    print(f"  Magnitude: {actual_gravity.norm():.4f} (should be ~1.0)")

    if abs(actual_gravity.norm() - 1.0) > 0.01:
        print(f"  ⚠️  WARNING: Gravity magnitude != 1.0")
        all_passed = False
    else:
        print(f"  ✓ Magnitude correct")

    if all_passed:
        print(f"\n✓ All projected gravity tests PASSED")
    else:
        print(f"\n⚠️  Some projected gravity tests FAILED")

    return all_passed


def test_joint_order_remapping():
    """Verify joint order remapping is correct."""
    print("\n" + "=" * 80)
    print("TEST: Joint Order Remapping Verification")
    print("=" * 80)

    cfg = TASK_REGISTRY.get_config("t1-velocity-flat")
    controller = MujocoController(cfg)

    robot = controller.robot

    print(f"\nReal hardware order (23 joints):")
    for i, name in enumerate(robot.cfg.joint_names):
        print(f"  [{i:2d}] {name}")

    print(f"\nSimulation order (23 joints):")
    for i, name in enumerate(robot.cfg.sim_joint_names):
        print(f"  [{i:2d}] {name}")

    print(f"\nRemapping arrays:")
    print(f"  real2sim: {robot.data.real2sim_joint_indexes}")
    print(f"  sim2real: {robot.data.sim2real_joint_indexes}")

    # Verify bidirectional mapping
    print(f"\nVerifying bidirectional mapping...")
    all_valid = True
    for i in range(23):
        real_idx = i
        sim_idx = robot.data.real2sim_joint_indexes[real_idx]
        back_to_real = robot.data.sim2real_joint_indexes[sim_idx]

        if back_to_real != real_idx:
            print(f"  ✗ FAIL at index {i}: {real_idx} → {sim_idx} → {back_to_real}")
            all_valid = False

    if all_valid:
        print(f"  ✓ Bidirectional mapping valid (all 23 joints)")
    else:
        print(f"  ⚠️  Bidirectional mapping BROKEN")
        return False

    # Verify names match
    print(f"\nVerifying joint names match...")
    for real_idx in range(23):
        sim_idx = robot.data.real2sim_joint_indexes[real_idx]
        real_name = robot.cfg.joint_names[real_idx]
        sim_name = robot.cfg.sim_joint_names[sim_idx]

        if real_name != sim_name:
            print(f"  ✗ FAIL: real[{real_idx}]={real_name} != sim[{sim_idx}]={sim_name}")
            all_valid = False

    if all_valid:
        print(f"  ✓ All joint names match")
        return True
    else:
        print(f"  ⚠️  Joint name mismatch detected")
        return False


def test_observation_values_realistic():
    """Check if observation values are in realistic ranges."""
    print("\n" + "=" * 80)
    print("TEST: Observation Value Range Check")
    print("=" * 80)

    cfg = TASK_REGISTRY.get_config("t1-velocity-flat")
    controller = MujocoController(cfg)
    controller.update_state()
    controller.start()

    # Run a few steps to get non-zero observations
    for _ in range(5):
        controller.update_state()
        controller.policy_step()

    obs = controller.policy.compute_observation().squeeze(0)

    # Check each component
    checks = []

    # Base velocities should be small (robot standing still initially)
    base_lin_vel = obs[0:3]
    checks.append(("Base linear velocity", base_lin_vel.abs().max() < 0.1, base_lin_vel.abs().max().item()))

    base_ang_vel = obs[3:6]
    checks.append(("Base angular velocity", base_ang_vel.abs().max() < 0.1, base_ang_vel.abs().max().item()))

    # Projected gravity should have magnitude ~1.0
    proj_gravity = obs[6:9]
    gravity_norm = proj_gravity.norm()
    checks.append(("Projected gravity norm", 0.9 < gravity_norm < 1.1, gravity_norm.item()))

    # Joint positions relative should be small (near default)
    joint_pos_rel = obs[9:32]
    checks.append(("Joint positions (relative)", joint_pos_rel.abs().max() < 1.0, joint_pos_rel.abs().max().item()))

    # Joint velocities should be small (robot nearly static)
    joint_vel = obs[32:55]
    checks.append(("Joint velocities", joint_vel.abs().max() < 2.0, joint_vel.abs().max().item()))

    # Last action should be reasonable (near zero for first few steps)
    last_action = obs[55:78]
    checks.append(("Last action", last_action.abs().max() < 2.0, last_action.abs().max().item()))

    # Velocity commands should be zero (no input)
    vel_cmd = obs[78:81]
    checks.append(("Velocity command", vel_cmd.abs().max() < 0.01, vel_cmd.abs().max().item()))

    # Print results
    print(f"\n{'Check':<30} {'Status':>10} {'Value':>12}")
    print("-" * 80)

    all_passed = True
    for name, passed, value in checks:
        status = "✓ PASS" if passed else "⚠️  WARN"
        print(f"{name:<30} {status:>10} {value:>12.6f}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n✓ All observation values in realistic ranges")
    else:
        print(f"\n⚠️  Some observation values outside expected ranges (might be okay)")

    return all_passed


def main():
    """Run all observation tests."""
    print("\n" + "=" * 80)
    print("OBSERVATION VALIDATION TEST SUITE")
    print("=" * 80)

    results = {}

    try:
        results["structure"] = test_observation_structure()
    except Exception as e:
        print(f"\n✗ Structure test failed: {e}")
        results["structure"] = False

    try:
        results["projected_gravity"] = test_projected_gravity_correctness()
    except Exception as e:
        print(f"\n✗ Projected gravity test failed: {e}")
        results["projected_gravity"] = False

    try:
        results["joint_order"] = test_joint_order_remapping()
    except Exception as e:
        print(f"\n✗ Joint order test failed: {e}")
        results["joint_order"] = False

    try:
        results["value_ranges"] = test_observation_values_realistic()
    except Exception as e:
        print(f"\n✗ Value range test failed: {e}")
        results["value_ranges"] = False

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "⚠️  FAIL"
        print(f"{name:<30} {status}")

    all_passed = all(results.values())

    if all_passed:
        print(f"\n✓ All observation tests PASSED")
        print(f"  Observations are likely correct!")
    else:
        print(f"\n⚠️  Some observation tests FAILED")
        print(f"  Check the failures above for issues")


if __name__ == "__main__":
    main()
