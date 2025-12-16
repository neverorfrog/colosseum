"""Quick test to compare control loop with and without fixed sleep.

This test helps confirm the timing hypothesis by running the same policy
with different sleep strategies.
"""

import time
import numpy as np
import torch

from colosseum.deploy.core.registry import TASK_REGISTRY
from colosseum.deploy.backends.mujoco import MujocoController
from colosseum.deploy.core.registry import auto_register_tasks
auto_register_tasks()

def test_with_fixed_sleep(controller: MujocoController, num_steps: int = 100):
    """Test control loop with fixed sleep (current implementation)."""
    print("\n" + "=" * 60)
    print("TEST: Fixed Sleep (Current Implementation)")
    print("=" * 60)

    target_dt = controller.cfg.physics_dt * controller.cfg.mujoco.decimation
    controller.update_state()
    controller.start()

    loop_times = []
    base_z_positions = []

    print(f"Running {num_steps} steps with fixed {target_dt*1000:.1f}ms sleep...")

    for i in range(num_steps):
        loop_start = time.perf_counter()

        # FIXED SLEEP FIRST (current implementation)
        time.sleep(target_dt)

        controller.update_state()
        dof_targets = controller.policy_step()
        controller.ctrl_step(dof_targets)

        loop_time = time.perf_counter() - loop_start
        loop_times.append(loop_time)

        # Track base height to detect falling
        base_z = controller.robot.data.root_pos_w[2].item()
        base_z_positions.append(base_z)

        if i % 20 == 0 or i == num_steps - 1:
            print(f"Step {i:3d}: {loop_time*1000:6.2f}ms, base_z={base_z:.4f}m")

    loop_times_ms = np.array(loop_times) * 1000
    print(f"\nLoop timing statistics:")
    print(f"  Mean:   {loop_times_ms.mean():.2f}ms (target: {target_dt*1000:.2f}ms)")
    print(f"  Std:    {loop_times_ms.std():.2f}ms")
    print(f"  Overhead: {(loop_times_ms.mean() - target_dt*1000):.2f}ms")

    # Check for falling
    initial_z = base_z_positions[0]
    final_z = base_z_positions[-1]
    min_z = min(base_z_positions)
    z_drop = initial_z - final_z

    print(f"\nBase height analysis:")
    print(f"  Initial: {initial_z:.4f}m")
    print(f"  Final:   {final_z:.4f}m")
    print(f"  Min:     {min_z:.4f}m")
    print(f"  Drop:    {z_drop:.4f}m ({z_drop/initial_z*100:.1f}%)")

    if z_drop > 0.1:
        print(f"  ⚠️  ROBOT FALLING: Dropped {z_drop*100:.1f}cm")
    elif z_drop > 0.05:
        print(f"  ⚠️  Unstable: Height loss {z_drop*100:.1f}cm")
    else:
        print(f"  ✓ Stable")

    return {
        "loop_times_ms": loop_times_ms,
        "base_z": base_z_positions,
        "z_drop": z_drop,
        "stable": z_drop < 0.05,
    }


def test_with_minimal_sleep(controller: MujocoController, num_steps: int = 100):
    """Test control loop with minimal sleep (no fixed delay)."""
    print("\n" + "=" * 60)
    print("TEST: Minimal Sleep (Hypothesis Fix)")
    print("=" * 60)

    # Reset robot to initial state
    controller.mj_data.qpos[:] = np.concatenate([
        np.array(controller.cfg.mujoco.init_pos, dtype=np.float32),
        np.array(controller.cfg.mujoco.init_quat, dtype=np.float32),
        controller.robot.default_joint_pos.numpy(),
    ])
    controller.mj_data.qvel[:] = 0.0
    import mujoco
    mujoco.mj_forward(controller.mj_model, controller.mj_data)

    controller.update_state()
    controller.start()

    target_dt = controller.cfg.physics_dt * controller.cfg.mujoco.decimation
    loop_times = []
    base_z_positions = []

    print(f"Running {num_steps} steps with minimal (1ms) sleep...")

    for i in range(num_steps):
        loop_start = time.perf_counter()

        controller.update_state()
        dof_targets = controller.policy_step()
        controller.ctrl_step(dof_targets)

        # MINIMAL SLEEP (1ms) - just yield to OS
        time.sleep(0.001)

        loop_time = time.perf_counter() - loop_start
        loop_times.append(loop_time)

        # Track base height
        base_z = controller.robot.data.root_pos_w[2].item()
        base_z_positions.append(base_z)

        if i % 20 == 0 or i == num_steps - 1:
            print(f"Step {i:3d}: {loop_time*1000:6.2f}ms, base_z={base_z:.4f}m")

    loop_times_ms = np.array(loop_times) * 1000
    print(f"\nLoop timing statistics:")
    print(f"  Mean:   {loop_times_ms.mean():.2f}ms (target: {target_dt*1000:.2f}ms)")
    print(f"  Std:    {loop_times_ms.std():.2f}ms")
    print(f"  Note: Running faster than target (physics may be ahead of real-time)")

    # Check for falling
    initial_z = base_z_positions[0]
    final_z = base_z_positions[-1]
    min_z = min(base_z_positions)
    z_drop = initial_z - final_z

    print(f"\nBase height analysis:")
    print(f"  Initial: {initial_z:.4f}m")
    print(f"  Final:   {final_z:.4f}m")
    print(f"  Min:     {min_z:.4f}m")
    print(f"  Drop:    {z_drop:.4f}m ({z_drop/initial_z*100:.1f}%)")

    if z_drop > 0.1:
        print(f"  ⚠️  ROBOT FALLING: Dropped {z_drop*100:.1f}cm")
    elif z_drop > 0.05:
        print(f"  ⚠️  Unstable: Height loss {z_drop*100:.1f}cm")
    else:
        print(f"  ✓ Stable")

    return {
        "loop_times_ms": loop_times_ms,
        "base_z": base_z_positions,
        "z_drop": z_drop,
        "stable": z_drop < 0.05,
    }


def test_with_adaptive_timing(controller: MujocoController, num_steps: int = 100):
    """Test control loop with adaptive timing (like play.py)."""
    print("\n" + "=" * 60)
    print("TEST: Adaptive Timing (Play.py Style)")
    print("=" * 60)

    # Reset robot to initial state
    controller.mj_data.qpos[:] = np.concatenate([
        np.array(controller.cfg.mujoco.init_pos, dtype=np.float32),
        np.array(controller.cfg.mujoco.init_quat, dtype=np.float32),
        controller.robot.default_joint_pos.numpy(),
    ])
    controller.mj_data.qvel[:] = 0.0
    import mujoco
    mujoco.mj_forward(controller.mj_model, controller.mj_data)

    controller.update_state()
    controller.start()

    target_dt = controller.cfg.physics_dt * controller.cfg.mujoco.decimation
    loop_times = []
    base_z_positions = []

    # Adaptive timing state
    time_until_next_frame = 0.0
    last_time = time.perf_counter()

    print(f"Running {num_steps} steps with adaptive timing...")

    step = 0
    while step < num_steps:
        loop_start = time.perf_counter()

        # Calculate elapsed time since last frame
        current_time = time.perf_counter()
        elapsed = current_time - last_time
        last_time = current_time

        time_until_next_frame -= elapsed

        # Skip frame if we're ahead of schedule
        if time_until_next_frame > 0:
            time.sleep(0.001)  # Minimal sleep to yield
            continue

        # Reset timer for next frame
        time_until_next_frame += target_dt
        if time_until_next_frame < -target_dt:
            time_until_next_frame = 0.0  # Catch up if too far behind

        # Execute control step
        controller.update_state()
        dof_targets = controller.policy_step()
        controller.ctrl_step(dof_targets)

        loop_time = time.perf_counter() - loop_start
        loop_times.append(loop_time)

        # Track base height
        base_z = controller.robot.data.root_pos_w[2].item()
        base_z_positions.append(base_z)

        if step % 20 == 0 or step == num_steps - 1:
            print(f"Step {step:3d}: {loop_time*1000:6.2f}ms, base_z={base_z:.4f}m, "
                  f"next_frame_in={time_until_next_frame*1000:.1f}ms")

        step += 1

    loop_times_ms = np.array(loop_times) * 1000
    print(f"\nLoop timing statistics:")
    print(f"  Mean:   {loop_times_ms.mean():.2f}ms (target: {target_dt*1000:.2f}ms)")
    print(f"  Std:    {loop_times_ms.std():.2f}ms")

    # Check for falling
    initial_z = base_z_positions[0]
    final_z = base_z_positions[-1]
    min_z = min(base_z_positions)
    z_drop = initial_z - final_z

    print(f"\nBase height analysis:")
    print(f"  Initial: {initial_z:.4f}m")
    print(f"  Final:   {final_z:.4f}m")
    print(f"  Min:     {min_z:.4f}m")
    print(f"  Drop:    {z_drop:.4f}m ({z_drop/initial_z*100:.1f}%)")

    if z_drop > 0.1:
        print(f"  ⚠️  ROBOT FALLING: Dropped {z_drop*100:.1f}cm")
    elif z_drop > 0.05:
        print(f"  ⚠️  Unstable: Height loss {z_drop*100:.1f}cm")
    else:
        print(f"  ✓ Stable")

    return {
        "loop_times_ms": loop_times_ms,
        "base_z": base_z_positions,
        "z_drop": z_drop,
        "stable": z_drop < 0.05,
    }


def main():
    """Run sleep timing comparison tests."""
    print("\n" + "=" * 80)
    print("SLEEP TIMING COMPARISON TEST")
    print("=" * 80)
    print("\nThis test compares robot stability with different control loop timing:")
    print("  1. Fixed Sleep (current): Sleep BEFORE state update (20ms delay)")
    print("  2. Minimal Sleep: Sleep AFTER control (1ms, no artificial delay)")
    print("  3. Adaptive Timing: Like play.py (frame skipping)")

    task_name = "t1-velocity-rough"
    cfg = TASK_REGISTRY.get_config(task_name)

    # Test 1: Fixed sleep (current implementation)
    controller1 = MujocoController(cfg)
    result1 = test_with_fixed_sleep(controller1, num_steps=100)

    # Test 2: Minimal sleep
    controller2 = MujocoController(cfg)
    result2 = test_with_minimal_sleep(controller2, num_steps=100)

    # Test 3: Adaptive timing
    controller3 = MujocoController(cfg)
    result3 = test_with_adaptive_timing(controller3, num_steps=100)

    # Summary comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'Method':<20} {'Mean Loop Time':>15} {'Z Drop':>12} {'Status':>10}")
    print("-" * 80)
    print(f"{'Fixed Sleep':<20} {result1['loop_times_ms'].mean():>12.2f}ms "
          f"{result1['z_drop']:>10.4f}m {'STABLE' if result1['stable'] else 'FALLING':>10}")
    print(f"{'Minimal Sleep':<20} {result2['loop_times_ms'].mean():>12.2f}ms "
          f"{result2['z_drop']:>10.4f}m {'STABLE' if result2['stable'] else 'FALLING':>10}")
    print(f"{'Adaptive Timing':<20} {result3['loop_times_ms'].mean():>12.2f}ms "
          f"{result3['z_drop']:>10.4f}m {'STABLE' if result3['stable'] else 'FALLING':>10}")

    print("\nKey Observations:")
    if not result1['stable'] and result2['stable']:
        print("  ✓ Removing fixed sleep IMPROVES stability")
        print("  ✓ This confirms the timing hypothesis!")
    elif not result1['stable'] and not result2['stable']:
        print("  ⚠️  Both unstable - timing may not be the only issue")
    else:
        print("  ? All stable - may need longer test duration")

    if result3['stable'] and result3['z_drop'] < result2['z_drop']:
        print("  ✓ Adaptive timing performs BEST")
        print("  ✓ Recommend implementing adaptive timing fix")


if __name__ == "__main__":
    main()
