"""Quick debug script to add diagnostic prints to MujocoController.

Add this to your mujoco.py run() method to see real-time diagnostics.
"""

import time


# ============================================================================
# Add this to MujocoController.__init__() after super().__init__(cfg):
# ============================================================================

def init_diagnostics(self):
    """Initialize diagnostic tracking (add to __init__)."""
    self._diag_enabled = True
    self._diag_step = 0
    self._diag_loop_times = []
    self._diag_base_heights = []
    self._diag_start_time = None
    print("\n" + "="*60)
    print("DIAGNOSTICS ENABLED")
    print("="*60)


# ============================================================================
# Replace your run() method with this instrumented version:
# ============================================================================

def run_with_diagnostics(self):
    """Instrumented run() method with diagnostic output."""
    import numpy as np
    import mujoco.viewer

    with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
        self.viewer = viewer
        cam = self.viewer.cam
        cam.elevation = -20

        if self.vel_command is not None and self.input_source is not None:
            print(f"\n{self.input_source.get_operation_hint()}")

        # Initial state
        self.update_state()
        self.start()

        # Diagnostic setup
        if self._diag_enabled:
            self._diag_start_time = time.perf_counter()
            print(f"\nTarget control frequency: {1/(self.cfg.physics_dt * self.cfg.mujoco.decimation):.1f}Hz")
            print(f"Physics timestep: {self.cfg.physics_dt*1000:.2f}ms")
            print(f"Decimation: {self.cfg.mujoco.decimation}x")
            print(f"\nStarting control loop...\n")

        # Control loop
        while self.viewer.is_running() and self.is_running:
            loop_start = time.perf_counter()

            # ORIGINAL: Fixed sleep
            sleep_start = time.perf_counter()
            time.sleep(self.cfg.physics_dt * self.cfg.mujoco.decimation)
            sleep_time = time.perf_counter() - sleep_start

            # State update
            update_start = time.perf_counter()
            self.update_state()
            update_time = time.perf_counter() - update_start

            # Policy inference
            policy_start = time.perf_counter()
            dof_targets = self.policy_step()
            policy_time = time.perf_counter() - policy_start

            # Control step
            ctrl_start = time.perf_counter()
            self.ctrl_step(dof_targets)
            ctrl_time = time.perf_counter() - ctrl_start

            # Viewer update
            cam.lookat[:] = self.mj_data.qpos.astype(np.float32)[0:3]
            self.viewer.sync()

            # Diagnostics
            if self._diag_enabled:
                loop_time = time.perf_counter() - loop_start
                self._diag_loop_times.append(loop_time)

                base_z = self.robot.data.root_pos_w[2].item()
                self._diag_base_heights.append(base_z)

                self._diag_step += 1

                # Print every 50 steps
                if self._diag_step % 50 == 0:
                    elapsed = time.perf_counter() - self._diag_start_time
                    avg_loop = np.mean(self._diag_loop_times[-50:]) * 1000
                    avg_sleep = np.mean([sleep_time]) * 1000  # Just current sleep
                    z_initial = self._diag_base_heights[0]
                    z_current = base_z
                    z_drop = z_initial - z_current

                    print(f"Step {self._diag_step:4d} | "
                          f"Elapsed: {elapsed:6.1f}s | "
                          f"Loop: {avg_loop:6.2f}ms | "
                          f"Sleep: {avg_sleep:5.2f}ms | "
                          f"Base Z: {z_current:.4f}m | "
                          f"Drop: {z_drop*100:5.2f}cm | "
                          f"{'FALLING!' if z_drop > 0.1 else 'OK' if z_drop < 0.05 else 'Unstable'}")

                    # Detailed breakdown every 200 steps
                    if self._diag_step % 200 == 0:
                        print(f"  └─ Breakdown: sleep={sleep_time*1000:.2f}ms, "
                              f"update={update_time*1000:.2f}ms, "
                              f"policy={policy_time*1000:.2f}ms, "
                              f"ctrl={ctrl_time*1000:.2f}ms")
                        total_latency = sleep_time + update_time + policy_time
                        print(f"  └─ Obs Latency: {total_latency*1000:.2f}ms "
                              f"({total_latency/(self.cfg.physics_dt * self.cfg.mujoco.decimation):.2f}x target)")

        # Final diagnostics
        if self._diag_enabled:
            print("\n" + "="*60)
            print("DIAGNOSTICS SUMMARY")
            print("="*60)
            loop_times_ms = np.array(self._diag_loop_times) * 1000
            print(f"\nLoop timing:")
            print(f"  Mean:   {loop_times_ms.mean():.2f}ms")
            print(f"  Std:    {loop_times_ms.std():.2f}ms")
            print(f"  Target: {self.cfg.physics_dt * self.cfg.mujoco.decimation * 1000:.2f}ms")

            z_initial = self._diag_base_heights[0]
            z_final = self._diag_base_heights[-1]
            z_min = min(self._diag_base_heights)
            z_drop = z_initial - z_final
            print(f"\nBase height:")
            print(f"  Initial: {z_initial:.4f}m")
            print(f"  Final:   {z_final:.4f}m")
            print(f"  Min:     {z_min:.4f}m")
            print(f"  Drop:    {z_drop:.4f}m ({z_drop/z_initial*100:.1f}%)")

            if z_drop > 0.1:
                print(f"\n⚠️  ROBOT FALLING: Lost {z_drop*100:.1f}cm of height")
            elif z_drop > 0.05:
                print(f"\n⚠️  Unstable: Lost {z_drop*100:.1f}cm of height")
            else:
                print(f"\n✓ Stable: Height maintained")


# ============================================================================
# USAGE INSTRUCTIONS:
# ============================================================================
"""
To enable diagnostics in your MujocoController:

1. In mujoco.py __init__, add after super().__init__(cfg):

   # Diagnostics
   self._diag_enabled = True
   self._diag_step = 0
   self._diag_loop_times = []
   self._diag_base_heights = []
   self._diag_start_time = None

2. Replace the run() method body with the instrumented version above, OR
   add the diagnostic tracking code to your existing run() method.

3. Run deployment:
   pixi run python -m colosseum.deploy.backends.mujoco --task t1-velocity-flat

4. Watch for:
   - Loop time consistently > 20ms indicates overhead issues
   - Base Z dropping indicates robot falling
   - "FALLING!" message indicates confirmed instability

Example output:
   Step   50 | Elapsed:    1.0s | Loop:  20.45ms | Sleep: 20.01ms | Base Z: 0.6950m | Drop:  0.00cm | OK
   Step  100 | Elapsed:    2.0s | Loop:  20.46ms | Sleep: 20.01ms | Base Z: 0.6940m | Drop:  0.10cm | OK
   Step  150 | Elapsed:    3.0s | Loop:  20.47ms | Sleep: 20.01ms | Base Z: 0.6920m | Drop:  0.30cm | Unstable
   Step  200 | Elapsed:    4.0s | Loop:  20.48ms | Sleep: 20.01ms | Base Z: 0.6850m | Drop:  1.00cm | FALLING!
     └─ Breakdown: sleep=20.01ms, update=0.12ms, policy=0.18ms, ctrl=0.15ms
     └─ Obs Latency: 20.31ms (1.02x target)

If you see "FALLING!" with high observation latency, the timing hypothesis is confirmed!
"""


# ============================================================================
# ALTERNATIVE: Minimal instrumentation (just add prints to existing code)
# ============================================================================
"""
Simplest option - just add these lines to your existing run() method:

    # Add at start of loop
    loop_start = time.perf_counter()

    # Add before viewer.sync()
    if self._step_count % 50 == 0:
        loop_time = time.perf_counter() - loop_start
        base_z = self.robot.data.root_pos_w[2].item()
        print(f"Step {self._step_count:4d}: {loop_time*1000:6.2f}ms, z={base_z:.4f}m")
"""


if __name__ == "__main__":
    print(__doc__)
