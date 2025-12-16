# Deployment Diagnostics Tests

This directory contains diagnostic tests to investigate the T1 robot deployment stability issue.

## Quick Start

### Test 1: Sleep Timing Comparison (Recommended First)

This test directly compares robot stability with different control loop timing strategies:

```bash
pixi run python tests/test_sleep_timing.py
```

**What it does:**
- Runs robot with **fixed 20ms sleep** (current implementation)
- Runs robot with **minimal 1ms sleep** (removes artificial delay)
- Runs robot with **adaptive timing** (like play.py)
- Compares stability by tracking base height over 100 steps

**Expected outcome if timing is the issue:**
- Fixed sleep: Robot falls (height drops significantly)
- Minimal/Adaptive: Robot stable (height maintained)

**Runtime:** ~15-20 seconds

---

### Test 2: Full Diagnostics Suite

Comprehensive analysis of all potential issues:

```bash
pixi run python tests/test_deployment_diagnostics.py
```

**What it tests:**

1. **Control Loop Timing** (100 steps)
   - Measures loop execution time, sleep time, overhead
   - Calculates total observation latency
   - Identifies timing bottlenecks

2. **Action Scaling Verification**
   - Prints per-joint action scaling factors
   - Checks if scaling is uniform or per-joint
   - Compares with expected training value (0.25)

3. **Actuator Configuration**
   - Lists all MuJoCo actuators with Kp, Kd, effort limits
   - Compares with RobotConfig expected values
   - Verifies programmatic actuator creation

4. **Solver Settings**
   - Checks MuJoCo timestep, iterations, solver type
   - Compares with training configuration
   - Identifies mismatches

5. **State Freshness Analysis** (50 steps)
   - Tracks time from observation to physics step
   - Measures base position changes
   - Quantifies observation staleness

**Expected findings:**
- Total latency: ~21-24ms (sleep + overhead)
- Action scaling: Likely per-joint (not uniform 0.25)
- Actuators: Should match RobotConfig
- Solver: Iterations likely 50 (default) instead of 10 (training)

**Runtime:** ~30-40 seconds

---

## Interpreting Results

### Sleep Timing Test Results

**If fixed sleep shows falling but minimal/adaptive are stable:**
- ✅ **Confirms timing hypothesis** - this is the primary issue
- ✅ Proceed with adaptive timing fix (Priority 1)

**If all three show falling:**
- ⚠️ Timing is not the only issue
- Check action scaling (TEST 2)
- Check solver settings (TEST 2)
- Verify actuator configuration (TEST 2)

**If all three are stable:**
- May need longer test duration (increase `num_steps`)
- Try with velocity commands (add joystick input)

---

### Full Diagnostics Results

**Test 1: Control Loop Timing**
```
ISSUE: Policy sees observations that are 21.5ms old!
       This is 1.08x the target control period!
```
- **Root cause:** Fixed sleep before state update
- **Impact:** Stale observations cause destabilization
- **Fix:** Implement adaptive timing

**Test 2: Action Scaling**
```
⚠️  Action scaling is PER-JOINT (varies by 45.2%)
⚠️  This might differ from training's uniform 0.25 scaling!
```
- **Issue:** Training uses uniform 0.25, deployment computes per-joint
- **Impact:** May cause unexpected joint ranges
- **Fix:** Match training's uniform scaling formula

**Test 3: Actuator Configuration**
```
✓ All actuators match RobotConfig
```
- **Status:** Actuators configured correctly
- **No action needed**

**Test 4: Solver Settings**
```
⚠️  Iterations mismatch: 50 vs 10
⚠️  LS iterations mismatch: 10 vs 20
```
- **Issue:** Using MuJoCo defaults instead of training values
- **Impact:** Different physics behavior (more accurate but slower)
- **Fix:** Set `model.opt.iterations = 10` and `model.opt.ls_iterations = 20`

**Test 5: State Freshness**
```
Time from observation to physics step:
  Mean:   1.234ms
⚠️  Observations are captured BEFORE physics step,
    but they reflect state from 1.2ms ago + sleep time!
```
- **Issue:** State update ordering compounds latency
- **Fix:** Update state AFTER physics step (or use adaptive timing)

---

## Diagnostic Output Files

Tests print to stdout. To save results:

```bash
pixi run python tests/test_sleep_timing.py > sleep_timing_results.txt
pixi run python tests/test_deployment_diagnostics.py > full_diagnostics_results.txt
```

---

## Next Steps Based on Results

### If timing is confirmed as primary issue:

1. **Implement adaptive timing fix** in `src/colosseum/deploy/backends/mujoco.py`
   - Replace fixed sleep with adaptive frame timing
   - See `external/mjlab/src/mjlab/viewer/base.py:231-262` for reference

2. **Fix solver settings** (quick win)
   ```python
   self.mj_model.opt.iterations = 10
   self.mj_model.opt.ls_iterations = 20
   ```

3. **Verify action scaling** matches training
   - Check if uniform 0.25 or per-joint formula is correct

### If other issues found:

- **Action scaling mismatch:** Compare with training's action manager
- **Actuator issues:** Debug spec compilation and actuator creation
- **Solver differences:** Match training physics configuration

---

## Additional Debug Options

### Run specific test only:

```python
# In test_deployment_diagnostics.py
if __name__ == "__main__":
    test_control_loop_timing(num_steps=100)
    # Or: test_action_scaling()
    # Or: test_actuator_configuration()
    # Or: test_solver_settings()
    # Or: test_state_freshness(num_steps=50)
```

### Change task:

```bash
pixi run python tests/test_sleep_timing.py t1-velocity-rough
pixi run python tests/test_deployment_diagnostics.py t1-velocity-rough
```

### Increase test duration:

Edit `num_steps` parameter in test files (default: 100 for timing, 50 for state freshness)

---

## Troubleshooting

**ImportError: No module named 'colosseum'**
- Make sure to use `pixi run python` (not bare `python`)
- Pixi ensures correct environment with all dependencies

**Viewer window doesn't open**
- Tests run in headless mode (no viewer)
- To see robot visually, run the actual deployment:
  ```bash
  pixi run python -m colosseum.deploy.backends.mujoco --task t1-velocity-flat
  ```

**Tests crash immediately**
- Check policy checkpoint exists
- Verify ONNX model can be loaded
- Check task is registered: `pixi run python -m colosseum.deploy.core.registry`

---

## Summary of Test Files

| File | Purpose | Runtime | Key Insight |
|------|---------|---------|-------------|
| `test_sleep_timing.py` | **Quick stability comparison** | ~20s | Confirms timing hypothesis |
| `test_deployment_diagnostics.py` | **Comprehensive analysis** | ~40s | Identifies all mismatches |

**Recommendation:** Run `test_sleep_timing.py` first to quickly confirm the timing issue, then `test_deployment_diagnostics.py` for complete analysis.
