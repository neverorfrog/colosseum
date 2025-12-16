# Quick Diagnostic Commands

## 🚀 Start Here

**Quick stability test (recommended first):**
```bash
pixi run python tests/test_sleep_timing.py
```
⏱️ Takes ~20 seconds, directly compares control loop timing strategies

---

## 📊 All Diagnostic Tests

### 1. Sleep Timing Comparison ⭐ RECOMMENDED
```bash
pixi run python tests/test_sleep_timing.py
```

**Tests 3 scenarios:**
- Fixed 20ms sleep (current implementation)
- Minimal 1ms sleep (no artificial delay)
- Adaptive timing (like play.py)

**What to look for:**
```
COMPARISON SUMMARY
Method               Mean Loop Time       Z Drop     Status
--------------------------------------------------------------------------------
Fixed Sleep                    20.45ms     0.1234m    FALLING
Minimal Sleep                   1.23ms     0.0012m     STABLE
Adaptive Timing                20.15ms     0.0008m     STABLE
```

If you see:
- ✅ Fixed = FALLING, Others = STABLE → **Timing is the issue!**
- ⚠️ All = FALLING → Multiple issues present
- ✅ All = STABLE → Increase test duration or add velocity commands

---

### 2. Observation Validation ⭐ NEW
```bash
pixi run python tests/test_observations.py
```

**Tests observation computation:**
- Observation structure and size
- Projected gravity correctness (quaternion math)
- Joint order remapping (real vs sim)
- Realistic value ranges

**What to look for:**
```
✓ All observation tests PASSED
  Observations are likely correct!
```

---

### 3. ONNX Model Consistency ⭐ NEW
```bash
pixi run python tests/test_onnx_consistency.py
```

**Tests ONNX model export:**
- Model loading and metadata
- Inference determinism
- Output ranges
- Responsiveness to input changes
- Real observation inference

**What to look for:**
```
✓ All ONNX tests PASSED
  ONNX model is likely correct!
```

---

### 4. Full Diagnostics Suite
```bash
pixi run python tests/test_deployment_diagnostics.py
```

**Runs 5 comprehensive tests:**
1. Control loop timing analysis
2. Action scaling verification
3. Actuator configuration check
4. Physics solver settings
5. State freshness measurement

**What to look for:**
```
✓ Timing Analysis:
  - Total latency: 21.5ms
  - Primary issue: Fixed sleep before state update

⚠️  Action Scaling:
  - Per-joint scaling (may differ from training)

✓ Actuators: Configuration matches RobotConfig

⚠️  Solver: Settings differ from training
```

---

### 3. Save Results to File
```bash
pixi run python tests/test_sleep_timing.py > sleep_results.txt
pixi run python tests/test_deployment_diagnostics.py > full_results.txt
```

Then review the files for detailed analysis.

---

## 🐛 Quick Debug (Live During Deployment)

Add real-time diagnostics to your running deployment:

**Option 1: Minimal (easiest)**

Add to [mujoco.py:117](src/colosseum/deploy/backends/mujoco.py#L117) in the `while` loop:
```python
# Add at start of loop
loop_start = time.perf_counter()

# Add before viewer.sync()
if self._step_count % 50 == 0:
    loop_time = time.perf_counter() - loop_start
    base_z = self.robot.data.root_pos_w[2].item()
    print(f"Step {self._step_count:4d}: {loop_time*1000:6.2f}ms, z={base_z:.4f}m")
```

**Option 2: Full instrumentation**

See [tests/quick_debug.py](tests/quick_debug.py) for complete instrumented run() method

Then run deployment normally:
```bash
pixi run python -m colosseum.deploy.backends.mujoco --task t1-velocity-flat
```

---

## 🔍 Interpreting Results

### Sleep Timing Test

**✅ Good Result (Confirms Timing Hypothesis):**
```
Fixed Sleep                    20.45ms     0.1234m    FALLING ← Problem
Minimal Sleep                   1.23ms     0.0012m     STABLE ← Fixed!
Adaptive Timing                20.15ms     0.0008m     STABLE ← Best!
```
→ **Action:** Implement adaptive timing fix (Priority 1)

**⚠️ Bad Result (Multiple Issues):**
```
Fixed Sleep                    20.45ms     0.1234m    FALLING
Minimal Sleep                   1.23ms     0.1156m    FALLING ← Still falling!
Adaptive Timing                20.15ms     0.1089m    FALLING
```
→ **Action:** Check action scaling and solver settings (run full diagnostics)

---

### Full Diagnostics Test

**Key Metrics to Check:**

1. **Total Latency** (Test 1)
   - Good: <5ms
   - Warning: 10-15ms
   - **Problem: >20ms** ← Current issue

2. **Action Scaling** (Test 2)
   - Good: "Uniform scaling: 0.250000"
   - **Problem: "Per-joint scaling (varies by 45%)"**

3. **Actuators** (Test 3)
   - Good: "All actuators match RobotConfig"
   - Problem: Any mismatches listed

4. **Solver** (Test 4)
   - Good: "All solver settings match training"
   - **Problem: "Iterations: 50 vs 10 (training)"**

5. **State Freshness** (Test 5)
   - Good: <2ms from observation to physics
   - **Problem: Observation captured BEFORE physics**

---

## 🔧 Quick Fixes Based on Results

### If timing test confirms the issue:

**Fix 1: Remove fixed sleep** (quick test)
```python
# In mujoco.py line 118, replace:
sleep(self.cfg.physics_dt * self.cfg.mujoco.decimation)

# With:
sleep(0.001)  # Minimal sleep
```

**Fix 2: Implement adaptive timing** (proper fix)
- See plan document for full implementation
- Reference: `external/mjlab/src/mjlab/viewer/base.py:231-262`

### If solver settings differ:

**Fix: Match training configuration**
```python
# In mujoco.py after self.mj_model = spec.compile()
self.mj_model.opt.iterations = 10      # Match training
self.mj_model.opt.ls_iterations = 20   # Match training
```

### If action scaling is per-joint:

**Verify with training:** Need to check if training also uses per-joint or uniform

---

## 📁 Test File Locations

```
tests/
├── test_sleep_timing.py              ← Quick stability test (⭐ START HERE)
├── test_observations.py              ← Observation computation validation (⭐ NEW)
├── test_onnx_consistency.py          ← ONNX model correctness (⭐ NEW)
├── test_deployment_diagnostics.py    ← Full analysis suite
├── quick_debug.py                    ← Code snippets for live debugging
├── ALL_POTENTIAL_ISSUES.md           ← Complete issue catalog (⭐ NEW)
├── DIAGNOSTICS_README.md             ← Detailed documentation
└── RUN_DIAGNOSTICS.md                ← This file (quick reference)
```

---

## ❓ Common Issues

**"ModuleNotFoundError: No module named 'colosseum'"**
→ Use `pixi run python` (not bare `python`)

**Tests crash immediately**
→ Check policy checkpoint exists and is loadable

**All tests show robot is stable**
→ Increase test duration: edit `num_steps` in test files

**Want to see robot visually during test**
→ Tests run headless. For visual, run: `pixi run python -m colosseum.deploy.backends.mujoco --task t1-velocity-flat`

---

## 📋 Recommended Workflow

1. **Quick stability test:** `pixi run python tests/test_sleep_timing.py`
   - Confirms if timing is the primary issue (~20 seconds)

2. **Observation validation:** `pixi run python tests/test_observations.py`
   - Verifies observation computation correctness (~10 seconds)

3. **ONNX consistency:** `pixi run python tests/test_onnx_consistency.py`
   - Checks ONNX model export correctness (~30 seconds)

4. **Full diagnostics:** `pixi run python tests/test_deployment_diagnostics.py`
   - Comprehensive analysis of all issues (~40 seconds)

5. **If timing confirmed:** Implement adaptive timing fix
   - See plan document for implementation details

6. **Address other issues:** Based on test results
   - Observation issues: Fix projected gravity or joint order
   - ONNX issues: Re-export model or use TorchScript
   - Solver settings: Match training configuration

7. **Verify fix:** Re-run all tests after implementing changes
   - Should see "STABLE" and "PASS" across all tests

---

## 🎯 Expected Timeline

- **Quick test:** 20 seconds
- **Full diagnostics:** 40 seconds
- **Review results:** 5 minutes
- **Implement timing fix:** 30-60 minutes
- **Test deployment:** 5 minutes
- **Total:** ~1.5 hours to diagnose and fix

**Good luck! 🚀**
