# Deployment Diagnostics Test Suite

## 🚀 Quick Start

```bash
# Run all 4 diagnostic tests (takes ~100 seconds)
pixi run python tests/test_sleep_timing.py
pixi run python tests/test_observations.py
pixi run python tests/test_onnx_consistency.py
pixi run python tests/test_deployment_diagnostics.py
```

---

## 📚 Documentation Index

### 1. **TEST_SUITE_SUMMARY.md** ⭐ **START HERE**
High-level overview of the entire test suite.
- What each test does
- Why it matters
- How to interpret results
- Success criteria

### 2. **RUN_DIAGNOSTICS.md**
Quick reference guide with copy-paste commands.
- All test commands
- Expected outputs
- Common issues and fixes
- Workflow recommendations

### 3. **ALL_POTENTIAL_ISSUES.md**
Complete catalog of 10 potential deployment issues.
- Detailed descriptions
- Root cause analysis
- Test procedures
- Expected outcomes
- Confidence levels

### 4. **DIAGNOSTICS_README.md**
Detailed technical documentation for each test.
- Test implementations
- Diagnostic output examples
- Troubleshooting guide
- Advanced debugging

---

## 🧪 Test Files

### Core Diagnostic Tests

1. **test_sleep_timing.py** ⭐ **RUN FIRST**
   - **Purpose:** Compare control loop timing strategies
   - **Runtime:** ~20 seconds
   - **Confidence:** 85% (highest)
   - **Tests:** Fixed sleep vs Minimal vs Adaptive timing

2. **test_observations.py** ⭐ **NEW**
   - **Purpose:** Validate observation computation
   - **Runtime:** ~10 seconds
   - **Confidence:** 40%
   - **Tests:** Projected gravity, joint order, structure, values

3. **test_onnx_consistency.py** ⭐ **NEW**
   - **Purpose:** Verify ONNX model correctness
   - **Runtime:** ~30 seconds
   - **Confidence:** 30%
   - **Tests:** Loading, determinism, ranges, responsiveness

4. **test_deployment_diagnostics.py**
   - **Purpose:** Comprehensive analysis of all issues
   - **Runtime:** ~40 seconds
   - **Confidence:** Varies by subtest
   - **Tests:** Timing, action scaling, actuators, solver, state freshness

### Utility Files

5. **quick_debug.py**
   - Code snippets for inline debugging
   - Add to MujocoController.run() for real-time diagnostics

---

## 🎯 Problem Breakdown

### By Confidence Level

**High Confidence (75%+):**
- ✅ Control loop timing issue (85%)

**Medium Confidence (25-75%):**
- ⚠️ Observation computation issues (40%)
- ⚠️ Action scaling mismatch (40%)
- ⚠️ ONNX conversion issues (30%)

**Low Confidence (<25%):**
- ⚠️ Solver settings (15%)
- ⚠️ Joint order remapping (20%)
- ✅ PD gains (5% - verified correct)
- ✅ Init height (5% - already fixed)

### By Test Coverage

**Fully Covered:**
- Control loop timing → test_sleep_timing.py
- Observations → test_observations.py
- ONNX export → test_onnx_consistency.py
- Joint order → test_observations.py

**Partially Covered:**
- Action scaling → test_deployment_diagnostics.py
- Solver settings → test_deployment_diagnostics.py
- Actuators → test_deployment_diagnostics.py

**Already Verified:**
- PD gains → verified matching motor specs
- Init height → fixed (0.665 → 0.70m)
- Decimation → verified matching training

---

## 📋 Recommended Test Order

### Phase 1: Quick Diagnosis (30 seconds)
```bash
pixi run python tests/test_sleep_timing.py
```
**Goal:** Confirm if timing is the primary issue

**Outcome:**
- ✅ Fixed=FALLING, Others=STABLE → Timing is the issue (implement fix)
- ⚠️ All=FALLING → Multiple issues (proceed to Phase 2)
- ✅ All=STABLE → Issue intermittent or config-dependent

---

### Phase 2: Deep Dive (40 seconds)
```bash
pixi run python tests/test_observations.py
pixi run python tests/test_onnx_consistency.py
```
**Goal:** Check observation computation and ONNX model

**Outcome:**
- ✅ All pass → Observations and ONNX are correct
- ⚠️ Observations fail → Fix projected gravity or joint order
- ⚠️ ONNX fails → Re-export model or use TorchScript

---

### Phase 3: Comprehensive Analysis (40 seconds)
```bash
pixi run python tests/test_deployment_diagnostics.py
```
**Goal:** Full analysis of all potential mismatches

**Outcome:**
- Detailed breakdown of timing, action scaling, solver, etc.
- Use to find remaining issues after fixing primary ones

---

## 🔍 How to Use Results

### If Only Timing Test Fails
**Diagnosis:** Control loop timing is the issue (85% confidence)

**Fix:**
1. Implement adaptive timing in MujocoController.run()
2. Set solver iterations to match training (quick win)
3. Re-test to confirm stability

**Files to modify:**
- `src/colosseum/deploy/backends/mujoco.py` (lines 107-124)

---

### If Observation Test Fails
**Diagnosis:** Observation computation has bugs (40% confidence)

**Common issues:**
- Projected gravity: Wrong quaternion convention
- Joint order: Remapping broken
- Observation order: Doesn't match training

**Fix:**
1. Compare with training observation computation
2. Fix projected gravity if needed
3. Verify joint order remapping
4. Re-test to confirm observations correct

**Files to check:**
- `src/colosseum/mdp/observations.py` (projected gravity)
- `src/colosseum/deploy/core/robot.py` (joint remapping)
- `src/colosseum/tasks/velocity/deploy/t1_23dof/policy.py` (observation building)

---

### If ONNX Test Fails
**Diagnosis:** ONNX export has issues (30% confidence)

**Common issues:**
- Model not responding to inputs (frozen)
- Non-deterministic inference
- Missing observation normalizer

**Fix:**
1. Re-export with correct ONNX settings
2. Check Holosoma's ONNX export for reference
3. Consider using TorchScript instead
4. Re-test to confirm model correct

**Files to check:**
- `src/colosseum/deploy/utils/export.py` (export configuration)
- `external/mjlab/src/mjlab/tasks/velocity/rl/exporter.py` (ONNX export)

---

### If Full Diagnostics Shows Multiple Issues
**Diagnosis:** Compound problem (multiple issues)

**Fix priority:**
1. Fix observations first (most fundamental)
2. Fix ONNX export (affects inference)
3. Fix timing (affects control)
4. Fix solver/action scaling (fine-tuning)

**Approach:**
- Fix one issue at a time
- Re-test after each fix
- Document which fix solved the problem

---

## 🎓 Learning Resources

### Understanding the Codebase
- `CLAUDE.md` - Project architecture overview
- `deployment_docs/` - Previous deployment fixes
- `docs/` - Complete project documentation

### Key Files to Understand
- `src/colosseum/deploy/backends/mujoco.py` - Deployment controller
- `src/colosseum/tasks/velocity/deploy/t1_23dof/policy.py` - Policy inference
- `src/colosseum/robots/t1_23dof/deploy_config.py` - Robot configuration
- `external/mjlab/src/mjlab/viewer/base.py` - Working timing reference

### Comparison References
- Holosoma: `external/holosoma/` - Working ONNX deployment
- Play script: `external/mjlab/src/mjlab/scripts/play.py` - Working training playback

---

## 🐛 Troubleshooting

### Tests won't run
```bash
# Make sure to use pixi run python
pixi run python tests/test_sleep_timing.py

# NOT bare python (will fail with ModuleNotFoundError)
python tests/test_sleep_timing.py  # ✗ WRONG
```

### ImportError for colosseum modules
- Use `pixi run python` (not bare `python`)
- Check that you're in the project root directory

### Tests crash immediately
- Verify policy checkpoint exists
- Check ONNX model can be loaded
- Verify task registered: `pixi run python -m colosseum.deploy.core.registry`

### Want to see robot visually
- Tests run headless (no viewer)
- For visual: `pixi run python -m colosseum.deploy.backends.mujoco --task t1-velocity-flat`

### All tests show robot stable
- Increase test duration: edit `num_steps` in test files
- Add velocity commands (use joystick input)
- Try longer episodes in actual deployment

---

## ✅ Success Checklist

Before declaring the deployment fixed:

- [ ] All 4 diagnostic tests pass
- [ ] Robot stable for 100+ steps (not just 10)
- [ ] Robot responds correctly to velocity commands
- [ ] Robot maintains balance when perturbed
- [ ] Performance acceptable (50Hz policy frequency)
- [ ] Observations match training expectations
- [ ] ONNX/PyTorch inference equivalent
- [ ] Control loop timing optimized
- [ ] Configuration matches training

---

## 📞 Support

If you're stuck:

1. **Review documentation:**
   - Start with `TEST_SUITE_SUMMARY.md`
   - Check `ALL_POTENTIAL_ISSUES.md` for your specific issue

2. **Compare with working implementations:**
   - Play script: How does mjlab handle timing?
   - Holosoma: How do they export ONNX?

3. **Add more diagnostics:**
   - Use `quick_debug.py` for inline debugging
   - Add print statements to see intermediate values

4. **Simplify the problem:**
   - Test with dummy policy (return default_qpos)
   - Test with zero velocity commands
   - Test on flat terrain only

---

## 🎯 Next Steps

1. **Run the tests** (see Quick Start above)
2. **Review results** (see TEST_SUITE_SUMMARY.md)
3. **Fix issues** (in priority order)
4. **Re-test** (verify fixes work)
5. **Document** (for future debugging)

Good luck! 🚀🤖
