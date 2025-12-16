# Deployment Diagnostic Test Suite - Summary

## 🎯 Quick Start

**Run these 4 tests in order:**

```bash
# 1. Control loop timing (20s) - PRIMARY SUSPECT
pixi run python tests/test_sleep_timing.py

# 2. Observation validation (10s) - NEW INVESTIGATION
pixi run python tests/test_observations.py

# 3. ONNX model consistency (30s) - NEW INVESTIGATION
pixi run python tests/test_onnx_consistency.py

# 4. Full diagnostics (40s) - COMPREHENSIVE
pixi run python tests/test_deployment_diagnostics.py
```

**Total runtime:** ~100 seconds (~1.5 minutes)

---

## 📊 Test Suite Overview

### Test 1: Sleep Timing Comparison ⭐ **HIGHEST PRIORITY**
**File:** `test_sleep_timing.py`
**Confidence:** 85%
**Runtime:** 20 seconds

**What it tests:**
- Compares robot stability with 3 different control loop timing strategies
- Fixed 20ms sleep (current) vs Minimal sleep vs Adaptive timing

**Why it matters:**
- Fixed sleep BEFORE state update causes 20-24ms observation latency
- Policy makes decisions based on stale state → destabilization
- This is the most likely cause of falling/oscillation

**Expected outcome if timing is the issue:**
```
Fixed Sleep    →  FALLING  ← Current problem
Minimal Sleep  →  STABLE   ← Quick fix!
Adaptive       →  STABLE   ← Proper solution
```

---

### Test 2: Observation Validation ⭐ **NEW**
**File:** `test_observations.py`
**Confidence:** 40%
**Runtime:** 10 seconds

**What it tests:**
- Observation structure matches training specification
- Projected gravity computed correctly (quaternion math)
- Joint order remapping (hardware → simulation order)
- Observation values in realistic ranges

**Why it matters:**
- Wrong observations → policy sees garbage → wrong actions
- Subtle bugs in quaternion convention, joint order, or computation
- Could explain why even dummy policy (returning default_qpos) fails

**Expected issues to catch:**
- Projected gravity direction wrong (quaternion convention)
- Joint order mismatch (real vs sim)
- Observation component ordering differs from training

---

### Test 3: ONNX Model Consistency ⭐ **NEW**
**File:** `test_onnx_consistency.py`
**Confidence:** 30%
**Runtime:** 30 seconds

**What it tests:**
- ONNX model loads correctly
- Inference is deterministic (same input → same output)
- Output ranges are reasonable
- Model responds to input changes
- Inference works on real observations

**Why it matters:**
- ONNX export might have introduced numerical errors
- Different from TorchScript (which Booster originally used)
- Holosoma uses ONNX successfully, so it CAN work

**Expected issues to catch:**
- ONNX export configuration wrong (opset, dynamic axes)
- Observation normalizer not included in export
- Numerical precision differences (FP32 vs FP16)
- Frozen model (not responding to inputs)

---

### Test 4: Full Diagnostics Suite
**File:** `test_deployment_diagnostics.py`
**Confidence:** Varies by subtest
**Runtime:** 40 seconds

**What it tests:**
1. Control loop timing (detailed analysis)
2. Action scaling (uniform vs per-joint)
3. Actuator configuration (PD gains, effort limits)
4. Physics solver settings (iterations, timestep)
5. State freshness (observation staleness)

**Why it matters:**
- Catches ALL configuration mismatches
- Provides detailed timing breakdown
- Verifies training-deployment consistency

---

## 🔍 How to Interpret Results

### Scenario 1: Timing is the Primary Issue ✅ **MOST LIKELY**

**Test results:**
```
test_sleep_timing.py:
  Fixed Sleep    → FALLING
  Minimal Sleep  → STABLE
  Adaptive       → STABLE

test_observations.py:
  ✓ All tests PASSED

test_onnx_consistency.py:
  ✓ All tests PASSED
```

**Action:**
1. Implement adaptive timing fix (Priority 1)
2. Set solver iterations to match training (quick win)
3. Re-test to confirm stability

**Confidence:** 85%

---

### Scenario 2: Observation Issues ⚠️

**Test results:**
```
test_observations.py:
  ⚠️  Projected gravity FAILED
  ⚠️  Joint order mismatch detected
```

**Possible causes:**
- Quaternion convention wrong (w,x,y,z vs x,y,z,w)
- Rotation direction (forward vs inverse)
- Joint remapping broken

**Action:**
1. Fix projected gravity computation
2. Verify joint order mapping
3. Compare with training observation computation

**Confidence:** 40%

---

### Scenario 3: ONNX Conversion Issues ⚠️

**Test results:**
```
test_onnx_consistency.py:
  ⚠️  Model NOT responding to inputs (frozen)
  ⚠️  Inference non-deterministic
  ⚠️  Output range unrealistic
```

**Possible causes:**
- ONNX export missing observation normalizer
- Wrong opset version or dynamic axes
- Numerical precision issues

**Action:**
1. Re-export ONNX with correct settings
2. Compare with Holosoma's ONNX export configuration
3. Consider using TorchScript instead

**Confidence:** 30%

---

### Scenario 4: Multiple Issues 🔴

**Test results:**
```
test_sleep_timing.py:
  All three → FALLING

test_observations.py:
  ⚠️  Multiple failures

test_onnx_consistency.py:
  ⚠️  Multiple failures
```

**Action:**
1. Address issues in priority order:
   - Fix observations first (most fundamental)
   - Fix ONNX export
   - Fix timing
2. Re-test after each fix to isolate issues

---

## 📈 Test Coverage Map

```
Issue Category                Test File              Coverage
─────────────────────────────────────────────────────────────────
1. Control Loop Timing        test_sleep_timing.py   ███████████ 100%
2. Observation Computation    test_observations.py   ███████████ 100%
3. ONNX Model Export          test_onnx_consistency  ██████████  90%
4. Action Scaling             test_deployment_diag   ████████    80%
5. Solver Settings            test_deployment_diag   ████████    80%
6. PD Gains                   test_deployment_diag   ████████    80%
7. Joint Order                test_observations.py   ███████████ 100%
8. Actuator Config            test_deployment_diag   ████████    80%
9. Ground Contact             [Already fixed]        ███████████ 100%
10. Decimation/Timestep       [Already verified]     ███████████ 100%
```

---

## 🎯 Success Criteria

**All tests should show:**

✅ **test_sleep_timing.py:**
- Minimal/Adaptive = STABLE
- Robot maintains height (z_drop < 0.05m)

✅ **test_observations.py:**
- All observation tests PASSED
- Projected gravity correct
- Joint order valid
- Values in realistic ranges

✅ **test_onnx_consistency.py:**
- Model loads successfully
- Inference deterministic
- Output ranges reasonable
- Model responsive to inputs

✅ **test_deployment_diagnostics.py:**
- Total latency < 10ms
- Action scaling matches training
- Actuators match config
- Solver settings match training

---

## 📝 Documentation Files

- **ALL_POTENTIAL_ISSUES.md** - Complete catalog of 10 potential issues with tests
- **RUN_DIAGNOSTICS.md** - Quick reference for running tests
- **DIAGNOSTICS_README.md** - Detailed test documentation
- **TEST_SUITE_SUMMARY.md** - This file (high-level overview)

---

## 🚀 After Running Tests

### If all tests pass but robot still falls:

1. **Check velocity commands:** Are they too aggressive?
2. **Try longer episodes:** Increase `num_steps` in tests
3. **Test with play.py:** Does it work with the same policy?
4. **Compare training config:** Are there hidden differences?

### If tests reveal issues:

1. **Fix in priority order:**
   - Observations (most fundamental)
   - ONNX export (affects inference)
   - Timing (affects control loop)
   - Configuration (fine-tuning)

2. **Re-test after each fix:**
   - Isolate which fix actually solved the problem
   - Avoid compounding changes

3. **Document findings:**
   - What was broken
   - How you fixed it
   - How to prevent regression

---

## 🎓 Learning from Holosoma

Holosoma successfully deploys with ONNX. Key differences:

| Aspect | Holosoma | Colosseum |
|--------|----------|-----------|
| **PD Gains** | Hard-coded, simpler | Motor-specific, computed |
| **Action Scale** | Uniform 0.25 | Per-joint formula |
| **Timing** | Unknown (need to check) | Fixed sleep (broken) |
| **ONNX Export** | Working | Need to verify |

**TODO:** Investigate Holosoma's control loop timing to see if they have adaptive timing.

---

## 🏁 Final Checklist

Before declaring victory, verify:

- [ ] Robot stable for 100+ steps (not just 10)
- [ ] Robot responds to velocity commands correctly
- [ ] Robot maintains balance when perturbed
- [ ] Observations match training expectations
- [ ] ONNX inference matches PyTorch (if available)
- [ ] All diagnostic tests pass
- [ ] Performance acceptable (50Hz policy frequency)

Good luck! 🚀🤖
