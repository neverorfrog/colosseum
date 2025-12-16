"""Test ONNX model consistency and correctness."""

import torch
import numpy as np
from pathlib import Path
from colosseum.deploy.core.registry import TASK_REGISTRY, auto_register_tasks
from colosseum.deploy.backends.mujoco import MujocoController

auto_register_tasks()


def test_onnx_model_loading():
    """Test ONNX model loads correctly."""
    print("\n" + "=" * 80)
    print("TEST: ONNX Model Loading")
    print("=" * 80)

    cfg = TASK_REGISTRY.get_config("t1-velocity-flat")
    controller = MujocoController(cfg)
    controller.update_state()
    controller.start()

    policy = controller.policy
    model = policy._model

    print(f"\nModel type: {type(model)}")
    print(f"Is ONNX wrapper: {hasattr(model, 'session')}")

    if hasattr(model, 'session'):
        session = model.session
        print(f"\nONNX Runtime Session:")
        print(f"  Providers: {session.get_providers()}")
        print(f"  Inputs:")
        for inp in session.get_inputs():
            print(f"    - {inp.name}: shape={inp.shape}, dtype={inp.type}")
        print(f"  Outputs:")
        for out in session.get_outputs():
            print(f"    - {out.name}: shape={out.shape}, dtype={out.type}")

        print(f"\n✓ ONNX model loaded successfully")
        return True
    else:
        print(f"\n⚠️  Model is not ONNX (might be TorchScript)")
        return False


def test_onnx_metadata():
    """Check ONNX model metadata."""
    print("\n" + "=" * 80)
    print("TEST: ONNX Model Metadata")
    print("=" * 80)

    try:
        import onnx
    except ImportError:
        print("⚠️  'onnx' package not installed, skipping metadata check")
        return False

    # Find ONNX model path
    cfg = TASK_REGISTRY.get_config("t1-velocity-flat")
    onnx_path = Path(cfg.policy.checkpoint_path)

    if not onnx_path.exists():
        print(f"⚠️  ONNX file not found: {onnx_path}")
        return False

    print(f"\nLoading ONNX model: {onnx_path}")
    model = onnx.load(str(onnx_path))

    print(f"\nONNX Model Info:")
    print(f"  IR Version: {model.ir_version}")
    print(f"  Opset Version: {model.opset_import[0].version if model.opset_import else 'N/A'}")
    print(f"  Producer: {model.producer_name}")
    print(f"  Model Version: {model.model_version}")

    print(f"\nInputs:")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: shape={shape}, dtype={inp.type.tensor_type.elem_type}")

    print(f"\nOutputs:")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: shape={shape}, dtype={out.type.tensor_type.elem_type}")

    print(f"\nMetadata Properties:")
    if model.metadata_props:
        for prop in model.metadata_props:
            print(f"  {prop.key}: {prop.value}")
    else:
        print(f"  (no metadata)")

    print(f"\n✓ ONNX metadata loaded")
    return True


def test_onnx_inference_determinism():
    """Test ONNX inference is deterministic."""
    print("\n" + "=" * 80)
    print("TEST: ONNX Inference Determinism")
    print("=" * 80)

    cfg = TASK_REGISTRY.get_config("t1-velocity-flat")
    controller = MujocoController(cfg)
    controller.update_state()
    controller.start()

    policy = controller.policy

    # Create deterministic test input
    torch.manual_seed(42)
    test_obs = torch.randn(1, 81)  # 81-dim velocity observation for T1 23-DOF

    # Run inference 3 times
    print(f"\nRunning inference 3 times on same input...")
    outputs = []
    for i in range(3):
        output = policy._model(test_obs)
        outputs.append(output)
        print(f"  Run {i+1}: mean={output.mean():.6f}, std={output.std():.6f}, "
              f"range=[{output.min():.6f}, {output.max():.6f}]")

    # Check if outputs are identical
    diff_01 = (outputs[0] - outputs[1]).abs().max().item()
    diff_12 = (outputs[1] - outputs[2]).abs().max().item()
    diff_02 = (outputs[0] - outputs[2]).abs().max().item()

    print(f"\nDifferences between runs:")
    print(f"  Run 0 vs Run 1: max_diff = {diff_01:.10f}")
    print(f"  Run 1 vs Run 2: max_diff = {diff_12:.10f}")
    print(f"  Run 0 vs Run 2: max_diff = {diff_02:.10f}")

    if diff_01 == 0 and diff_12 == 0 and diff_02 == 0:
        print(f"\n✓ ONNX inference is DETERMINISTIC (identical outputs)")
        return True
    elif max(diff_01, diff_12, diff_02) < 1e-6:
        print(f"\n✓ ONNX inference is DETERMINISTIC (negligible differences)")
        return True
    else:
        print(f"\n⚠️  ONNX inference has non-negligible differences")
        print(f"    This might indicate numerical instability")
        return False


def test_onnx_output_range():
    """Test ONNX output is in expected range."""
    print("\n" + "=" * 80)
    print("TEST: ONNX Output Range Check")
    print("=" * 80)

    cfg = TASK_REGISTRY.get_config("t1-velocity-flat")
    controller = MujocoController(cfg)
    controller.update_state()
    controller.start()

    policy = controller.policy

    # Test with various inputs
    print(f"\nTesting with different observation patterns...")

    test_cases = [
        ("Zeros", torch.zeros(1, 81)),
        ("Ones", torch.ones(1, 81)),
        ("Random normal", torch.randn(1, 81)),
        ("Random uniform [-1,1]", 2 * torch.rand(1, 81) - 1),
        ("Large positive", torch.full((1, 81), 5.0)),
        ("Large negative", torch.full((1, 81), -5.0)),
    ]

    all_reasonable = True

    print(f"\n{'Test Case':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Status':>10}")
    print("-" * 90)

    for name, test_obs in test_cases:
        output = policy._model(test_obs)

        mean = output.mean().item()
        std = output.std().item()
        min_val = output.min().item()
        max_val = output.max().item()

        # Expected: normalized actions roughly in [-1, 1]
        reasonable = (-3.0 < min_val < 3.0) and (-3.0 < max_val < 3.0)
        status = "✓ OK" if reasonable else "⚠️  WARN"

        print(f"{name:<25} {mean:>10.4f} {std:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {status:>10}")

        if not reasonable:
            all_reasonable = False

    if all_reasonable:
        print(f"\n✓ All outputs in reasonable range")
    else:
        print(f"\n⚠️  Some outputs outside expected range (might be clipped)")

    return all_reasonable


def test_onnx_gradient_test():
    """Test ONNX model responds to input changes (sanity check)."""
    print("\n" + "=" * 80)
    print("TEST: ONNX Sensitivity to Input Changes")
    print("=" * 80)

    cfg = TASK_REGISTRY.get_config("t1-velocity-flat")
    controller = MujocoController(cfg)
    controller.update_state()
    controller.start()

    policy = controller.policy

    # Base observation
    torch.manual_seed(42)
    base_obs = torch.randn(1, 81)
    base_output = policy._model(base_obs)

    # Test sensitivity to each observation component
    print(f"\nTesting sensitivity to perturbations...")

    perturbations = [
        ("Small perturbation (0.1)", base_obs + 0.1),
        ("Medium perturbation (1.0)", base_obs + 1.0),
        ("Different random obs", torch.randn(1, 81)),
    ]

    print(f"\n{'Perturbation':<30} {'Max Output Diff':>20} {'Status':>10}")
    print("-" * 80)

    all_responsive = True

    for name, perturbed_obs in perturbations:
        perturbed_output = policy._model(perturbed_obs)
        diff = (perturbed_output - base_output).abs().max().item()

        # Model should respond to input changes
        responsive = diff > 1e-6
        status = "✓ OK" if responsive else "⚠️  WARN"

        print(f"{name:<30} {diff:>20.6f} {status:>10}")

        if not responsive:
            all_responsive = False

    if all_responsive:
        print(f"\n✓ Model is RESPONSIVE to input changes")
    else:
        print(f"\n⚠️  Model is NOT responding to inputs (might be frozen or broken)")

    return all_responsive


def test_onnx_vs_actual_observations():
    """Test ONNX inference on actual robot observations."""
    print("\n" + "=" * 80)
    print("TEST: ONNX Inference on Real Observations")
    print("=" * 80)

    cfg = TASK_REGISTRY.get_config("t1-velocity-flat")
    controller = MujocoController(cfg)
    controller.update_state()
    controller.start()

    # Run a few steps
    print(f"\nRunning 10 control steps...")

    outputs = []
    for i in range(10):
        controller.update_state()
        obs = controller.policy.compute_observation()
        action = controller.policy._model(obs).flatten()
        outputs.append(action)

        if i % 2 == 0:
            print(f"  Step {i:2d}: action mean={action.mean():.4f}, std={action.std():.4f}, "
                  f"range=[{action.min():.4f}, {action.max():.4f}]")

    # Check statistics
    actions = torch.stack(outputs)
    mean_action = actions.mean(dim=0)
    std_action = actions.std(dim=0)

    print(f"\nAction statistics over 10 steps:")
    print(f"  Mean action: mean={mean_action.mean():.4f}, std={mean_action.std():.4f}")
    print(f"  Action variability: mean={std_action.mean():.4f}, max={std_action.max():.4f}")

    # Actions should vary (not constant)
    if std_action.max() < 1e-6:
        print(f"\n⚠️  Actions are CONSTANT (model might be frozen or observations wrong)")
        return False
    else:
        print(f"\n✓ Actions vary over time (model is working)")
        return True


def main():
    """Run all ONNX consistency tests."""
    print("\n" + "=" * 80)
    print("ONNX CONSISTENCY TEST SUITE")
    print("=" * 80)

    results = {}

    try:
        results["loading"] = test_onnx_model_loading()
    except Exception as e:
        print(f"\n✗ Loading test failed: {e}")
        results["loading"] = False

    try:
        results["metadata"] = test_onnx_metadata()
    except Exception as e:
        print(f"\n✗ Metadata test failed: {e}")
        results["metadata"] = False

    try:
        results["determinism"] = test_onnx_inference_determinism()
    except Exception as e:
        print(f"\n✗ Determinism test failed: {e}")
        results["determinism"] = False

    try:
        results["output_range"] = test_onnx_output_range()
    except Exception as e:
        print(f"\n✗ Output range test failed: {e}")
        results["output_range"] = False

    try:
        results["responsiveness"] = test_onnx_gradient_test()
    except Exception as e:
        print(f"\n✗ Responsiveness test failed: {e}")
        results["responsiveness"] = False

    try:
        results["real_observations"] = test_onnx_vs_actual_observations()
    except Exception as e:
        print(f"\n✗ Real observations test failed: {e}")
        results["real_observations"] = False

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "⚠️  FAIL"
        print(f"{name:<30} {status}")

    all_passed = all(results.values())

    if all_passed:
        print(f"\n✓ All ONNX tests PASSED")
        print(f"  ONNX model is likely correct!")
    else:
        print(f"\n⚠️  Some ONNX tests FAILED")
        print(f"  Check the failures above for ONNX conversion issues")


if __name__ == "__main__":
    main()
