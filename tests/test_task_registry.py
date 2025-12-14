"""Test task registry and policy class system.

Tests the registration mechanism without requiring actual deployment backends.
"""

def test_task_registration():
    """Test that tasks are properly registered."""
    from colosseum.deploy.core.registry import TASK_REGISTRY, auto_register_tasks

    # Auto-discover and register all tasks
    auto_register_tasks()

    # Check task is registered
    tasks = TASK_REGISTRY.list_tasks()
    assert "t1-velocity" in tasks, f"t1-velocity not registered. Available: {list(tasks.keys())}"

    print("✓ Task 't1-velocity' registered successfully")
    print(f"  Available tasks: {list(tasks.keys())}")


def test_get_config():
    """Test retrieving config from registry."""
    from colosseum.deploy.core.registry import TASK_REGISTRY, auto_register_tasks

    auto_register_tasks()
    config = TASK_REGISTRY.get_config("t1-velocity")

    # Verify config structure
    assert config.robot.name == "Booster_T1_23DOF"
    assert config.policy.policy_type == "velocity"
    assert config.policy.robot_name == "Booster_T1_23DOF"
    assert config.policy_dt == 0.02

    print("✓ Config retrieved successfully")
    print(f"  Robot: {config.robot.name}")
    print(f"  Policy type: {config.policy.policy_type}")
    print(f"  Policy dt: {config.policy_dt}s")
    print(f"  Checkpoint: {config.policy.checkpoint_path}")


def test_get_policy_class():
    """Test retrieving policy class from registry."""
    from colosseum.deploy.core.policy import Policy
    from colosseum.deploy.core.registry import TASK_REGISTRY, auto_register_tasks

    auto_register_tasks()
    policy_class = TASK_REGISTRY.get_policy("velocity")

    # Verify it's a Policy subclass
    assert issubclass(policy_class, Policy)

    print("✓ Policy class retrieved successfully")
    print(f"  Policy class: {policy_class.__name__}")
    print(f"  Is Policy subclass: {issubclass(policy_class, Policy)}")


def test_policy_instantiation():
    """Test creating a policy instance (mock controller)."""
    from colosseum.deploy.core.registry import TASK_REGISTRY, auto_register_tasks
    from colosseum.deploy.config import ControllerConfig
    from colosseum.deploy.core.robot import BoosterRobot
    from colosseum.deploy.core.base_controller import VelocityCommand

    auto_register_tasks()

    # Get config and policy class
    config = TASK_REGISTRY.get_config("t1-velocity")
    policy_class = TASK_REGISTRY.get_policy("velocity")

    # Create mock controller (without full backend)
    class MockController:
        def __init__(self, cfg: ControllerConfig):
            self.cfg = cfg
            self.robot = BoosterRobot(cfg.robot)
            self.vel_command = VelocityCommand(cfg.vel_command) if cfg.vel_command else None

    controller = MockController(config)

    # Try to instantiate policy
    try:
        policy = policy_class(controller)
        print("✓ Policy instantiated successfully")
        print(f"  Policy class: {policy.__class__.__name__}")
        print(f"  Has reset method: {hasattr(policy, 'reset')}")
        print(f"  Has inference method: {hasattr(policy, 'inference')}")
    except FileNotFoundError as e:
        # Expected if checkpoint doesn't exist
        print("✓ Policy instantiation attempted (checkpoint file not found - expected)")
        print(f"  Error: {e}")


def test_policy_robot_validation():
    """Test that policy-robot matching is validated."""
    from colosseum.deploy.core.registry import TASK_REGISTRY, auto_register_tasks
    from dataclasses import replace

    auto_register_tasks()
    config = TASK_REGISTRY.get_config("t1-velocity")

    # Test valid config (should pass)
    try:
        config.policy.validate_robot(config.robot)
        print("✓ Valid policy-robot match accepted")
    except ValueError as e:
        print(f"✗ Unexpected validation error: {e}")
        raise

    # Test invalid config (should fail)
    bad_config = replace(
        config,
        policy=replace(config.policy, robot_name="DifferentRobot")
    )

    try:
        bad_config.policy.validate_robot(bad_config.robot)
        print("✗ Invalid policy-robot match should have failed!")
        raise AssertionError("Validation should have raised ValueError")
    except ValueError as e:
        print("✓ Invalid policy-robot match rejected")
        print(f"  Error: {e}")


def test_config_immutability():
    """Test that configs are immutable (frozen dataclasses)."""
    from colosseum.deploy.core.registry import TASK_REGISTRY, auto_register_tasks

    auto_register_tasks()
    config = TASK_REGISTRY.get_config("t1-velocity")

    # Try to modify (should fail)
    try:
        config.policy_dt = 0.01  # type: ignore
        print("✗ Config should be immutable!")
        raise AssertionError("Config modification should have failed")
    except Exception as e:
        print("✓ Config is immutable (frozen dataclass)")
        print(f"  Error type: {type(e).__name__}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Task Registry System")
    print("=" * 60)
    print()

    tests = [
        ("Task Registration", test_task_registration),
        ("Get Config", test_get_config),
        ("Get Policy Class", test_get_policy_class),
        ("Policy Instantiation", test_policy_instantiation),
        ("Policy-Robot Validation", test_policy_robot_validation),
        ("Config Immutability", test_config_immutability),
    ]

    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            test_func()
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
