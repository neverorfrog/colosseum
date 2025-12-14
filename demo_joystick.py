#!/usr/bin/env python3
"""Demo script to test JoystickInputSource with a real joystick.

This script connects to a joystick and prints:
- Connection status
- Button presses (custom mode and RL gait)
- Velocity commands (forward, lateral, yaw)

Usage:
    python demo_joystick.py

Press Ctrl+C to exit.
"""
import sys
import time
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from colosseum.deploy.input.joystick import JoystickInputSource
from colosseum.deploy.input.config import InputConfig


def main():
    """Main demo function."""
    print("=" * 60)
    print("Joystick Input Source Demo")
    print("=" * 60)
    print()

    # Create config (using default Logitech F710 mapping)
    try:
        config = InputConfig.logitech_preset()
        print(f"✓ Joystick config created (Logitech F710 preset)")
        print(f"  - Control threshold: {config.control_threshold}")
    except ImportError as e:
        print(f"✗ Error creating config: {e}")
        return 1

    # Initialize joystick
    try:
        joystick = JoystickInputSource(config)
        print(f"✓ Joystick connected successfully!")
        print(f"  {joystick.get_operation_hint()}")
        print()
    except RuntimeError as e:
        print(f"✗ Error connecting to joystick: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Ensure joystick is plugged in")
        print("  2. Try: ls /dev/input/event*")
        print("  3. Verify gamepad has required axes (X, Y, Z)")
        return 1

    # Track previous state to detect changes
    prev_state = joystick.get_state()
    button_pressed = {"custom_mode": False, "rl_gait": False}

    print("Listening for input... (Press Ctrl+C to exit)")
    print("-" * 60)
    print()

    try:
        while True:
            current_state = joystick.get_state()
            changed = False

            # Check velocity commands
            if (
                abs(current_state.vx - prev_state.vx) > 0.01
                or abs(current_state.vy - prev_state.vy) > 0.01
                or abs(current_state.vyaw - prev_state.vyaw) > 0.01
            ):
                print(
                    f"[MOVEMENT] vx={current_state.vx:6.3f}  "
                    f"vy={current_state.vy:6.3f}  "
                    f"vyaw={current_state.vyaw:6.3f}"
                )
                changed = True

            # Check custom mode button
            if current_state.custom_mode_pressed != prev_state.custom_mode_pressed:
                state_str = "PRESSED" if current_state.custom_mode_pressed else "RELEASED"
                print(f"[BUTTON] Custom Mode: {state_str}")
                changed = True

            # Check RL gait button
            if current_state.rl_gait_pressed != prev_state.rl_gait_pressed:
                state_str = "PRESSED" if current_state.rl_gait_pressed else "RELEASED"
                print(f"[BUTTON] RL Gait: {state_str}")
                changed = True

            if changed:
                print()

            prev_state = current_state
            time.sleep(0.05)  # 20 Hz polling

    except KeyboardInterrupt:
        print()
        print("-" * 60)
        print("Shutting down...")

    finally:
        joystick.close()
        print("✓ Joystick connection closed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
