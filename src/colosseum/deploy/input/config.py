"""Input source configuration for deployment."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import evdev


@dataclass(frozen=True)
class InputConfig:
    """Configuration for input sources (joystick/keyboard).

    Attributes:
        input_type: Input backend selection. "auto" tries joystick first,
            falls back to keyboard if unavailable.
        control_threshold: Dead-zone threshold for analog inputs (0-1)
        x_axis: Joystick axis for forward/backward movement
        y_axis: Joystick axis for lateral movement
        yaw_axis: Joystick axis for yaw rotation
        custom_mode_button: Button code for custom mode trigger
        rl_gait_button: Button code for RL gait trigger
    """

    input_type: Literal["auto", "joystick", "keyboard"] = "auto"
    control_threshold: float = 0.1

    # Default mapping (left stick for translation, right stick X for yaw)
    # Left stick: ABS_Y (forward/backward), ABS_X (lateral)
    # Right stick: ABS_RX (yaw rotation)
    x_axis: int = evdev.ecodes.ABS_Y if evdev else 1
    y_axis: int = evdev.ecodes.ABS_X if evdev else 0
    yaw_axis: int = evdev.ecodes.ABS_RX if evdev else 3
    custom_mode_button: int = evdev.ecodes.BTN_A if evdev else 304
    rl_gait_button: int = evdev.ecodes.BTN_B if evdev else 305

    @classmethod
    def logitech_preset(cls) -> InputConfig:
        """Logitech F710 gamepad preset with standard mapping."""
        if evdev is None:
            raise ImportError("evdev is required for joystick presets. Please install evdev.")
        return cls(
            x_axis=evdev.ecodes.ABS_Y,      # Left stick up/down
            y_axis=evdev.ecodes.ABS_X,      # Left stick left/right
            yaw_axis=evdev.ecodes.ABS_RX,   # Right stick left/right (fixed!)
            custom_mode_button=evdev.ecodes.BTN_A,
            rl_gait_button=evdev.ecodes.BTN_B,
        )

    @classmethod
    def xiaoji_preset(cls) -> InputConfig:
        """XiaoJi gamepad preset."""
        if evdev is None:
            raise ImportError("evdev is required for joystick presets. Please install evdev.")
        return cls(
            x_axis=evdev.ecodes.ABS_Y,
            y_axis=evdev.ecodes.ABS_X,
            yaw_axis=evdev.ecodes.ABS_RX,  # Different axis
            custom_mode_button=evdev.ecodes.BTN_B,  # Swapped
            rl_gait_button=evdev.ecodes.BTN_A,
        )