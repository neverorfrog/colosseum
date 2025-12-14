"""Input source factory and public API."""
from __future__ import annotations
from typing import Optional

from .base import BaseInputSource, InputState
from .config import InputConfig


def create_input_source(config: Optional[InputConfig] = None) -> BaseInputSource:
    """Create input source with automatic fallback.

    Factory function that creates the appropriate input source based on
    configuration. Follows holosoma's create_command_sender pattern.

    Args:
        config: Input configuration. If None, uses default (auto-detection).

    Returns:
        Configured input source

    Raises:
        RuntimeError: If explicit input_type is unavailable

    Example:
        >>> # Auto-detection (try joystick, fall back to keyboard)
        >>> source = create_input_source()
        >>>
        >>> # Force keyboard
        >>> source = create_input_source(InputConfig(input_type="keyboard"))
        >>>
        >>> # Custom joystick mapping
        >>> source = create_input_source(InputConfig.xiaoji_preset())
    """
    if config is None:
        config = InputConfig()

    if config.input_type == "auto":
        # Try joystick first, fall back to keyboard
        try:
            from .joystick import JoystickInputSource
            return JoystickInputSource(config)
        except Exception as e:
            print(f"Joystick unavailable ({e}), falling back to keyboard")
            from .keyboard import KeyboardInputSource
            return KeyboardInputSource(config)

    elif config.input_type == "joystick":
        # Explicit joystick (raises if unavailable)
        from .joystick import JoystickInputSource
        return JoystickInputSource(config)

    elif config.input_type == "keyboard":
        # Explicit keyboard
        from .keyboard import KeyboardInputSource
        return KeyboardInputSource(config)

    else:
        raise ValueError(f"Unknown input_type: {config.input_type}")


# Public API exports
__all__ = [
    "BaseInputSource",
    "InputConfig",
    "InputState",
    "create_input_source",
]