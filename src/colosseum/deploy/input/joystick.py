"""Joystick input source using evdev."""
from __future__ import annotations
import threading
import time
from typing import Any, cast
import evdev
from .base import BaseInputSource


class JoystickInputSource(BaseInputSource):
    """Joystick input using evdev library.

    Automatically detects compatible gamepad devices and polls events
    in a dedicated thread for low-latency input.
    """

    def _init_backend(self) -> None:
        """Initialize joystick device and start polling thread."""
        # Find compatible joystick device
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        joystick = None

        for device in devices:
            caps = device.capabilities()
            # Check for both absolute axes and keys (typical gamepad)
            if evdev.ecodes.EV_ABS in caps and evdev.ecodes.EV_KEY in caps:
                abs_axes = cast(list[tuple[int, Any]], caps.get(evdev.ecodes.EV_ABS, []))
                axes = [code for code, info in abs_axes]

                # Verify required axes present
                if all(
                    code in axes
                    for code in [
                        self.config.x_axis,
                        self.config.y_axis,
                        self.config.yaw_axis,
                    ]
                ):
                    # Store axis ranges for normalization
                    self.axis_ranges = {code: info for code, info in abs_axes}
                    print(f"Found suitable joystick: {device.name}")
                    joystick = device
                    break

        if not joystick:
            raise RuntimeError(
                "No suitable joystick found. "
                "Ensure gamepad is connected and has required axes."
            )

        self.joystick = joystick
        print(f"Selected joystick: {joystick.name}")

        # Start polling thread
        self.poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.poll_thread.start()

    def _poll_loop(self) -> None:
        """Poll joystick events in dedicated thread."""
        while self._running:
            try:
                event = self.joystick.read_one()
                if event:
                    if event.type == evdev.ecodes.EV_ABS:
                        self._handle_axis(event.code, event.value)
                    elif event.type == evdev.ecodes.EV_KEY:
                        self._handle_button(event.code, event.value)
                else:
                    time.sleep(0.001)  # Small sleep when no events
            except Exception as e:
                if not self._running:
                    break  # Expected during shutdown
                print(f"Error in joystick polling: {e}")
                time.sleep(0.01)

    def _handle_axis(self, code: int, value: int) -> None:
        """Handle axis event and update state."""
        # Normalize value to [-1, 1] using device's axis range
        normalized = self._normalize_axis(value, code)

        # Apply dead-zone
        if abs(normalized) < self.config.control_threshold:
            normalized = 0.0

        # Update appropriate state field
        with self._lock:
            if code == self.config.x_axis:
                self._state.vx = -normalized  # Inverted for forward=positive
            elif code == self.config.y_axis:
                self._state.vy = -normalized  # Inverted for left=positive
            elif code == self.config.yaw_axis:
                self._state.vyaw = -normalized

    def _normalize_axis(self, value: int, axis_code: int) -> float:
        """Normalize hardware axis value to [-1, 1] range.

        Args:
            value: Raw axis value from device
            axis_code: evdev axis code

        Returns:
            Normalized value in [-1, 1]
        """
        absinfo = self.axis_ranges[axis_code]
        min_val = absinfo.min
        max_val = absinfo.max

        # Linear mapping: [min, max] → [-1, 1]
        return ((value - min_val) / (max_val - min_val)) * 2 - 1

    def _handle_button(self, code: int, value: int) -> None:
        """Handle button event and update state.

        Args:
            code: Button code
            value: 1=pressed, 0=released
        """
        with self._lock:
            if code == self.config.custom_mode_button:
                self._state.custom_mode_pressed = (value == 1)
            elif code == self.config.rl_gait_button:
                self._state.rl_gait_pressed = (value == 1)

    def get_operation_hint(self) -> str:
        """Get joystick control instructions."""
        return (
            "Joystick control: "
            "Left stick for forward/backward/lateral, "
            "Right stick for yaw rotation"
        )

    def get_custom_mode_operation_hint(self) -> str:
        return "Press joystick button A to start custom mode."

    def get_rl_gait_operation_hint(self) -> str:
        return "Press joystick button B to start RL gait."

    def close(self) -> None:
        """Clean up joystick resources."""
        super().close()
        if hasattr(self, "joystick"):
            try:
                self.joystick.close()
            except Exception as e:
                print(f"Error closing joystick: {e}")
        if hasattr(self, "poll_thread"):
            self.poll_thread.join(timeout=1.0)