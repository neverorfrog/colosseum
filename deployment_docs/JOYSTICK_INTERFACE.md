# Backend-Agnostic Joystick Interface - Detailed Implementation Guide

## Overview

Implement a backend-agnostic input interface for colosseum deployment following the holosoma `command_sender` pattern. This provides unified joystick/keyboard control for both MuJoCo simulation and Booster robot deployment.

## Design Pattern (Following Holosoma)

```
BaseInputSource (abstract base class)
    ↓
create_input_source(config) - Factory with auto-detection
    ↓
JoystickInputSource | KeyboardInputSource (concrete implementations)
```

This mirrors holosoma's `BasicCommandSender` → `create_command_sender()` → `BoosterCommandSender`/`UnitreeCommandSender` pattern.

---

## STEP-BY-STEP IMPLEMENTATION GUIDE

### STEP 1: Create Input Configuration (`src/colosseum/deploy/input/config.py`)

**Purpose:** Define configuration dataclass for input sources

**Complete Code:**

```python
"""Input source configuration for deployment."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

try:
    import evdev
except ImportError:
    evdev = None  # Graceful degradation if evdev not installed


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

    # Logitech F710 default mapping
    x_axis: int = evdev.ecodes.ABS_Y if evdev else 1
    y_axis: int = evdev.ecodes.ABS_X if evdev else 0
    yaw_axis: int = evdev.ecodes.ABS_Z if evdev else 2
    custom_mode_button: int = evdev.ecodes.BTN_A if evdev else 304
    rl_gait_button: int = evdev.ecodes.BTN_B if evdev else 305

    @classmethod
    def logitech_preset(cls) -> InputConfig:
        """Logitech F710 gamepad preset."""
        if evdev is None:
            raise RuntimeError("evdev not installed")
        return cls(
            x_axis=evdev.ecodes.ABS_Y,
            y_axis=evdev.ecodes.ABS_X,
            yaw_axis=evdev.ecodes.ABS_Z,
            custom_mode_button=evdev.ecodes.BTN_A,
            rl_gait_button=evdev.ecodes.BTN_B,
        )

    @classmethod
    def xiaoji_preset(cls) -> InputConfig:
        """XiaoJi gamepad preset."""
        if evdev is None:
            raise RuntimeError("evdev not installed")
        return cls(
            x_axis=evdev.ecodes.ABS_Y,
            y_axis=evdev.ecodes.ABS_X,
            yaw_axis=evdev.ecodes.ABS_RX,  # Different axis
            custom_mode_button=evdev.ecodes.BTN_B,  # Swapped
            rl_gait_button=evdev.ecodes.BTN_A,
        )
```

**Why this design:**
- Frozen dataclass for immutability (matches existing config pattern)
- Graceful evdev import (allows config to load even without evdev)
- Preset factory methods for common controllers
- Default values for Logitech F710 (most common gamepad)

---

### STEP 2: Create Base Input Source (`src/colosseum/deploy/input/base.py`)

**Purpose:** Abstract base class defining the input interface contract

**Complete Code:**

```python
"""Abstract base class for input sources."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading
from typing import Optional

from .config import InputConfig


@dataclass
class InputState:
    """Container for normalized input state.

    All values normalized to [-1, 1] range for backend independence.
    Actual velocity limits applied by controller based on VelocityCommandConfig.
    """
    vx: float = 0.0   # Forward/backward velocity (normalized)
    vy: float = 0.0   # Lateral velocity (normalized)
    vyaw: float = 0.0  # Yaw rotation velocity (normalized)
    custom_mode_pressed: bool = False
    rl_gait_pressed: bool = False


class BaseInputSource(ABC):
    """Abstract base class for input sources.

    Provides thread-safe access to velocity commands and button states.
    Concrete implementations handle device-specific details.

    Similar to holosoma's BasicCommandSender pattern.
    """

    def __init__(self, config: InputConfig):
        """Initialize input source.

        Args:
            config: Input configuration
        """
        self.config = config
        self._lock = threading.Lock()
        self._running = True
        self._state = InputState()

        # Subclass initializes backend-specific components
        self._init_backend()

    @abstractmethod
    def _init_backend(self) -> None:
        """Initialize backend-specific components.

        Called during __init__ after base initialization.
        Should set up device connections, threads, etc.
        """
        pass

    @abstractmethod
    def get_operation_hint(self) -> str:
        """Get user-facing operation instructions.

        Returns:
            Human-readable string describing how to control the robot
        """
        pass

    def get_custom_mode_operation_hint(self) -> str:
        """Get custom mode activation instructions."""
        return "Press designated button to start custom mode."

    def get_rl_gait_operation_hint(self) -> str:
        """Get RL gait activation instructions."""
        return "Press designated button to start RL gait."

    # Thread-safe getters for velocity commands
    def get_vx_cmd(self) -> float:
        """Get forward/backward velocity command [-1, 1]."""
        with self._lock:
            return self._state.vx

    def get_vy_cmd(self) -> float:
        """Get lateral velocity command [-1, 1]."""
        with self._lock:
            return self._state.vy

    def get_vyaw_cmd(self) -> float:
        """Get yaw velocity command [-1, 1]."""
        with self._lock:
            return self._state.vyaw

    def get_state(self) -> InputState:
        """Get complete input state (atomic read)."""
        with self._lock:
            return InputState(
                vx=self._state.vx,
                vy=self._state.vy,
                vyaw=self._state.vyaw,
                custom_mode_pressed=self._state.custom_mode_pressed,
                rl_gait_pressed=self._state.rl_gait_pressed,
            )

    # Button state checks
    def start_custom_mode(self) -> bool:
        """Check if custom mode button is currently pressed."""
        with self._lock:
            return self._state.custom_mode_pressed

    def start_rl_gait(self) -> bool:
        """Check if RL gait button is currently pressed."""
        with self._lock:
            return self._state.rl_gait_pressed

    def close(self) -> None:
        """Clean up resources. Safe to call multiple times."""
        self._running = False

    # Context manager protocol
    def __enter__(self) -> BaseInputSource:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
```

**Why this design:**
- `InputState` dataclass: Clean separation of data from logic
- Thread-safe access: All getters use `_lock` for multi-threaded safety
- Normalized values [-1, 1]: Backend-independent, controller scales to actual limits
- Abstract `_init_backend()`: Template method pattern, subclasses customize
- Context manager: Ensures cleanup with `with` statement
- Matches holosoma's `BasicCommandSender` interface style

---

### STEP 3: Create Joystick Input Source (`src/colosseum/deploy/input/joystick.py`)

**Purpose:** evdev-based joystick backend with automatic device detection

**Complete Code:**

```python
"""Joystick input source using evdev."""
from __future__ import annotations
import threading
import time

try:
    import evdev
except ImportError:
    raise ImportError(
        "evdev is required for joystick support. Install with: pip install evdev"
    )

from .base import BaseInputSource, InputState
from .config import InputConfig


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
                abs_info = caps.get(evdev.ecodes.EV_ABS, [])
                axes = [code for (code, info) in abs_info]

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
                    self.axis_ranges = {code: info for code, info in abs_info}
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
```

**Why this design:**
- Automatic device detection: Searches `/dev/input/*` for compatible gamepad
- Axis range calibration: Uses device's `absinfo` for accurate normalization
- Dead-zone filtering: Prevents drift from resting stick position
- Dedicated thread: Low-latency event processing without blocking controller
- Error handling: Graceful degradation on device disconnect

---

### STEP 4: Create Keyboard Input Source (`src/colosseum/deploy/input/keyboard.py`)

**Purpose:** Terminal-based keyboard fallback with incremental control

**Complete Code:**

```python
"""Keyboard input source using terminal control."""
from __future__ import annotations
import atexit
import select
import sys
import termios
import threading
import tty

from .base import BaseInputSource
from .config import InputConfig


class KeyboardInputSource(BaseInputSource):
    """Keyboard input using terminal cbreak mode.

    Provides incremental velocity control via WASD + QE keys.
    Automatically saves/restores terminal state.
    """

    def _init_backend(self) -> None:
        """Initialize keyboard control."""
        self.keyboard_start_custom_mode = False
        self.keyboard_start_rl_gait = False

        # Check if stdin is a TTY
        try:
            if sys.stdin.isatty():
                self._stdin_tty = True
                self._old_termios = termios.tcgetattr(sys.stdin.fileno())
            else:
                self._stdin_tty = False
                self._old_termios = None
                print("Warning: stdin is not a TTY, keyboard input disabled")
                return
        except Exception:
            self._stdin_tty = False
            self._old_termios = None
            print("Warning: Failed to setup terminal, keyboard input disabled")
            return

        # Register cleanup on exit
        atexit.register(self.close)

        # Start keyboard listener thread
        self.listener_thread = threading.Thread(
            target=self._keyboard_listener,
            daemon=True
        )
        self.listener_thread.start()

    def _keyboard_listener(self) -> None:
        """Listen for keyboard input in cbreak mode."""
        if not self._stdin_tty:
            return

        fd = sys.stdin.fileno()
        try:
            tty.setcbreak(fd)
            while self._running:
                # Poll with timeout for clean shutdown
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch == "\x03":  # Ctrl-C
                        continue
                    elif ch == " ":
                        self._handle_key("space")
                    else:
                        self._handle_key(ch)
        finally:
            # Restore terminal settings
            if self._old_termios is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, self._old_termios)
                except Exception:
                    pass

    def _handle_key(self, key: str) -> None:
        """Handle keyboard press.

        Args:
            key: Single character or "space"
        """
        with self._lock:
            # Mode buttons
            if key == "x":
                self.keyboard_start_custom_mode = True
            elif key == "r":
                self.keyboard_start_rl_gait = True

            # Forward/backward (incremental)
            elif key == "w":
                old = self._state.vx
                self._state.vx = min(self._state.vx + 0.1, 1.0)
                print(f"VX: {old:.1f} => {self._state.vx:.1f}")
            elif key == "s":
                old = self._state.vx
                self._state.vx = max(self._state.vx - 0.1, -1.0)
                print(f"VX: {old:.1f} => {self._state.vx:.1f}")

            # Lateral (incremental)
            elif key == "a":
                old = self._state.vy
                self._state.vy = min(self._state.vy + 0.1, 1.0)
                print(f"VY: {old:.1f} => {self._state.vy:.1f}")
            elif key == "d":
                old = self._state.vy
                self._state.vy = max(self._state.vy - 0.1, -1.0)
                print(f"VY: {old:.1f} => {self._state.vy:.1f}")

            # Yaw (incremental)
            elif key == "q":
                old = self._state.vyaw
                self._state.vyaw = min(self._state.vyaw + 0.1, 1.0)
                print(f"VYaw: {old:.1f} => {self._state.vyaw:.1f}")
            elif key == "e":
                old = self._state.vyaw
                self._state.vyaw = max(self._state.vyaw - 0.1, -1.0)
                print(f"VYaw: {old:.1f} => {self._state.vyaw:.1f}")

            # Full stop
            elif key == "space":
                self._state.vx = 0.0
                self._state.vy = 0.0
                self._state.vyaw = 0.0
                print("FULL STOP")

    def start_custom_mode(self) -> bool:
        """Check keyboard custom mode trigger."""
        with self._lock:
            return self.keyboard_start_custom_mode

    def start_rl_gait(self) -> bool:
        """Check keyboard RL gait trigger."""
        with self._lock:
            return self.keyboard_start_rl_gait

    def get_operation_hint(self) -> str:
        """Get keyboard control instructions."""
        return (
            "Keyboard control: "
            "w/s (forward/back), a/d (left/right), q/e (yaw), "
            "Space (stop)"
        )

    def get_custom_mode_operation_hint(self) -> str:
        return "Press keyboard 'x' to start custom mode."

    def get_rl_gait_operation_hint(self) -> str:
        return "Press keyboard 'r' to start RL gait."

    def close(self) -> None:
        """Clean up terminal state."""
        super().close()
        # Restore terminal if we changed it
        if self._stdin_tty and self._old_termios is not None:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    self._old_termios
                )
            except Exception:
                pass
```

**Why this design:**
- Incremental control: Each keypress adjusts velocity by 0.1 (more intuitive than absolute)
- TTY detection: Gracefully handles non-TTY stdin (piped input, IDE consoles)
- Terminal restoration: Uses `atexit` + finally block to ensure cleanup
- Cbreak mode: Non-blocking input without requiring Enter key
- Visual feedback: Prints velocity changes for user awareness

---

### STEP 5: Create Factory Function (`src/colosseum/deploy/input/__init__.py`)

**Purpose:** Factory with automatic fallback (joystick → keyboard)

**Complete Code:**

```python
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
```

**Why this design:**
- Lazy imports: Only import backend when needed (evdev may not be installed)
- Automatic fallback: Default "auto" mode tries joystick → keyboard
- Explicit control: Users can force specific backend via config
- Error messages: Clear feedback on why fallback occurred
- Clean API: Single entry point for all input sources

---

### 3. BaseController Integration

**Modify `src/colosseum/deploy/core/base_controller.py`:**

1. Add field:
```python
self.input_source: BaseInputSource | None = None
```

2. In `__init__()`:
```python
if self.cfg.vel_command is not None:
    from colosseum.deploy.input import create_input_source
    self.input_source = create_input_source(self.cfg.input)
```

3. Add default implementation:
```python
def update_vel_command(self) -> None:
    """Default implementation using input_source."""
    if self.input_source is None or self.vel_command is None:
        return

    vx_norm = self.input_source.get_vx_cmd()
    vy_norm = self.input_source.get_vy_cmd()
    vyaw_norm = self.input_source.get_vyaw_cmd()

    self.vel_command.lin_vel_x = vx_norm * self.vel_command.vx_max
    self.vel_command.lin_vel_y = vy_norm * self.vel_command.vy_max
    self.vel_command.ang_vel_yaw = vyaw_norm * self.vel_command.vyaw_max
```

4. In `stop()`:
```python
if self.input_source is not None:
    self.input_source.close()
```

### 4. MujocoController Migration

**Modify `src/colosseum/deploy/backends/mujoco.py`:**

1. **REMOVE** `update_vel_command()` method (lines 64-84) - use base class implementation
2. **REMOVE** stdin parsing logic
3. In `run()` (line 136), add operation hint:
```python
if self.vel_command is not None and self.input_source is not None:
    print(f"\n{self.input_source.get_operation_hint()}")
```

### 5. BoosterRobotPortal Migration

**Modify `src/colosseum/deploy/backends/booster.py`:**

1. Replace RemoteControlService import (line 51):
```python
from colosseum.deploy.input import create_input_source
```

2. In `__init__()` (line 120):
```python
self.input_source = create_input_source(cfg.input)  # Replace RemoteControlService
```

3. In `_low_state_handler()` (lines 308-310):
```python
cmd[0]["vx"] = self.input_source.get_vx_cmd()
cmd[0]["vy"] = self.input_source.get_vy_cmd()
cmd[0]["vyaw"] = self.input_source.get_vyaw_cmd()
```

4. In `start_custom_mode_conditionally()` (line 352):
```python
if self.input_source.start_custom_mode():
```

5. In `start_rl_gait_conditionally()` (line 391):
```python
if self.input_source.start_rl_gait():
```

6. In `cleanup()` (line 440):
```python
self.input_source.close()
```

7. Update operation hints (lines 350, 389, 410):
```python
print(f"{self.input_source.get_custom_mode_operation_hint()}")
print(f"{self.input_source.get_rl_gait_operation_hint()}")
print(f"{self.input_source.get_operation_hint()}")
```

**Note:** `BoosterRobotController.update_vel_command()` remains unchanged - it reads from synced buffer.

### 6. Deprecate RemoteControlService

**Modify `src/colosseum/deploy/utils/remote_control_service.py`:**
```python
import warnings

class RemoteControlService:
    def __init__(self, config=None):
        warnings.warn(
            "RemoteControlService is deprecated. Use colosseum.deploy.input instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # ... existing code ...
```

## Implementation Order

1. **Create input module** (no breaking changes)
   - `src/colosseum/deploy/input/base.py`
   - `src/colosseum/deploy/input/config.py`
   - `src/colosseum/deploy/input/joystick.py`
   - `src/colosseum/deploy/input/keyboard.py`
   - `src/colosseum/deploy/input/__init__.py`

2. **Update configuration**
   - Add `InputConfig` to `src/colosseum/deploy/config/controller.py`
   - Export from `src/colosseum/deploy/config/__init__.py`

3. **Update BaseController**
   - Add `input_source` field
   - Add default `update_vel_command()` implementation
   - Cleanup in `stop()`

4. **Migrate MujocoController**
   - Remove stdin parsing
   - Add operation hint

5. **Migrate BoosterRobotPortal**
   - Replace RemoteControlService with create_input_source()
   - Update all call sites

6. **Add deprecation warning**
   - Update RemoteControlService

## Multi-Process Architecture Support

**BoosterRobotPortal** uses multi-process architecture:
- **Main process:** Input collection via `input_source` → writes to `synced_command` buffer
- **Inference process:** Reads from `synced_command` buffer (unchanged)

Input source lives only in main process, so inference process never directly accesses it.

## Usage Examples

### Auto-detection (Default)
```python
cfg = ControllerConfig(
    robot=T1_23DOF_DEPLOY_CFG,
    policy=PolicyConfig(task_name="velocity", checkpoint_path="model.onnx"),
    vel_command=VelocityCommandConfig(),
    # input=None means auto-detection
)
```

### Force keyboard
```python
cfg = ControllerConfig(
    robot=T1_23DOF_DEPLOY_CFG,
    policy=PolicyConfig(...),
    vel_command=VelocityCommandConfig(),
    input=InputConfig(input_type="keyboard"),
)
```

### Custom joystick mapping
```python
cfg = ControllerConfig(
    robot=T1_23DOF_DEPLOY_CFG,
    policy=PolicyConfig(...),
    vel_command=VelocityCommandConfig(),
    input=InputConfig(
        x_axis=evdev.ecodes.ABS_Y,
        y_axis=evdev.ecodes.ABS_X,
        yaw_axis=evdev.ecodes.ABS_RX,  # Different axis
    ),
)
```

## Backward Compatibility

- Existing code with `vel_command=None` continues to work (no input)
- Existing code with `vel_command=set, input=None` auto-detects (new behavior, no breaking change)
- RemoteControlService remains functional with deprecation warning

## Benefits

1. ✅ Backend-agnostic: Works with MuJoCo and Booster
2. ✅ Automatic fallback: Joystick → keyboard if unavailable
3. ✅ Type-safe: Pydantic frozen dataclasses
4. ✅ Thread-safe: Internal locking for concurrent access
5. ✅ Multi-process safe: IPC via SyncedArray in Booster
6. ✅ Extensible: Easy to add ROSInputSource, NetworkInputSource, etc.
7. ✅ Configuration-driven: All behavior via InputConfig
8. ✅ Backward compatible: Existing code works unchanged

## Critical Files

- `src/colosseum/deploy/input/base.py` - NEW
- `src/colosseum/deploy/input/joystick.py` - NEW
- `src/colosseum/deploy/input/keyboard.py` - NEW
- `src/colosseum/deploy/input/config.py` - NEW
- `src/colosseum/deploy/input/__init__.py` - NEW
- `src/colosseum/deploy/config/controller.py` - MODIFY
- `src/colosseum/deploy/core/base_controller.py` - MODIFY
- `src/colosseum/deploy/backends/mujoco.py` - MODIFY
- `src/colosseum/deploy/backends/booster.py` - MODIFY
- `src/colosseum/deploy/utils/remote_control_service.py` - DEPRECATE
