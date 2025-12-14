"""Abstract base class for input sources."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading

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