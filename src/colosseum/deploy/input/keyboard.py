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