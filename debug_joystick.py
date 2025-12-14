
#!/usr/bin/env python3
"""Debug script to identify correct joystick axis and button mappings.

This script lists all input devices and shows real-time events with their codes.
Use this to find the correct axis codes for your gamepad.

Usage:
    python debug_joystick.py
"""
import evdev
import time


def print_device_info():
    """Print all input devices and their capabilities."""
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    
    if not devices:
        print("No input devices found!")
        return
    
    print("=" * 80)
    print("Available Input Devices")
    print("=" * 80)
    print()
    
    for idx, device in enumerate(devices):
        print(f"[{idx}] {device.name}")
        print(f"    Path: {device.path}")
        print(f"    Phys: {device.phys}")
        print()
        
        caps = device.capabilities()
        
        # Print absolute axes
        if evdev.ecodes.EV_ABS in caps:
            print("    Absolute Axes (EV_ABS):")
            abs_axes = caps[evdev.ecodes.EV_ABS]
            for code, absinfo in abs_axes:
                axis_name = evdev.ecodes.ABS[code]
                print(f"      Code {code:2d} ({axis_name:15s}): min={absinfo.min:4d}, max={absinfo.max:4d}, flat={absinfo.flat:4d}, fuzz={absinfo.fuzz:4d}")
        
        # Print keys/buttons
        if evdev.ecodes.EV_KEY in caps:
            print("    Keys/Buttons (EV_KEY):")
            key_codes = caps[evdev.ecodes.EV_KEY]
            # Group by button type
            buttons = []
            for code in key_codes:
                try:
                    btn_name = evdev.ecodes.BTN[code]
                    buttons.append((code, btn_name))
                except KeyError:
                    try:
                        btn_name = evdev.ecodes.KEY[code]
                        buttons.append((code, btn_name))
                    except KeyError:
                        buttons.append((code, f"UNKNOWN_{code}"))
            
            # Sort and print
            for code, name in sorted(buttons):
                print(f"      Code {code:3d} ({name})")
        
        print()


def monitor_events(device_path=None):
    """Monitor events from a specific device."""
    if device_path is None:
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        gamepad = None
        
        for device in devices:
            caps = device.capabilities()
            if evdev.ecodes.EV_ABS in caps and evdev.ecodes.EV_KEY in caps:
                gamepad = device
                print(f"Using device: {device.name}")
                break
        
        if gamepad is None:
            print("No suitable gamepad found!")
            return
    else:
        gamepad = evdev.InputDevice(device_path)
        print(f"Using device: {gamepad.name}")
    
    print()
    print("=" * 80)
    print("Real-time Event Monitor")
    print("=" * 80)
    print("Move axes and press buttons. Press Ctrl+C to exit.")
    print()
    
    try:
        for event in gamepad.read_loop():
            if event.type == evdev.ecodes.EV_ABS:
                axis_name = evdev.ecodes.ABS[event.code]
                print(f"[EV_ABS ] Code {event.code:2d} ({axis_name:15s}): {event.value:4d}")
            elif event.type == evdev.ecodes.EV_KEY:
                try:
                    btn_name = evdev.ecodes.BTN[event.code]
                except KeyError:
                    try:
                        btn_name = evdev.ecodes.KEY[event.code]
                    except KeyError:
                        btn_name = f"UNKNOWN_{event.code}"
                
                state = "PRESSED" if event.value == 1 else "RELEASED"
                print(f"[EV_KEY ] Code {event.code:3d} ({btn_name:20s}): {state}")
    except KeyboardInterrupt:
        print()
        print("Stopped monitoring.")


if __name__ == "__main__":
    import sys
    
    print_device_info()
    
    print("\n")
    print("Starting event monitor...")
    print("Move the RIGHT STICK (yaw axis) and press RT/LT buttons.")
    print()
    time.sleep(2)
    
    monitor_events()
