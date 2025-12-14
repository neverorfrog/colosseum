# Booster T1 Deployment Guide

**Complete guide for deploying trained velocity tracking policies on the real Booster T1 humanoid robot.**

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Hardware Setup](#hardware-setup)
4. [Software Setup](#software-setup)
5. [Deploying Your Policy](#deploying-your-policy)
6. [Control Modes](#control-modes)
7. [Safety Procedures](#safety-procedures)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Configuration](#advanced-configuration)

---

## Overview

This guide walks you through deploying a trained velocity tracking policy (from `colosseum/tasks/velocity/`) onto the physical Booster T1 robot using the deployment infrastructure inspired by holosoma_inference.

**Deployment Pipeline:**
```
Trained Policy (ONNX)
  ↓
Colosseum Deployment Controller
  ↓
Booster SDK (DDS Communication)
  ↓
Real Robot Hardware
```

**Control Loop:**
- **Policy Rate:** 50 Hz (configurable)
- **Low-Level Control:** 500 Hz (handled by robot firmware)
- **Control Type:** Position control with PD gains
- **Communication:** DDS (CycloneDDS) via Booster SDK

---

## Prerequisites

### Software Requirements

1. **Colosseum Environment:**
   ```bash
   cd /path/to/colosseum
   pixi install  # Install all dependencies including Booster SDK
   ```

2. **Booster Robotics SDK:**
   - Should be automatically installed via pixi
   - Verify: `pixi run python -c "import booster_robotics_sdk; print('OK')"`

3. **Trained Policy:**
   - ONNX model exported from your training run
   - Must match T1 23-DOF configuration
   - Location: e.g., `models/t1_velocity_policy.onnx`

### Hardware Requirements

1. **Booster T1 Robot** (23-DOF full body)
   - Powered on and calibrated
   - In safe operating environment
   - Emergency stop accessible

2. **Control Computer:**
   - Linux system (Ubuntu 20.04+ recommended)
   - Network connection to robot
   - USB port for joystick (optional)

3. **Input Device (Choose One):**
   - **Option A:** USB/Bluetooth gamepad (Xbox, PlayStation controller)
   - **Option B:** Keyboard control (built-in)

### Network Setup

The robot communicates via DDS, which can run in two modes:

**Local Mode (Simulation/Testing):**
```bash
# Robot controller and policy on same machine
export COLOSSEUM_ROBOT_INTERFACE="lo"  # localhost
export COLOSSEUM_DOMAIN_ID=0
```

**Network Mode (Real Robot):**
```bash
# Robot on separate hardware, connected via Ethernet
export COLOSSEUM_ROBOT_INTERFACE="eth0"  # or wlan0
export COLOSSEUM_DOMAIN_ID=0  # Must match robot's domain
```

---

## Hardware Setup

### 1. Robot Preparation

**Safety First:**
1. Clear at least 3m radius around robot
2. Ensure emergency stop is accessible
3. Have spotter ready to catch robot if needed
4. Start with robot in sitting position

**Power On Sequence:**
```
1. Connect robot to power supply (ensure sufficient capacity)
2. Turn on main power switch
3. Wait for boot sequence (LED indicators)
4. Verify all joints initialized (listen for servo activation)
5. Check robot is in kIdle or kPrepareStand mode
```

**Initial Pose:**
- Robot should be in **sitting** or **kneeling** position
- All joints within normal range
- No joint at hard limit

### 2. Network Connection

**Wired Ethernet (Recommended):**
```bash
# 1. Connect Ethernet cable between robot and computer
# 2. Verify connection
ping <robot_ip_address>

# 3. Check DDS communication
pixi run python -c "
from booster_robotics_sdk import ChannelFactory
from booster_robotics_sdk.common import get_interface_ip
ip = get_interface_ip('eth0')
ChannelFactory.Instance().Init(0, ip)
print(f'DDS initialized on {ip}')
"
```

**Wireless (Alternative):**
```bash
# Connect to same WiFi network as robot
# Check latency (should be <5ms)
ping -c 10 <robot_ip_address>
```

### 3. Joystick Setup (Optional)

**Connect Gamepad:**
```bash
# 1. Plug in USB gamepad (or pair Bluetooth)
# 2. Verify detection
pixi run python -c "
import evdev
devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
for d in devices:
    print(f'{d.path}: {d.name}')
"

# Should see your gamepad listed, e.g.:
# /dev/input/event3: Xbox Wireless Controller
```

**Test Joystick:**
```bash
# Use evtest to verify axes/buttons
sudo apt install evtest
evtest /dev/input/event3

# Move sticks and press buttons - you should see events
```

---

## Software Setup

### 1. Export Your Trained Policy

**From Training Checkpoint:**

```python
# scripts/export_velocity_policy.py
import torch
from mjlab.utils.torch_jit_utils import load_jit_onnx
from colosseum.tasks.velocity.deploy.t1_23dof.robot_cfg import T1_23DOF_DEPLOY_CFG

# Load trained checkpoint
policy_path = "runs/velocity_t1_23dof/nn/policy.pt"
policy = torch.jit.load(policy_path)
policy.eval()

# Create example input (matches observation space)
obs_dim = 101  # From your training config
dummy_input = {"actor_obs": torch.zeros(1, obs_dim)}

# Export to ONNX with metadata
output_path = "models/t1_velocity_policy.onnx"
torch.onnx.export(
    policy,
    (dummy_input,),
    output_path,
    input_names=["actor_obs"],
    output_names=["actions"],
    dynamic_axes={"actor_obs": {0: "batch"}, "actions": {0: "batch"}},
    opset_version=17,
)

# Add PD gains to ONNX metadata (crucial!)
import onnx
model = onnx.load(output_path)

# Get training gains from robot config
kp = T1_23DOF_DEPLOY_CFG.joint_stiffness
kd = T1_23DOF_DEPLOY_CFG.joint_damping

# Store as metadata
meta = model.metadata_props.add()
meta.key = "motor_kp"
meta.value = ",".join(map(str, kp))

meta = model.metadata_props.add()
meta.key = "motor_kd"
meta.value = ",".join(map(str, kd))

onnx.save(model, output_path)
print(f"Policy exported to {output_path} with PD gains")
```

Run export:
```bash
pixi run python scripts/export_velocity_policy.py
```

### 2. Verify Policy Configuration

**Check Observation Space Match:**

Your policy's observation space must match the deployment configuration. For T1 velocity task:

```python
# Expected observation space (101 dims for T1 velocity)
{
    "base_ang_vel": 3,        # IMU gyro
    "projected_gravity": 3,   # IMU-based gravity vector
    "dof_pos": 29,            # Joint positions (relative to default)
    "dof_vel": 29,            # Joint velocities
    "actions": 29,            # Previous action
    "command_lin_vel": 2,     # Velocity command (x, y)
    "command_ang_vel": 1,     # Angular velocity command (yaw)
    "sin_phase": 2,           # Gait phase sin (per foot)
    "cos_phase": 2,           # Gait phase cos (per foot)
}
# Total: 3 + 3 + 29 + 29 + 29 + 2 + 1 + 2 + 2 = 100 dims
# (Check your training config for exact dimensions)
```

**Verify ONNX Model:**
```bash
pixi run python -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('models/t1_velocity_policy.onnx')

# Check input/output shapes
print('Inputs:', session.get_inputs()[0].name, session.get_inputs()[0].shape)
print('Outputs:', session.get_outputs()[0].name, session.get_outputs()[0].shape)

# Check metadata (PD gains)
metadata = session.get_modelmeta().custom_metadata_map
print('Motor KP:', metadata.get('motor_kp', 'NOT FOUND'))
print('Motor KD:', metadata.get('motor_kd', 'NOT FOUND'))

# Test inference
dummy_input = {'actor_obs': np.zeros((1, 101), dtype=np.float32)}
output = session.run(None, dummy_input)
print('Test inference output shape:', output[0].shape)  # Should be (1, 29)
"
```

### 3. Configuration Files

**Robot Configuration** (`src/colosseum/tasks/velocity/deploy/t1_23dof/robot_cfg.py`):

This file defines all robot-specific parameters. Key sections:

```python
T1_23DOF_DEPLOY_CFG = RobotCfg(
    name="Booster_T1_23DOF",

    # Joint ordering (matches real robot hardware)
    joint_names=[
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", ...,
    ],

    # Joint ordering in simulation (typically alphabetical)
    sim_joint_names=[
        "Ankle_Pitch_Left", "Ankle_Pitch_Right", ...,
    ],

    # PD gains (MUST match training values!)
    joint_stiffness=[206.83, 188.76, ...],  # From t1_actuators.py
    joint_damping=[13.17, 12.02, ...],

    # Default standing pose
    default_joint_pos=[0.0, 0.0, 0.2, -1.35, ...],

    # Safety limits
    effort_limit=[45.0, 45.0, ...],  # Nm

    # Robot model
    mjcf_path=str(src_dir() / "robots/booster_t1/xmls/T1_23dof.xml"),
)
```

**CRITICAL:** Ensure `joint_stiffness` and `joint_damping` match the values used during training (computed from motor specs in `t1_actuators.py`). Mismatched gains will cause instability or sluggish behavior.

---

## Deploying Your Policy

### Option 1: Using MuJoCo Simulation (Sim-to-Sim Testing)

Before deploying to the real robot, **always test in MuJoCo simulation first:**

```bash
# Run velocity deployment in MuJoCo
pixi run python -m colosseum.deploy.core.controllers.mujoco_controller \
    --robot-cfg t1_23dof \
    --policy-cfg velocity \
    --model-path models/t1_velocity_policy.onnx \
    --use-joystick  # Or omit for keyboard control
```

**What to Verify:**
- [ ] Robot maintains balance
- [ ] Velocity commands are tracked smoothly
- [ ] No oscillations or instability
- [ ] Actions stay within joint limits
- [ ] Gait is smooth and periodic

**Keyboard Controls (Simulation):**
```
]           Start policy
o           Stop policy (hold default pose)
i           Go to default standing pose
=           Toggle walk/stand mode

w/s         Increase/decrease forward velocity
a/d         Increase/decrease lateral velocity
q/e         Increase/decrease yaw velocity
z           Zero all velocity commands

v/b         Increase/decrease KP gain multiplier
f/g         Increase/decrease KD gain multiplier
r           Reset gain multipliers to 1.0

ESC         Exit
```

### Option 2: Deploying to Real Robot

**⚠️ SAFETY CHECKLIST:**
- [ ] Robot in safe starting pose
- [ ] Clear 3m radius around robot
- [ ] Emergency stop accessible
- [ ] Spotter ready
- [ ] Policy tested in simulation
- [ ] Network connection stable (<5ms latency)
- [ ] Gains match training values

**Deploy Command:**

```bash
# Set network configuration
export COLOSSEUM_ROBOT_INTERFACE="eth0"  # or your network interface
export COLOSSEUM_DOMAIN_ID=0

# Run deployment
pixi run python -m colosseum.deploy.core.controllers.mujoco_controller \
    --robot-cfg t1_23dof \
    --policy-cfg velocity \
    --model-path models/t1_velocity_policy.onnx \
    --use-joystick \
    --real-robot  # Connects to real robot via Booster SDK
```

**Startup Sequence:**

```
[Stage 1] SDK Initialization
  ├─ ChannelFactory.Init(domain_id, ip)
  ├─ B1LowCmdPublisher.InitChannel()
  ├─ B1LowStateSubscriber.InitChannel()
  └─ B1LocoClient.ChangeMode(kCustom)

[Stage 2] Policy Loading
  ├─ Load ONNX model
  ├─ Extract PD gains from metadata
  └─ Initialize observation buffers

[Stage 3] Safe Initialization
  ├─ Read current robot state
  ├─ Interpolate to default standing pose (3 seconds)
  └─ Wait for user command

[Stage 4] Policy Active
  └─ 50Hz control loop (press ] to start)
```

**What You'll See:**

```
=== Colosseum Deployment Controller ===
Robot: Booster T1 23-DOF
Policy: Velocity Tracking (101 obs → 29 actions)
Network: eth0 (192.168.1.100)
Domain: 0

[Init] Loading policy from models/t1_velocity_policy.onnx
[Init] Extracted PD gains: KP=[206.8, ...], KD=[13.2, ...]
[Init] Connecting to robot...
[Init] Robot state: kCustom mode
[Init] Moving to default pose... (0/180 steps)
[Ready] Press ] to start policy, i to reset pose

[Active] FPS: 50.1 | Latency: read=0.5ms, inference=1.2ms, pub=0.3ms
[Active] Commands: vx=0.5 m/s, vy=0.0 m/s, vyaw=0.0 rad/s
[Active] Phase: L=0.0π, R=1.0π
```

---

## Control Modes

### Keyboard Control

**Mode Switching:**
- `]` - **Start policy** (robot begins tracking velocity commands)
- `o` - **Stop policy** (holds current pose with PD control)
- `i` - **Reset to default pose** (interpolates to standing pose over 3s)
- `=` - **Toggle stand/walk** (switches between standing and walking)

**Velocity Commands (when policy active):**
```
Forward/Back:   w (increase) / s (decrease)
Left/Right:     a (increase) / d (decrease)
Rotate:         q (CCW) / e (CW)
Zero velocity:  z

Velocity limits:
  - Forward/back: ±0.8 m/s
  - Lateral: ±0.5 m/s
  - Yaw: ±0.5 rad/s
```

**Safety Controls:**
```
v/b/f/g/r    Adjust PD gain multipliers (emergency damping)
ESC          Emergency stop and exit
```

### Joystick Control

**Recommended Setup:** Xbox or PlayStation controller

**Controls:**
```
Left Stick:
  Forward/Back  →  Linear velocity X (±0.8 m/s)
  Left/Right    →  Linear velocity Y (±0.5 m/s)

Right Stick:
  Left/Right    →  Angular velocity Yaw (±0.5 rad/s)

Buttons:
  A button      →  Start policy
  B button      →  Stop policy
  Y button      →  Reset to default pose
  Start         →  Toggle stand/walk mode
  L1 + R1       →  Emergency stop and exit

D-pad:
  Up/Down       →  Adjust KP gain multiplier
  Left/Right    →  Adjust KD gain multiplier
```

**Joystick Advantages:**
- Smooth analog velocity control
- Instantaneous zero velocity (center stick)
- Two-handed operation for complex commands
- Can control while observing robot from distance

---

## Safety Procedures

### Pre-Deployment Checks

**Robot State:**
```bash
# Check robot mode (should be kIdle or kPrepareStand)
pixi run python -c "
from booster_robotics_sdk import B1LocoClient
client = B1LocoClient()
client.Init()
mode = client.GetRobotMode()
print(f'Robot mode: {mode}')
"

# Check battery level
pixi run python -c "
from booster_robotics_sdk import B1LowStateSubscriber
def check_battery(msg):
    print(f'Battery: {msg.power_v}V, {msg.power_a}A')
    exit(0)
sub = B1LowStateSubscriber(check_battery)
sub.InitChannel()
import time; time.sleep(2)
"
```

**Joint Limits:**
```bash
# Verify all joints within safe range
pixi run python -c "
from colosseum.tasks.velocity.deploy.t1_23dof.robot_cfg import T1_23DOF_DEPLOY_CFG
import numpy as np

# Read current joint positions
# (integrate with state processor)

# Check against limits
qmin = -2.0  # Example limit
qmax = 2.0
for i, (name, q) in enumerate(zip(cfg.joint_names, current_q)):
    if q < qmin or q > qmax:
        print(f'WARNING: {name} = {q:.2f} rad (outside safe range)')
"
```

### Emergency Procedures

**Emergency Stop (Policy-Level):**
1. Press `ESC` key or `L1+R1` on joystick
2. Robot will **immediately stop** accepting commands
3. Built-in PD control maintains current pose
4. Program exits cleanly

**Emergency Stop (Hardware-Level):**
1. Press **physical emergency stop button** on robot
2. All motors disengage immediately
3. Robot will collapse - **be ready to support**

**Power Off:**
1. First stop policy (ESC)
2. Switch robot to kIdle mode:
   ```bash
   pixi run python -c "
   from booster_robotics_sdk import B1LocoClient, RobotMode
   client = B1LocoClient()
   client.Init()
   client.ChangeMode(RobotMode.kIdle)
   "
   ```
3. Wait 5 seconds
4. Turn off main power switch

### Fault Recovery

**Robot Falls:**
1. **DO NOT** try to restart immediately
2. Turn off policy (ESC)
3. Switch to kIdle mode
4. Manually position robot to sitting pose
5. Check for hardware damage
6. Review logs to identify cause

**Oscillations/Instability:**
1. **Immediately** reduce KP/KD gains (v/b/f/g keys)
2. Reduce velocity commands to zero (z key)
3. If continues, press ESC to stop
4. Likely causes:
   - PD gains too high (check deployment vs training values)
   - Network latency spikes
   - Policy observation mismatch

**Network Loss:**
1. Robot will timeout and hold last position
2. Reconnect network
3. Restart deployment script
4. Robot must be re-initialized from default pose

---

## Troubleshooting

### Policy Not Loading

**Error:** `FileNotFoundError: models/t1_velocity_policy.onnx`

**Fix:**
```bash
# Verify file exists
ls -lh models/t1_velocity_policy.onnx

# Check permissions
chmod 644 models/t1_velocity_policy.onnx

# Verify ONNX validity
pixi run python -c "
import onnx
model = onnx.load('models/t1_velocity_policy.onnx')
onnx.checker.check_model(model)
print('ONNX model valid')
"
```

### Robot Not Responding

**Error:** `DDS communication timeout`

**Diagnosis:**
```bash
# 1. Check network connectivity
ping <robot_ip>

# 2. Check DDS domain
echo $COLOSSEUM_DOMAIN_ID  # Should match robot

# 3. Verify Booster SDK
pixi run python -c "
from booster_robotics_sdk import ChannelFactory, get_interface_ip
ip = get_interface_ip('eth0')
print(f'Using interface: eth0 -> {ip}')
ChannelFactory.Instance().Init(0, ip)
"

# 4. Check robot mode
pixi run python -c "
from booster_robotics_sdk import B1LocoClient
client = B1LocoClient()
client.Init()
print(f'Robot mode: {client.GetRobotMode()}')
"
```

**Fix:**
- Ensure robot is powered on and initialized
- Check firewall isn't blocking DDS ports (default: 7400-7600)
- Verify both robot and computer on same subnet
- Try restarting robot controller

### Jerky/Unstable Motion

**Symptoms:**
- Robot oscillates around target position
- Joints vibrate or shake
- Sudden acceleration/deceleration

**Likely Causes:**

1. **PD Gain Mismatch:**
   ```python
   # Compare deployment gains to training gains
   from colosseum.robots.booster_t1.t1_actuators import (
       T1_ACTUATOR_HIP_PITCH,
       compute_pd_gains
   )

   # Training gains (example)
   motor = T1_ACTUATOR_HIP_PITCH
   train_kp, train_kd = compute_pd_gains(motor)

   # Deployment gains
   from colosseum.tasks.velocity.deploy.t1_23dof.robot_cfg import T1_23DOF_DEPLOY_CFG
   deploy_kp = T1_23DOF_DEPLOY_CFG.joint_stiffness[4]  # Example: hip pitch
   deploy_kd = T1_23DOF_DEPLOY_CFG.joint_damping[4]

   print(f"Training: KP={train_kp:.2f}, KD={train_kd:.2f}")
   print(f"Deployment: KP={deploy_kp:.2f}, KD={deploy_kd:.2f}")

   # Should match exactly!
   ```

2. **Action Scaling Mismatch:**
   - Check `policy_action_scale` in deployment matches training
   - Verify action limits in ONNX model

3. **Network Latency:**
   ```bash
   # Measure round-trip time
   ping -c 100 <robot_ip> | tail -1
   # Should be <5ms average
   ```

4. **Control Loop Rate:**
   - Verify policy running at 50Hz (check FPS output)
   - System CPU usage should be <30%

### Observation Mismatch

**Error:** Policy produces random/nonsensical actions

**Diagnosis:**
```python
# Compare training obs vs deployment obs
# Create test script: scripts/test_observation_match.py

import numpy as np
from colosseum.deploy.core.controllers.mujoco_controller import MujocoController
from colosseum.tasks.velocity.deploy.t1_23dof.robot_cfg import T1_23DOF_DEPLOY_CFG

# Initialize controller (simulation mode)
controller = MujocoController(T1_23DOF_DEPLOY_CFG, ...)

# Get observation
obs = controller.compute_observation()
print(f"Observation shape: {obs.shape}")  # Should be (1, 101)
print(f"Observation ranges:")
for key, idx in obs_indices.items():
    print(f"  {key}: min={obs[0,idx].min():.3f}, max={obs[0,idx].max():.3f}")

# Compare to training observation ranges
# (extract from training logs or config)
```

**Common Mismatches:**
- Joint position offset (check `default_joint_pos` used in both places)
- Velocity scaling (check `obs_scales` match training)
- Phase computation (verify gait period is same)
- Command normalization (check max velocities match)

### Joystick Not Detected

**Error:** `evdev: No suitable joystick device found`

**Fix:**
```bash
# 1. List all input devices
ls -l /dev/input/event*

# 2. Check permissions
sudo chmod a+r /dev/input/event*

# 3. Add user to input group (permanent fix)
sudo usermod -aG input $USER
# Log out and back in

# 4. Verify joystick detected
pixi run python -c "
import evdev
devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
for d in devices:
    caps = d.capabilities()
    if evdev.ecodes.EV_ABS in caps:
        print(f'Gamepad: {d.name} at {d.path}')
"
```

---

## Advanced Configuration

### Custom Observation Spaces

If you trained with a different observation space, modify the deployment config:

```python
# src/colosseum/tasks/velocity/deploy/t1_23dof/policy_cfg.py

from colosseum.deploy.core.controllers import PolicyCfg

class CustomVelocityPolicyCfg(PolicyCfg):
    """Custom velocity policy with additional observations"""

    observation_config = {
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "dof_pos",
            "dof_vel",
            "actions",
            "command_lin_vel",
            "command_ang_vel",
            # Add custom observations
            "foot_contact",  # NEW
            "base_height",   # NEW
        ]
    }

    observation_dims = {
        "base_ang_vel": 3,
        # ... (standard dims)
        "foot_contact": 2,  # NEW
        "base_height": 1,   # NEW
    }

    observation_scales = {
        "base_ang_vel": 1.0,
        # ... (standard scales)
        "foot_contact": 1.0,  # NEW
        "base_height": 1.0,   # NEW
    }
```

Then implement custom observation functions in your policy:

```python
# src/colosseum/deploy/tasks/velocity/policy.py

class CustomVelocityPolicy(VelocityPolicy):

    def compute_foot_contact(self, robot_data):
        """Compute foot contact state from force sensors"""
        # Access robot contact sensors
        contact_forces = robot_data.contact_forces  # Example
        foot_contact = (contact_forces > threshold).astype(float)
        return foot_contact

    def compute_base_height(self, robot_data):
        """Compute base height from IMU and kinematics"""
        base_pos_z = robot_data.base_pos[2]
        return np.array([[base_pos_z]])
```

### Custom Command Processing

Modify velocity command handling for different control schemes:

```python
# src/colosseum/deploy/tasks/velocity/policy.py

class VelocityPolicy:

    def process_joystick_input(self):
        """Custom joystick mapping"""
        joystick = self.interface.get_joystick_msg()

        # Example: Add button-triggered speed boost
        boost = 1.5 if joystick.keys & 0x1 else 1.0  # R1 button

        self.lin_vel_command[0] = joystick.ly * 0.8 * boost
        self.lin_vel_command[1] = joystick.lx * 0.5 * boost
        self.ang_vel_command[0] = joystick.rx * 0.5

        # Example: Mode switching with buttons
        if joystick.keys & 0x100:  # A button
            self.set_stand_mode()
        elif joystick.keys & 0x200:  # B button
            self.set_walk_mode()
```

### Gain Scheduling

Implement adaptive PD gains based on state:

```python
# src/colosseum/deploy/core/controllers/mujoco_controller.py

class MujocoController:

    def ctrl_step(self, dof_targets):
        """Control step with gain scheduling"""

        # Get current state
        dof_vel = self.robot.data.joint_vel

        # Schedule gains based on velocity
        vel_magnitude = np.linalg.norm(dof_vel)
        if vel_magnitude > HIGH_VEL_THRESHOLD:
            # Reduce gains at high velocity (stability)
            kp_multiplier = 0.8
            kd_multiplier = 1.2
        else:
            # Full gains at low velocity (tracking)
            kp_multiplier = 1.0
            kd_multiplier = 1.0

        # Apply scheduled gains
        self.interface.set_gain_levels(kp_multiplier, kd_multiplier)

        # Send command
        for _ in range(self.decimation):
            self.mj_data.ctrl = dof_targets
            mujoco.mj_step(self.mj_model, self.mj_data)
```

### Multi-Policy Deployment

Run multiple policies simultaneously (e.g., locomotion + manipulation):

```python
# scripts/deploy_multi_policy.py

from colosseum.deploy.core.controllers import MujocoController
from colosseum.tasks.velocity.deploy import VelocityPolicyCfg
from colosseum.tasks.manipulation.deploy import ManipulationPolicyCfg

# Load both policies
locomotion_policy = VelocityPolicyCfg(...)
manipulation_policy = ManipulationPolicyCfg(...)

# Create controller
controller = MujocoController(robot_cfg, [locomotion_policy, manipulation_policy])

# Control loop
while True:
    # Locomotion controls lower body (legs)
    loco_actions = locomotion_policy.compute_action(obs)

    # Manipulation controls upper body (arms)
    manip_actions = manipulation_policy.compute_action(obs)

    # Merge actions
    full_action = merge_actions(loco_actions, manip_actions, joint_mapping)

    # Send to robot
    controller.send_command(full_action)
```

---

## Best Practices

### Training-Deployment Consistency

**Critical Checklist:**
- [ ] PD gains match exactly (training vs deployment)
- [ ] Action scaling matches (`policy_action_scale`)
- [ ] Observation space identical (dims, scales, names)
- [ ] Default joint pose same
- [ ] Joint limits same
- [ ] Control frequency same (50 Hz typical)

**Verification Script:**
```bash
pixi run python scripts/verify_train_deploy_match.py \
    --training-config runs/velocity_t1_23dof/config.yaml \
    --deployment-config src/colosseum/tasks/velocity/deploy/t1_23dof/robot_cfg.py
```

### Iterative Deployment Process

**Recommended Workflow:**
```
1. Train policy in simulation (colosseum)
   ↓
2. Export to ONNX with metadata
   ↓
3. Test in MuJoCo sim-to-sim (deployment controller)
   ├─ Verify actions look reasonable
   ├─ Check stability and smoothness
   └─ Validate observation processing
   ↓
4. Deploy to real robot with LOW gains
   ├─ Start with 0.3x KP/KD multipliers
   ├─ Verify no oscillations
   └─ Gradually increase to 1.0x
   ↓
5. Deploy at full gains
   ├─ Start with low velocity commands
   ├─ Gradually increase command magnitudes
   └─ Monitor for instabilities
   ↓
6. Full performance deployment
```

### Safety Margins

**Conservative Limits (First Deployment):**
```python
# Reduce action magnitude
policy_action_scale = 0.15  # Instead of 0.25

# Reduce PD gains
kp_level = 0.5  # 50% of training gains
kd_level = 0.8  # 80% of training gains

# Limit velocity commands
max_lin_vel = 0.3  # m/s (instead of 0.8)
max_ang_vel = 0.2  # rad/s (instead of 0.5)

# Reduce control rate if needed
rl_rate = 25  # Hz (instead of 50)
```

**Gradually Increase:**
- If stable for 5 minutes → increase by 20%
- Monitor for oscillations, delays, or instability
- Full performance typically reached after 3-4 iterations

### Logging and Monitoring

**Enable Detailed Logging:**
```python
# Add to deployment script
import logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="deployment_log.txt",
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Log critical data every cycle
logger.debug(f"Obs: {obs}")
logger.debug(f"Action: {action}")
logger.debug(f"Commands: vx={vx}, vy={vy}, vyaw={vyaw}")
logger.debug(f"Latency: {latency_ms}ms")
```

**Real-Time Monitoring:**
```bash
# In separate terminal, monitor log
tail -f deployment_log.txt | grep -E "(ERROR|WARNING|Latency)"

# Monitor network latency
ping -i 0.2 <robot_ip>

# Monitor CPU usage
htop -p $(pgrep -f mujoco_controller)
```

---

## Summary

**Key Takeaways:**

1. **Always test in simulation first** before real robot deployment
2. **Match training and deployment configs exactly** (gains, scales, obs)
3. **Start conservative** (low gains, low commands) and gradually increase
4. **Monitor continuously** (latency, stability, FPS)
5. **Have emergency stop accessible** at all times
6. **Use joystick control** for smoother velocity commands

**Quick Start Checklist:**
```bash
# 1. Export policy
pixi run python scripts/export_velocity_policy.py

# 2. Test in sim
pixi run python -m colosseum.deploy.core.controllers.mujoco_controller \
    --model-path models/t1_velocity_policy.onnx

# 3. If stable, deploy to robot
export COLOSSEUM_ROBOT_INTERFACE="eth0"
pixi run python -m colosseum.deploy.core.controllers.mujoco_controller \
    --model-path models/t1_velocity_policy.onnx \
    --real-robot \
    --use-joystick

# 4. Start policy (press ] key or A button)
# 5. Give velocity commands (joystick or keyboard)
# 6. Monitor and adjust gains as needed
```

**Need Help?**
- Check `docs/HOLOSOMA_INFERENCE_ANALYSIS.md` for technical deep dive
- Review `docs/DEPLOYMENT_ARCHITECTURE.md` for system design
- See training docs for policy export instructions

**Happy deploying! 🤖**
