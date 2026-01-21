# Holosoma Inference Architecture Analysis

**Comprehensive technical analysis of the holosoma_inference deployment framework for humanoid robot policy deployment.**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Configuration System](#configuration-system)
4. [SDK Layer](#sdk-layer)
5. [Policy Layer](#policy-layer)
6. [Communication Layer](#communication-layer)
7. [Control Flow](#control-flow)
8. [Key Design Patterns](#key-design-patterns)
9. [Implementation Details](#implementation-details)
10. [Comparison with Colosseum](#comparison-with-colosseum)

---

## Executive Summary

The **holosoma_inference** package provides a production-ready, robot-agnostic deployment framework for humanoid robot reinforcement learning policies. Built by the Holosoma team at Booster Robotics, it demonstrates best practices for bridging the sim-to-real gap in legged robotics.

### Key Innovations

1. **Robot-Agnostic Architecture**
   - Unified interface abstracts vendor SDKs (Booster, Unitree, ROS2)
   - Factory pattern for command senders and state processors
   - Config-driven robot specifications

2. **Type-Safe Configuration System**
   - Pydantic dataclasses for validation
   - Tyro CLI integration for ergonomic argument parsing
   - Pre-configured presets for common robots (G1, T1)

3. **Performance-First Design**
   - 50Hz policy loop with <3ms latency per cycle
   - ONNX runtime for cross-platform inference
   - Lock-free concurrent state updates

4. **Safety and Observability**
   - Runtime gain adjustment for instability mitigation
   - Per-stage latency tracking (read, inference, publish)
   - Graceful degradation on network loss

### Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                          │
│  (CLI Args, Keyboard, Joystick, Programmatic API)          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Policy Layer                           │
│  - LocomotionPolicy / WholeBodyTrackingPolicy              │
│  - ONNX Inference Engine                                    │
│  - Observation Processing & History Management             │
│  - Command Processing (Velocity, Motion Clips)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                          │
│  - InterfaceWrapper (Unified Robot API)                    │
│  - Command Sender (Robot-Specific)                          │
│  - State Processor (Robot-Specific)                         │
│  - Remote Control Service (Joystick Handler)               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Communication Layer                         │
│  - Booster SDK (DDS/CycloneDDS)                            │
│  - Unitree SDK (C++ Binding)                                │
│  - ROS2 Bridge                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Physical Robot                           │
│  (Booster T1, Unitree G1, etc.)                            │
└─────────────────────────────────────────────────────────────┘
```

---

## System Architecture

### Three-Layer Design Philosophy

The architecture follows a clear separation of concerns across three layers:

#### Layer 1: Policy Layer (`policies/`)

**Responsibility:** Define the reinforcement learning policy interface and execution logic.

**Components:**
- `BasePolicy`: Abstract base class providing core policy infrastructure
- `LocomotionPolicy`: Velocity tracking for legged locomotion
- `WholeBodyTrackingPolicy`: Full-body motion clip tracking

**Key Abstractions:**
```python
class BasePolicy:
    """Abstract policy interface"""

    def run(self) -> None:
        """Main control loop (50Hz default)"""

    def policy_action(self) -> None:
        """Execute one policy step: read → infer → act"""

    def rl_inference(self, robot_state) -> np.ndarray:
        """Run ONNX inference: observations → actions"""

    def prepare_obs_for_rl(self, robot_state) -> dict:
        """Process robot state into policy observations"""
```

**Design Principles:**
- Policy code is **robot-agnostic** (uses InterfaceWrapper)
- Observations are **declaratively configured** (ObservationConfig)
- Inference is **backend-agnostic** (ONNX runtime)

#### Layer 2: Interface Layer (`sdk/`)

**Responsibility:** Provide a unified interface that abstracts robot-specific SDKs.

**Components:**
- `InterfaceWrapper`: Unified API for state reading and command sending
- `BoosterCommandSender` / `UnitreeCommandSender`: Robot-specific command serialization
- `BoosterStateProcessor` / `UnitreeStateProcessor`: Robot-specific state parsing
- `BoosterRemoteControlService`: Joystick input handling

**Key Abstractions:**
```python
class InterfaceWrapper:
    """Robot-agnostic interface"""

    def get_low_state(self) -> np.ndarray:
        """Read robot state: IMU + joint positions/velocities"""

    def send_low_command(self, q, dq, tau, kp, kd) -> None:
        """Send PD control commands to robot"""

    def get_joystick_msg(self) -> JoystickMessage:
        """Read joystick input"""
```

**Design Principles:**
- Vendor SDKs are **hidden behind interfaces**
- Robot selection is **compile-time** (factory pattern)
- State format is **standardized** across robots

#### Layer 3: Communication Layer (External)

**Responsibility:** Vendor-provided SDKs for low-level robot communication.

**Implementations:**
- **Booster SDK** (`booster_robotics_sdk`): DDS-based communication for T1
- **Unitree SDK** (`unitree_sdk2`): C++ binding for G1
- **ROS2 Bridge**: Universal robotic middleware

**Not part of holosoma_inference** - these are external dependencies.

### Data Flow

**Read Path (Robot → Policy):**
```
Physical Robot Sensors
  ↓ (500Hz)
Robot SDK (B1LowState DDS message)
  ↓
StateProcessor.low_state_handler()  [Callback thread]
  ↓
StateProcessor.prepare_low_state()  [Parse & standardize]
  ↓
InterfaceWrapper.get_low_state()  [Cached, lock-free read]
  ↓
Policy.prepare_obs_for_rl()  [Extract observations]
  ↓
Policy.rl_inference()  [ONNX forward pass]
```

**Write Path (Policy → Robot):**
```
Policy.rl_inference()  [ONNX output: actions]
  ↓
Policy.policy_action()  [Scale & offset actions → joint targets]
  ↓
InterfaceWrapper.send_low_command(q_target, ...)
  ↓
CommandSender.send_command()  [Serialize to robot format]
  ↓
Robot SDK (B1LowCmd DDS message)
  ↓ (50Hz)
Robot Low-Level Controller (PD control @ 500Hz)
  ↓
Physical Robot Motors
```

**Key Observation:** Policy runs at **50Hz**, but robot's low-level controller runs at **500Hz**. The robot performs local PD control between policy updates, ensuring smooth motion.

---

## Configuration System

### Design Philosophy

Holosoma uses **Pydantic** for configuration validation and **Tyro** for CLI argument parsing. This provides:
- **Type safety**: Invalid configs caught at runtime
- **IDE support**: Autocomplete and type checking
- **Documentation**: Auto-generated help from docstrings
- **Hierarchical structure**: Nested configs for modularity

### Configuration Hierarchy

```python
@dataclass
class InferenceConfig:
    """Top-level configuration"""

    robot: RobotConfig          # Hardware specifications
    observation: ObservationConfig  # Observation space definition
    task: TaskConfig            # Runtime execution parameters
```

### RobotConfig (`config/config_types/robot.py`)

Defines robot-specific parameters needed for deployment. The deployment configs are intentionally minimal - PD gains and many parameters are loaded from ONNX metadata or training configs.

**Structure:**
```python
@dataclass(frozen=True)
class RobotConfig:
    # Identity (required)
    robot_type: str  # "t1_29dof", "g1_29dof"
    robot: str       # "t1", "g1"

    # Default positions (required)
    default_dof_angles: tuple[float, ...]
    default_motor_angles: tuple[float, ...]

    # Mappings (required)
    motor2joint: tuple[int, ...]
    joint2motor: tuple[int, ...]
    dof_names: tuple[str, ...]
    dof_names_upper_body: tuple[str, ...]
    dof_names_lower_body: tuple[str, ...]

    # SDK configuration (with defaults)
    sdk_type: Literal["unitree", "booster", "ros2"] = "unitree"
    motor_type: Literal["serial", "parallel"] = "serial"
    message_type: Literal["HG", "GO2"] = "HG"

    # Dimensions (with defaults)
    num_motors: int = 29
    num_joints: int = 29
    num_upper_body_joints: int = 14

    # Link names (with defaults)
    torso_link_name: str = "torso_link"
    left_hand_link_name: str | None = None
    right_hand_link_name: str | None = None

    # PD gains - OPTIONAL (loaded from ONNX metadata!)
    motor_kp: tuple[float, ...] | None = None
    motor_kd: tuple[float, ...] | None = None

    # WBT stiff startup (optional)
    stiff_startup_pos: tuple[float, ...] | None = None
    stiff_startup_kp: tuple[float, ...] | None = None
    stiff_startup_kd: tuple[float, ...] | None = None

    # SDK-specific (optional)
    unitree_legged_const: dict[str, Any] | None = None
    weak_motor_joint_index: dict[str, int] | None = None
    motion: dict[str, list[str]] | None = None
```

**Example (T1 29-DOF Deployment Config):**
```python
t1_29dof = RobotConfig(
    # Identity
    robot_type="t1_29dof",
    robot="t1",

    # SDK Configuration
    sdk_type="booster",
    motor_type="serial",
    num_motors=29,
    num_joints=29,
    num_upper_body_joints=16,  # head(2) + arms(14)

    # Default standing pose
    default_dof_angles=(
        0.0, 0.0,  # head (yaw, pitch)
        0.2, -1.35, 0.0, -0.5, 0.0, 0.0, 0.0,  # left arm
        0.2, 1.35, 0.0, 0.5, 0.0, 0.0, 0.0,    # right arm
        0.0,  # waist
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # left leg
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # right leg
    ),
    default_motor_angles=(  # Identical for T1 (no motor remapping)
        0.0, 0.0,
        0.2, -1.35, 0.0, -0.5, 0.0, 0.0, 0.0,
        0.2, 1.35, 0.0, 0.5, 0.0, 0.0, 0.0,
        0.0,
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
    ),

    # Joint names in T1 hardware order
    dof_names=(
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
        "Waist",
        "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
        "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
        "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
    ),

    # Mappings (identity for T1)
    motor2joint=tuple(range(29)),
    joint2motor=tuple(range(29)),

    # Link names
    torso_link_name="Trunk",

    # PD gains - NOT in deployment config! Loaded from ONNX metadata.
    motor_kp=None,
    motor_kd=None,
)
```

**Design Decisions:**

1. **Why separate `motor` vs `joint`?**
   - Some robots have gearing where motor order ≠ joint order
   - Allows policies to use "canonical" joint order while hardware uses physical order
   - T1 uses identity mapping (motor index == joint index)

2. **Why are PD gains optional (None)?**
   - **Key innovation:** Gains are stored in ONNX model metadata, not deployment config!
   - During training, gains are defined in training configs (see next section)
   - At ONNX export, gains are embedded in model metadata
   - At deployment runtime, BasePolicy loads gains from ONNX
   - Config gains only used as fallback if ONNX metadata missing
   - This ensures **single source of truth** (training config → ONNX → deployment)

3. **Why frozen dataclass?**
   - Pydantic enforces immutability and type validation
   - Prevents accidental config modification at runtime
   - Better for type checking and hashing

### Training-Side PD Gains (Single Source of Truth)

PD gains are defined in the **training configuration** and exported to ONNX metadata.

**Location:** `holosoma/holosoma/config_values/robot.py`

**T1 29-DOF Training Config:**
```python
# From holosoma training framework
t1_29dof_waist_wrist = RobotConfig(
    # ... robot specs ...
    control=RobotControlConfig(
        control_type="P",  # Position control
        stiffness={
            "Head_yaw": 5, "Head_pitch": 5,
            "Hip_Yaw": 200, "Hip_Roll": 200, "Hip_Pitch": 200, "Knee": 200,
            "Ankle_Pitch": 50, "Ankle_Roll": 50,
            "Waist": 200,
            "Shoulder_Pitch": 20, "Shoulder_Roll": 20,
            "Elbow_Pitch": 20, "Elbow_Yaw": 20,
            "Wrist_Pitch": 20, "Wrist_Yaw": 20, "Hand_Roll": 20,
        },
        damping={
            "Head_yaw": 0.5, "Head_pitch": 0.5,
            "Hip_Yaw": 5, "Hip_Roll": 5, "Hip_Pitch": 5, "Knee": 5,
            "Ankle_Pitch": 3, "Ankle_Roll": 3,
            "Waist": 5,
            "Shoulder_Pitch": 0.5, "Shoulder_Roll": 0.5,
            "Elbow_Pitch": 0.5, "Elbow_Yaw": 0.5,
            "Wrist_Pitch": 0.5, "Wrist_Yaw": 0.5, "Hand_Roll": 0.5,
        },
        action_scale=0.25,
    ),
)
```

**Flow:**
1. Training: Gains defined in RobotControlConfig → used by simulator
2. Export: Gains embedded in ONNX metadata during policy export
3. Deployment: BasePolicy loads gains from ONNX metadata at runtime

**PD Gain Loading Priority:**
```python
# In BasePolicy._init_policy_components():
metadata = self.policy_session.get_modelmeta().custom_metadata_map

if 'motor_kp' in metadata:
    # Load from ONNX (highest priority)
    self.motor_kp = np.array([float(x) for x in metadata['motor_kp'].split(',')])
elif self.robot_config.motor_kp is not None:
    # Fallback to config override
    self.motor_kp = np.array(self.robot_config.motor_kp)
else:
    # Error: No gains available
    raise ValueError("No PD gains found in ONNX metadata or config!")
```

### ObservationConfig (`config/config_types/observation.py`)

Defines the observation space structure for the policy.

**Structure:**
```python
@dataclass
class ObservationConfig:
    """Observation space configuration"""

    # Observation groups (for multi-head policies)
    obs_dict: Dict[str, List[str]]
    # Example: {"actor_obs": ["base_ang_vel", "dof_pos", ...]}

    # Dimension of each observation term
    obs_dims: Dict[str, int]
    # Example: {"base_ang_vel": 3, "dof_pos": 29}

    # Normalization scales
    obs_scales: Dict[str, float]
    # Example: {"base_ang_vel": 1.0, "dof_vel": 0.1}

    # Temporal stacking (history length per group)
    history_length_dict: Dict[str, int]
    # Example: {"actor_obs": 1}  # No history stacking
```

**Example (T1 Locomotion):**
```python
loco_t1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",      # IMU gyroscope
            "projected_gravity", # Gravity in robot frame
            "dof_pos",           # Joint positions (relative to default)
            "dof_vel",           # Joint velocities
            "actions",           # Previous action (history)
            "command_lin_vel",   # Velocity command (x, y)
            "command_ang_vel",   # Angular velocity command (yaw)
            "sin_phase",         # Gait phase sin (per foot)
            "cos_phase",         # Gait phase cos (per foot)
        ]
    },
    obs_dims={
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "sin_phase": 2,  # Per foot
        "cos_phase": 2,
    },
    obs_scales={
        "base_ang_vel": 1.0,      # No scaling
        "projected_gravity": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.1,           # Reduce large velocities
        "actions": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1  # No temporal stacking
    }
)
# Total observation dim: 3+3+29+29+29+2+1+2+2 = 100
```

**Design Decisions:**

1. **Why separate `obs_dict` and `obs_dims`?**
   - `obs_dict` defines **logical grouping** (for multi-head policies)
   - `obs_dims` defines **tensor shapes** (for buffer allocation)

2. **Why `obs_scales`?**
   - **Normalization** improves policy training stability
   - Must be **identical** in training and deployment

3. **Why `history_length_dict`?**
   - Some observations benefit from **temporal context**
   - Example: velocity history helps estimate acceleration
   - Implemented via `collections.deque` for efficient rolling buffer

### TaskConfig (`config/config_types/task.py`)

Runtime execution parameters.

**Structure:**
```python
@dataclass
class TaskConfig:
    """Task execution configuration"""

    # Policy parameters
    model_path: str  # Path to ONNX model or wandb:// URI
    policy_action_scale: float = 0.25  # Action scaling factor

    # Control parameters
    rl_rate: int = 50  # Policy execution rate (Hz)
    decimation: int = 1  # Unused (for compatibility)

    # Gait parameters (locomotion-specific)
    use_phase: bool = True
    gait_period: float = 1.0  # Seconds per gait cycle

    # Communication parameters
    domain_id: int = 0  # DDS domain ID
    interface: str = "lo"  # Network interface (lo, eth0, wlan0)

    # Input mode
    use_joystick: bool = False  # True: joystick, False: keyboard
```

**Example (T1 Velocity Task):**
```python
task_config = TaskConfig(
    model_path="models/t1_velocity_policy.onnx",
    policy_action_scale=0.25,
    rl_rate=50,
    use_phase=True,
    gait_period=1.0,
    domain_id=0,
    interface="eth0",
    use_joystick=True,
)
```

### CLI Integration with Tyro

Tyro automatically generates CLI from config dataclasses:

```bash
# Full config specification
python run_policy.py \
    --robot.robot-type t1_29dof \
    --robot.sdk-type booster \
    --task.model-path models/policy.onnx \
    --task.rl-rate 50 \
    --task.use-joystick

# Using config presets
python run_policy.py inference:t1-29dof-loco \
    --task.model-path models/policy.onnx
```

**Config Presets** (`config/config_values/inference.py`):
```python
# Pre-configured inference configs
g1_29dof_loco = InferenceConfig(
    robot=g1_29dof,  # From robot.py
    observation=loco_g1_29dof,  # From observation.py
    task=TaskConfig(...),
)

t1_29dof_loco = InferenceConfig(
    robot=t1_29dof,
    observation=loco_t1_29dof,
    task=TaskConfig(...),
)

# CLI usage:
# python run_policy.py inference:t1-29dof-loco --task.model-path path/to/model.onnx
```

---

## SDK Layer

### InterfaceWrapper (`sdk/interface_wrapper.py`)

The **InterfaceWrapper** is the central abstraction that provides a unified interface for different robot SDKs.

**Responsibilities:**
1. Backend selection (sdk2py vs binding)
2. State reading abstraction
3. Command sending abstraction
4. Joystick input processing
5. Gain level management

**Initialization:**
```python
class InterfaceWrapper:
    def __init__(
        self,
        robot_config: RobotConfig,
        use_joystick: bool,
        domain_id: int,
        interface: str,
        motor_kp: np.ndarray,
        motor_kd: np.ndarray,
    ):
        # 1. Initialize SDK backend
        if robot_config.sdk_type == "booster":
            ip = get_interface_ip(interface)  # e.g., "192.168.1.100"
            ChannelFactory.Instance().Init(domain_id, ip)
            self.backend = "sdk2py"  # Python SDK
        elif robot_config.sdk_type == "unitree":
            self.backend = "binding"  # C++ binding

        # 2. Create command sender (factory pattern)
        self.command_sender = create_command_sender(robot_config)

        # 3. Create state processor (factory pattern)
        self.state_processor = create_state_processor(robot_config)

        # 4. Initialize joystick (if enabled)
        if use_joystick:
            self.joystick_service = create_remote_control_service(robot_config)

        # 5. Initialize gain levels
        self.kp_level = 1.0  # 100% of nominal gains
        self.kd_level = 1.0
```

**State Reading:**
```python
def get_low_state(self) -> np.ndarray:
    """
    Read current robot state (cached from subscriber callback)

    Returns:
        np.ndarray: shape (1, 4*N+24) where N = num_joints
            [0:3]     base_pos (x, y, z) - zeros (assumed at origin)
            [3:7]     base_quat (w, x, y, z) - from IMU
            [7:7+N]   joint_pos (N joints) - from encoders
            [7+N:...]  base_vel, joint_vel, tau_est, acc
    """
    if self.backend == "sdk2py":
        return self.state_processor.get_state()  # Cached state
    else:
        return self.state_processor.prepare_low_state(
            self.state_processor.latest_msg
        )
```

**Command Sending:**
```python
def send_low_command(
    self,
    cmd_q: np.ndarray,      # Target positions (N,)
    cmd_dq: np.ndarray,     # Target velocities (N,) - typically zeros
    cmd_tau: np.ndarray,    # Feedforward torques (N,) - typically zeros
    dof_pos_latest: np.ndarray,  # Current positions (for safety checks)
) -> None:
    """
    Send PD control commands to robot

    The robot will perform: tau = kp*(cmd_q - q) + kd*(cmd_dq - dq) + cmd_tau
    """
    # Apply gain level scaling
    kp = self.motor_kp * self.kp_level
    kd = self.motor_kd * self.kd_level

    # Send via robot-specific command sender
    self.command_sender.send_command(
        cmd_q, cmd_dq, cmd_tau,
        motor_kp=kp,
        motor_kd=kd,
        kp_level=self.kp_level,
        kd_level=self.kd_level,
        dof_pos_latest=dof_pos_latest,
    )
```

**Gain Management:**
```python
def set_gain_levels(self, kp_level: float, kd_level: float) -> None:
    """
    Adjust PD gain multipliers (for runtime safety tuning)

    Example use cases:
    - Start deployment with low gains (0.5x) for safety
    - Reduce gains during instability
    - Increase gains for tracking performance
    """
    self.kp_level = np.clip(kp_level, 0.1, 2.0)  # 10% to 200%
    self.kd_level = np.clip(kd_level, 0.1, 2.0)
```

### BoosterCommandSender (`sdk/command_sender/booster/booster_command_sender.py`)

Robot-specific implementation for Booster T1 command interface.

**Initialization:**
```python
class BoosterCommandSender:
    def __init__(self, config: RobotConfig):
        self.config = config
        self._init_sdk_components()

    def _init_sdk_components(self):
        """Initialize Booster SDK components"""
        # 1. Create command publisher
        self.lowcmd_publisher_ = B1LowCmdPublisher()
        self.lowcmd_publisher_.InitChannel()

        # 2. Create client for mode switching
        self.client = B1LocoClient()
        self.client.Init()

        # 3. Switch robot to custom control mode
        self.client.ChangeMode(RobotMode.kCustom)
        print("Robot switched to kCustom mode")
```

**Command Serialization:**
```python
def send_command(
    self,
    cmd_q: np.ndarray,
    cmd_dq: np.ndarray,
    cmd_tau: np.ndarray,
    motor_kp: np.ndarray,
    motor_kd: np.ndarray,
    **kwargs
) -> None:
    """Serialize and send command to T1"""

    # 1. Create command message
    low_cmd = LowCmd()
    low_cmd.cmd_type = LowCmdType.SERIAL  # Serial motor mode

    # 2. Fill motor commands
    for i in range(self.config.num_motors):
        # Handle joint-to-motor mapping (if needed)
        joint_id = self.config.motor2joint[i] if self.config.motor2joint else i

        # Set PD control parameters
        low_cmd.motor_cmd[i].q = cmd_q[joint_id]
        low_cmd.motor_cmd[i].dq = cmd_dq[joint_id]
        low_cmd.motor_cmd[i].tau = cmd_tau[joint_id]
        low_cmd.motor_cmd[i].kp = motor_kp[i]
        low_cmd.motor_cmd[i].kd = motor_kd[i]

    # 3. Send via DDS
    self.lowcmd_publisher_.Write(low_cmd)
```

**Key Observations:**
- Commands are sent via **DDS publish** (fire-and-forget)
- Robot firmware receives commands at 500Hz and applies PD control
- `kp` and `kd` can be **per-motor**, allowing heterogeneous gains

### BoosterStateProcessor (`sdk/state_processor/booster/booster_state_processor.py`)

Robot-specific implementation for Booster T1 state parsing.

**Initialization:**
```python
class BoosterStateProcessor:
    def __init__(self, config: RobotConfig):
        self.config = config

        # Allocate state buffers
        n = config.num_joints
        self.q = np.zeros(7 + n)      # base_pos(3) + base_quat(4) + joints(N)
        self.dq = np.zeros(6 + n)     # base_lin_vel(3) + base_ang_vel(3) + joints(N)
        self.tau_est = np.zeros(6 + n)
        self.ddq = np.zeros(6 + n)

        # Start subscriber
        self._init_sdk_components()

    def _init_sdk_components(self):
        """Subscribe to robot state"""
        self.robot_lowstate_subscriber = B1LowStateSubscriber(
            self.low_state_handler_b1  # Callback function
        )
        self.robot_lowstate_subscriber.InitChannel()
```

**State Parsing (Callback):**
```python
def low_state_handler_b1(self, msg):
    """
    Callback invoked at 500Hz by Booster SDK
    Runs in separate thread - must be thread-safe!
    """
    # 1. Extract IMU data
    imu_state = msg.imu_state

    # Base position (assumed at origin)
    self.q[0:3] = 0.0

    # Base orientation (convert RPY → quaternion)
    rpy = imu_state.rpy
    self.q[3:7] = rpy_to_quat(rpy)

    # Base angular velocity (gyro)
    self.dq[3:6] = imu_state.gyro

    # Linear acceleration (no velocity integration)
    self.ddq[0:3] = imu_state.acc

    # 2. Extract joint data
    robot_joint_state = msg.motor_state_serial
    for i in range(self.config.num_joints):
        # Handle motor-to-joint mapping
        motor_idx = self.config.joint2motor[i] if self.config.joint2motor else i

        # Joint position and velocity
        self.q[7 + i] = robot_joint_state[motor_idx].q
        self.dq[6 + i] = robot_joint_state[motor_idx].dq

        # Estimated torque (from current sensing)
        self.tau_est[6 + i] = robot_joint_state[motor_idx].tau_est
```

**State Retrieval:**
```python
def get_state(self) -> np.ndarray:
    """
    Return latest robot state (called from policy thread @ 50Hz)
    Thread-safe: reads cached values set by callback thread
    """
    return np.concatenate([self.q, self.dq, self.tau_est, self.ddq]).reshape(1, -1)
```

**Design Decisions:**

1. **Why callback-based?**
   - Robot publishes state at 500Hz (high rate)
   - Policy only needs state at 50Hz (low rate)
   - Callback caches latest state → policy reads cache (lock-free)

2. **Why base_pos = zeros?**
   - Real robot doesn't have absolute position sensor
   - Policy should be **position-invariant** anyway
   - Only relative measurements matter (IMU, joint encoders)

3. **Why RPY → quaternion?**
   - IMU outputs roll-pitch-yaw (Euler angles)
   - Policy expects quaternion (continuous representation)
   - Avoids gimbal lock for arbitrary orientations

### BoosterRemoteControlService (`sdk/command_sender/booster/remote_control_service.py`)

Handles joystick input via Linux `evdev` library.

**Initialization:**
```python
class BoosterRemoteControlService:
    def __init__(self):
        # 1. Detect connected joystick
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]

        # 2. Find device with required axes (sticks)
        required_axes = [ABS_X, ABS_Y, ABS_RX, ABS_RY]
        for device in devices:
            caps = device.capabilities()
            if evdev.ecodes.EV_ABS in caps:
                axes = [ax[0] for ax in caps[evdev.ecodes.EV_ABS]]
                if all(ax in axes for ax in required_axes):
                    self.joystick = device
                    print(f"Joystick: {device.name}")
                    break

        # 3. Initialize state
        self.lx = 0.0  # Left stick X (lateral velocity)
        self.ly = 0.0  # Left stick Y (forward velocity)
        self.rx = 0.0  # Right stick X (yaw velocity)
        self.keys = 0  # Button bitmask

        # 4. Start polling thread
        self._start_joystick_thread()

    def _start_joystick_thread(self):
        """Background thread for event polling"""
        self.thread = threading.Thread(target=self._poll_joystick, daemon=True)
        self.thread.start()
```

**Event Polling:**
```python
def _poll_joystick(self):
    """Poll joystick events (runs in background thread)"""
    while True:
        try:
            # Blocking read (waits for next event)
            events = self.joystick.read()

            for event in events:
                if event.type == evdev.ecodes.EV_ABS:
                    # Axis event (stick movement)
                    self._handle_axis(event.code, event.value)

                elif event.type == evdev.ecodes.EV_KEY:
                    # Button event (press/release)
                    self._handle_button(event.code, event.value)

        except Exception as e:
            print(f"Joystick error: {e}")
            time.sleep(0.1)
```

**Axis Mapping:**
```python
def _handle_axis(self, code: int, value: int):
    """Convert joystick axis to velocity command"""

    if code == ABS_Y:  # Left stick Y (forward/back)
        # Joystick range: 0 to 65535, center ~32768
        # Map to: [-max_vx, +max_vx]
        normalized = (value - 32768) / 32768.0  # [-1, +1]
        self.vx = normalized * 0.8  # Max 0.8 m/s
        self.ly = self.vx  # ly is alias for policy

    elif code == ABS_X:  # Left stick X (left/right)
        normalized = (value - 32768) / 32768.0
        self.vy = -normalized * 0.5  # Max 0.5 m/s (inverted)
        self.lx = self.vy

    elif code == ABS_RX:  # Right stick X (rotate)
        normalized = (value - 32768) / 32768.0
        self.vyaw = -normalized * 0.5  # Max 0.5 rad/s (inverted)
        self.rx = self.vyaw
```

**Button Mapping:**
```python
def _handle_button(self, code: int, value: int):
    """Update button bitmask"""
    # Button press: value=1, release: value=0

    if code == BTN_A:
        if value: self.keys |= 0x100  # Set bit 8
        else: self.keys &= ~0x100     # Clear bit 8

    elif code == BTN_B:
        if value: self.keys |= 0x200  # Bit 9
        else: self.keys &= ~0x200

    # ... similar for X, Y, L1, R1, etc.
```

**Key Insight:** Joystick runs in **separate thread** to avoid blocking policy loop. State is read via simple getters (thread-safe due to atomic reads on modern CPUs).

---

## Policy Layer

### BasePolicy (`policies/base.py`)

The abstract base class that all policies inherit from.

**Responsibilities:**
1. ONNX model loading and inference
2. Observation processing and history management
3. Input handling (keyboard/joystick)
4. Control loop orchestration
5. Latency tracking and rate limiting

**Initialization Flow:**
```python
class BasePolicy:
    def __init__(self, config: InferenceConfig):
        # 1. Robot configuration
        self._init_robot_config(config.robot)

        # 2. SDK initialization (must be first for DDS init)
        self._init_sdk_components()

        # 3. Observation configuration
        self._init_obs_config()

        # 4. Communication components
        self._init_communication_components()

        # 5. Policy initialization
        self._init_policy_components(config.task.model_path, ...)

        # 6. Command initialization
        self._init_command_components()

        # 7. Input handlers
        self._init_input_handlers()

        # 8. Phase tracking (for locomotion)
        self._init_phase_components()
```

**Key Initialization Steps:**

**Step 2: SDK Initialization**
```python
def _init_sdk_components(self):
    """Initialize robot SDK (must be first!)"""
    if self.robot_config.sdk_type == "booster":
        # Initialize DDS channel factory
        ip = get_interface_ip(self.interface)
        ChannelFactory.Instance().Init(self.domain_id, ip)
        print(f"DDS initialized: domain={self.domain_id}, ip={ip}")
```

**Step 4: Communication Components**
```python
def _init_communication_components(self):
    """Create interface wrapper"""
    self.interface = InterfaceWrapper(
        robot_config=self.robot_config,
        use_joystick=self.use_joystick,
        domain_id=self.domain_id,
        interface=self.interface_name,
        motor_kp=self.motor_kp,  # From ONNX metadata or config
        motor_kd=self.motor_kd,
    )
```

**Step 5: Policy Initialization (PD Gains from ONNX Metadata)**
```python
def _init_policy_components(self, model_path: str, ...):
    """Load ONNX model and extract PD gains from metadata.

    Key Innovation: PD gains are stored in ONNX metadata (single source of truth).
    Training configs define gains → exported to ONNX → loaded at deployment.
    """

    # 1. Load ONNX model
    self.policy_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']  # Can use GPU if available
    )

    # 2. Extract PD gains from metadata (primary source)
    metadata = self.policy_session.get_modelmeta().custom_metadata_map

    if 'motor_kp' in metadata:
        kp_str = metadata['motor_kp']  # "5,5,20,20,20,..."
        self.motor_kp = np.array([float(x) for x in kp_str.split(',')])
        print(f"Loaded Kp gains from ONNX metadata: {self.motor_kp[:5]}...")
    elif self.robot_config.motor_kp is not None:
        # Fallback: Config override (rare, for debugging)
        self.motor_kp = np.array(self.robot_config.motor_kp)
        print("Using Kp gains from config (ONNX metadata not found)")
    else:
        raise ValueError("No PD gains found in ONNX metadata or config!")

    if 'motor_kd' in metadata:
        kd_str = metadata['motor_kd']
        self.motor_kd = np.array([float(x) for x in kd_str.split(',')])
        print(f"Loaded Kd gains from ONNX metadata: {self.motor_kd[:5]}...")
    elif self.robot_config.motor_kd is not None:
        self.motor_kd = np.array(self.robot_config.motor_kd)
        print("Using Kd gains from config (ONNX metadata not found)")
    else:
        raise ValueError("No PD gains found in ONNX metadata or config!")

    # 3. Create inference function
    def policy_act(obs_dict):
        """Run ONNX inference"""
        return self.policy_session.run(None, obs_dict)[0]

    self.policy = policy_act
```

**PD Gain Flow (Training → Deployment):**
```
1. Training Config (holosoma/config_values/robot.py)
   └─> RobotControlConfig.stiffness/damping dicts

2. Policy Export (training script)
   └─> Embed gains in ONNX metadata
       metadata['motor_kp'] = "5,5,20,20,20,..."
       metadata['motor_kd'] = "0.5,0.5,0.5,0.5,..."

3. Deployment (BasePolicy initialization)
   └─> Load gains from ONNX metadata
       self.motor_kp = np.array([float(x) for x in metadata['motor_kp'].split(',')])

Result: Single source of truth, no config duplication!
```

**Step 3: Observation Configuration**
```python
def _init_obs_config(self):
    """Setup observation processing"""

    # Create history buffers (deques for efficient rolling window)
    self.obs_history = {}
    for obs_name in self.observation_config.obs_dict["actor_obs"]:
        history_len = self.observation_config.history_length_dict["actor_obs"]
        dim = self.observation_config.obs_dims[obs_name]

        # Deque of shape (1, dim) arrays
        self.obs_history[obs_name] = deque(
            [np.zeros((1, dim)) for _ in range(history_len)],
            maxlen=history_len
        )
```

### Main Control Loop

```python
def run(self):
    """Main control loop (50Hz default)"""

    # Initialize rate limiter
    rate = RateLimiter(self.rl_rate)  # 50 Hz
    latency_tracker = LatencyTracker()

    for it in itertools.count():
        latency_tracker.start_cycle()

        # 1. Process joystick input (if enabled)
        if self.use_joystick:
            self.process_joystick_input()

        # 2. Update gait phase (if enabled)
        if self.use_phase:
            self.update_phase_time()

        # 3. Execute policy action (core step)
        self.policy_action()

        latency_tracker.end_cycle()

        # 4. Print debug info (every 50 iterations = 1 second @ 50Hz)
        if it % 50 == 0:
            print(f"[Iter {it}] FPS: {latency_tracker.get_fps():.1f}")
            print(latency_tracker.get_stats_str())

        # 5. Sleep to maintain rate
        rate.sleep()
```

### Core Policy Action

```python
def policy_action(self):
    """Execute one policy step: read → infer → act"""

    # ===== Stage 1: Read State =====
    with latency_tracker.measure("read_state"):
        robot_state_data = self.interface.get_low_state()
        # Shape: (1, 4*N+24)

    # ===== Stage 2: Mode Handling =====
    with latency_tracker.measure("preprocessing"):
        if self.get_ready_state:
            # Initialization mode: interpolate to default pose
            progress = self.ready_step / 180.0  # 3.6 seconds @ 50Hz
            q_current = robot_state_data[:, 7:7+self.num_joints]
            q_default = self.default_dof_angles
            q_target = q_current + (q_default - q_current) * progress

            self.ready_step += 1
            if self.ready_step >= 180:
                self.get_ready_state = False
                print("Ready! Press ] to start policy")

        elif not self.use_policy_action:
            # Manual control mode: hold current pose
            q_target = robot_state_data[:, 7:7+self.num_joints]

        else:
            # Policy mode: will run inference
            pass

    # ===== Stage 3: Inference =====
    if self.use_policy_action:
        with latency_tracker.measure("inference"):
            scaled_action = self.rl_inference(robot_state_data)
            # scaled_action shape: (1, N)

    # ===== Stage 4: Postprocessing =====
    with latency_tracker.measure("postprocessing"):
        if self.use_policy_action:
            # Convert action to target position
            q_target = scaled_action + self.default_dof_angles

        # Clip to joint limits (safety)
        q_target = np.clip(q_target, self.joint_pos_min, self.joint_pos_max)

    # ===== Stage 5: Send Command =====
    with latency_tracker.measure("action_pub"):
        q_current = robot_state_data[:, 7:7+self.num_joints]
        self.interface.send_low_command(
            cmd_q=q_target[0],
            cmd_dq=np.zeros(self.num_joints),  # Target velocity = 0
            cmd_tau=np.zeros(self.num_joints),  # Feedforward torque = 0
            dof_pos_latest=q_current[0],
        )
```

### RL Inference

```python
def rl_inference(self, robot_state_data: np.ndarray) -> np.ndarray:
    """Run policy inference: observations → actions"""

    # 1. Prepare observations
    obs = self.prepare_obs_for_rl(robot_state_data)
    # obs = {"actor_obs": np.array((1, obs_dim))}

    # 2. Run ONNX inference
    policy_action = self.policy(obs)
    # policy_action shape: (1, num_joints)

    # 3. Clip actions (safety)
    policy_action = np.clip(policy_action, -100, 100)

    # 4. Scale actions
    scaled_action = policy_action * self.policy_action_scale

    # 5. Update action history (for next observation)
    self.last_action = scaled_action.copy()

    return scaled_action
```

### Observation Processing

```python
def prepare_obs_for_rl(self, robot_state_data: np.ndarray) -> dict:
    """Convert robot state to policy observation"""

    # ===== Step 1: Extract Observations =====
    obs_buffer = self.get_current_obs_buffer_dict(robot_state_data)
    # Returns dict with all observation terms

    # ===== Step 2: Scale Observations =====
    current_obs = self.parse_current_obs_dict(obs_buffer)
    # Applies obs_scales to each term

    # ===== Step 3: Update History =====
    group_obs = self._update_obs_history(current_obs)
    # Maintains temporal history via deques

    # ===== Step 4: Return Policy Input =====
    return {"actor_obs": group_obs["actor_obs"]}
```

**Observation Extraction:**
```python
def get_current_obs_buffer_dict(self, robot_state_data: np.ndarray) -> dict:
    """Extract individual observations from robot state"""

    # Parse robot state array
    base_quat = robot_state_data[:, 3:7]
    base_ang_vel = robot_state_data[:, 39:42]
    joint_pos = robot_state_data[:, 7:7+self.num_joints]
    joint_vel = robot_state_data[:, 39+3:39+3+self.num_joints]

    # Compute derived observations
    projected_gravity = self.compute_projected_gravity(base_quat)
    dof_pos_rel = joint_pos - self.default_dof_angles  # Relative to default

    # Get commands
    command_lin_vel = self.lin_vel_command
    command_ang_vel = self.ang_vel_command

    # Get phase (for locomotion)
    sin_phase = np.sin(self.phase)
    cos_phase = np.cos(self.phase)

    # Assemble observation dict
    obs_buffer = {
        "base_ang_vel": base_ang_vel,
        "projected_gravity": projected_gravity,
        "dof_pos": dof_pos_rel,
        "dof_vel": joint_vel,
        "actions": self.last_action,  # History
        "command_lin_vel": command_lin_vel,
        "command_ang_vel": command_ang_vel,
        "sin_phase": sin_phase,
        "cos_phase": cos_phase,
    }

    return obs_buffer
```

**Projected Gravity Computation:**
```python
def compute_projected_gravity(self, base_quat: np.ndarray) -> np.ndarray:
    """
    Compute gravity vector in robot body frame

    This is a key observation for balance control:
    - In standing pose: [0, 0, -1] (pointing down in body frame)
    - When tilted forward: [sin(pitch), 0, -cos(pitch)]
    - Provides orientation feedback without gimbal lock
    """
    gravity_world = np.array([[0.0, 0.0, -1.0]])  # World frame
    projected = quat_rotate_inverse(base_quat, gravity_world)
    return projected
```

### LocomotionPolicy (`policies/locomotion.py`)

Extends BasePolicy for velocity tracking locomotion.

**Key Addition: Gait Phase Management**
```python
class LocomotionPolicy(BasePolicy):

    def _init_phase_components(self):
        """Initialize gait phase tracking"""
        self.gait_period = 1.0  # seconds
        self.phase = np.array([[0.0, np.pi]])  # Per-foot phase offset
        # Left foot: 0.0, Right foot: π (opposite phase)

        self.phase_time = 0.0
        self.dt = 1.0 / self.rl_rate  # Time step

    def update_phase_time(self):
        """Update gait phase (called every control loop)"""
        # Increment phase
        self.phase_time += self.dt

        # Wrap to period
        if self.phase_time >= self.gait_period:
            self.phase_time -= self.gait_period

        # Compute per-foot phase
        phase_progress = 2 * np.pi * (self.phase_time / self.gait_period)
        self.phase = np.array([[
            phase_progress,          # Left foot
            phase_progress + np.pi   # Right foot (offset by π)
        ]])
```

**Gait Phase Intuition:**
- `sin(phase)` and `cos(phase)` provide **smooth periodic signal**
- Policy learns to synchronize leg motion with phase
- Opposite phase for left/right feet → alternating gait
- Phase resets periodically → stable periodic behavior

---

## Communication Layer

### Booster SDK Architecture

The Booster SDK uses **DDS (Data Distribution Service)** for communication, specifically **CycloneDDS** implementation.

**Key Concepts:**

1. **Domain:** Logical partition of network (domain_id=0 by default)
2. **Topics:** Named channels (e.g., "LowCmd", "LowState")
3. **Publishers:** Send messages on topics
4. **Subscribers:** Receive messages from topics
5. **Quality of Service (QoS):** Reliability, latency, history settings

**Initialization Pattern:**
```python
from booster_robotics_sdk import ChannelFactory, get_interface_ip

# 1. Initialize channel factory (must be first!)
ip = get_interface_ip("eth0")  # Get IP of network interface
ChannelFactory.Instance().Init(domain_id=0, ip)

# 2. Create publisher
from booster_robotics_sdk import B1LowCmdPublisher
cmd_publisher = B1LowCmdPublisher()
cmd_publisher.InitChannel()

# 3. Create subscriber
from booster_robotics_sdk import B1LowStateSubscriber
def state_callback(msg):
    process_state(msg)
state_subscriber = B1LowStateSubscriber(state_callback)
state_subscriber.InitChannel()

# 4. Create client (for mode switching)
from booster_robotics_sdk import B1LocoClient, RobotMode
client = B1LocoClient()
client.Init()
client.ChangeMode(RobotMode.kCustom)
```

### Message Types

**LowCmd (Commands):**
```python
class LowCmd:
    cmd_type: LowCmdType  # SERIAL or PARALLEL
    motor_cmd: List[MotorCmd]  # One per motor

class MotorCmd:
    q: float      # Target position (rad)
    dq: float     # Target velocity (rad/s)
    tau: float    # Feedforward torque (Nm)
    kp: float     # Position gain
    kd: float     # Velocity gain
```

**Robot Firmware PD Control:**
```python
# Robot executes this at 500Hz:
for i in range(num_motors):
    cmd = motor_cmd[i]
    state = motor_state[i]

    # PD control law
    error_pos = cmd.q - state.q
    error_vel = cmd.dq - state.dq
    tau_output = cmd.kp * error_pos + cmd.kd * error_vel + cmd.tau

    # Send to motor driver
    set_motor_torque(i, tau_output)
```

**LowState (Feedback):**
```python
class LowState:
    imu_state: IMUState
    motor_state_serial: List[MotorState]  # Serial mode
    motor_state_parallel: List[MotorState]  # Parallel mode

class IMUState:
    rpy: List[float]      # Roll, pitch, yaw (rad)
    gyro: List[float]     # Angular velocity (rad/s)
    acc: List[float]      # Linear acceleration (m/s²)

class MotorState:
    q: float              # Position (rad)
    dq: float             # Velocity (rad/s)
    tau_est: float        # Estimated torque (Nm)
```

### Communication Timing

```
Policy Thread (50 Hz)              Robot Firmware (500 Hz)
─────────────────────              ────────────────────────

[t=0ms]  compute_action()
[t=1ms]  send_low_command() ──┐
                               │
[t=20ms] compute_action()      │    [t=2ms]  receive command
[t=21ms] send_low_command() ──┼──> [t=2-22ms] PD control @ 500Hz
                               │    (10 control cycles)
[t=40ms] compute_action()      │    [t=2ms]  publish state
[t=41ms] send_low_command() ──┘    [t=4ms]  publish state
                                    [t=6ms]  publish state
         ↑                          ...
         │                          [t=22ms] receive command
    get_low_state()                 [t=22-42ms] PD control @ 500Hz
    (cached @ 50Hz)                 ...
```

**Key Insight:** Policy provides **waypoints** at 50Hz, robot **interpolates** at 500Hz using PD control. This is why PD gains must match training!

---

## Control Flow

### Complete End-to-End Flow

```
[User Input]
  │
  ├─ Joystick: BoosterRemoteControlService polls evdev @ background thread
  │  ├─ Axis events → velocity commands (vx, vy, vyaw)
  │  └─ Button events → mode switches (start/stop/reset)
  │
  └─ Keyboard: pynput listener @ background thread
     └─ Key events → velocity/mode commands

[Main Loop @ 50Hz]
  │
  ├─ [1] Process Input
  │  └─ interface.process_joystick_input() or keyboard commands
  │     └─ Updates: lin_vel_command, ang_vel_command, use_policy_action
  │
  ├─ [2] Update Phase (if locomotion)
  │  └─ phase_time += dt; phase = [[phase_progress, phase_progress + π]]
  │
  ├─ [3] Policy Action
  │  │
  │  ├─ [3.1] Read State
  │  │  └─ interface.get_low_state()
  │  │     └─ state_processor.get_state()  [cached from callback @ 500Hz]
  │  │
  │  ├─ [3.2] Prepare Observations
  │  │  └─ extract_obs(robot_state) → {ang_vel, gravity, dof_pos, ...}
  │  │     ├─ Compute projected_gravity(base_quat)
  │  │     ├─ Relativize: dof_pos -= default_dof_angles
  │  │     ├─ Get commands: lin_vel_command, ang_vel_command
  │  │     ├─ Get phase: sin(phase), cos(phase)
  │  │     └─ Stack with history → obs_dict
  │  │
  │  ├─ [3.3] Inference
  │  │  └─ policy_session.run(None, obs_dict) → actions
  │  │     └─ ONNX runtime (CPU or GPU)
  │  │
  │  ├─ [3.4] Postprocess Actions
  │  │  ├─ Clip: actions = clip(actions, -100, 100)
  │  │  ├─ Scale: actions *= policy_action_scale
  │  │  └─ Offset: q_target = actions + default_dof_angles
  │  │
  │  └─ [3.5] Send Command
  │     └─ interface.send_low_command(q_target, dq=0, tau=0)
  │        └─ command_sender.send_command()
  │           ├─ Create LowCmd message
  │           ├─ Fill motor_cmd[i] = {q, dq, tau, kp, kd}
  │           └─ lowcmd_publisher.Write(low_cmd)  [DDS publish]
  │
  └─ [4] Rate Limiting
     └─ rate.sleep() → sleep until t = 20ms (for 50Hz)

[Robot Firmware @ 500Hz]
  │
  ├─ Receive LowCmd (from DDS)
  │
  ├─ For each motor:
  │  └─ tau = kp*(cmd.q - state.q) + kd*(cmd.dq - state.dq) + cmd.tau
  │     └─ set_motor_torque(tau)
  │
  ├─ Read Sensors (IMU, encoders)
  │
  └─ Publish LowState (to DDS)
     └─ Triggers state_processor callback
```

---

## Key Design Patterns

### 1. Factory Pattern (SDK Abstraction)

**Problem:** Need to support multiple robot SDKs with different interfaces.

**Solution:** Factory functions create robot-specific implementations behind unified interface.

```python
# sdk/__init__.py
def create_command_sender(config: RobotConfig):
    if config.sdk_type == "booster":
        return BoosterCommandSender(config)
    elif config.sdk_type == "unitree":
        return UnitreeCommandSender(config)
    else:
        raise ValueError(f"Unknown SDK: {config.sdk_type}")

def create_state_processor(config: RobotConfig):
    if config.sdk_type == "booster":
        return BoosterStateProcessor(config)
    elif config.sdk_type == "unitree":
        return UnitreeStateProcessor(config)
    else:
        raise ValueError(f"Unknown SDK: {config.sdk_type}")
```

**Benefits:**
- Policy code is robot-agnostic
- Adding new robot = implement two classes (sender + processor)
- No runtime overhead (factory called once at initialization)

### 2. Observer Pattern (State Updates)

**Problem:** Robot state arrives at high rate (500Hz), policy only needs low rate (50Hz).

**Solution:** Subscriber callback pattern with cached state.

```python
# State arrives via callback @ 500Hz
class StateProcessor:
    def __init__(self):
        self.cached_state = None
        subscriber = B1LowStateSubscriber(self.callback)
        subscriber.InitChannel()

    def callback(self, msg):
        """Called by SDK thread @ 500Hz"""
        self.cached_state = self.parse(msg)

    def get_state(self):
        """Called by policy thread @ 50Hz"""
        return self.cached_state  # Lock-free read!
```

**Benefits:**
- Policy never blocks waiting for state
- Automatic rate conversion (500Hz → 50Hz)
- Thread-safe without locks (atomic pointer read)

### 3. Strategy Pattern (Policy Types)

**Problem:** Different tasks need different policy behaviors (locomotion vs manipulation).

**Solution:** Inherit from BasePolicy and override specific methods.

```python
class BasePolicy:
    def prepare_obs_for_rl(self, robot_state):
        """Default observation processing"""
        ...

class LocomotionPolicy(BasePolicy):
    def prepare_obs_for_rl(self, robot_state):
        """Add gait phase to observations"""
        obs = super().prepare_obs_for_rl(robot_state)
        obs["sin_phase"] = np.sin(self.phase)
        obs["cos_phase"] = np.cos(self.phase)
        return obs

class WholeBodyTrackingPolicy(BasePolicy):
    def prepare_obs_for_rl(self, robot_state):
        """Add motion clip reference to observations"""
        obs = super().prepare_obs_for_rl(robot_state)
        obs["motion_command"] = self.get_motion_clip()
        return obs
```

**Benefits:**
- Code reuse via inheritance
- Easy to add new policy types
- Composition of behaviors

### 4. Dependency Injection (Configuration)

**Problem:** Hardcoded values scatter across codebase, hard to maintain.

**Solution:** All parameters defined in config, injected at initialization.

```python
# Bad: Hardcoded
class Policy:
    def __init__(self):
        self.rl_rate = 50  # Hardcoded!
        self.action_scale = 0.25  # Hardcoded!

# Good: Dependency injection
class Policy:
    def __init__(self, config: InferenceConfig):
        self.rl_rate = config.task.rl_rate
        self.action_scale = config.task.policy_action_scale
```

**Benefits:**
- Single source of truth (config files)
- Easy to experiment (change config, not code)
- Type-safe validation (Pydantic)

### 5. Template Method (Control Loop)

**Problem:** All policies share same control loop structure, but differ in details.

**Solution:** BasePolicy defines loop structure, subclasses fill in specific steps.

```python
class BasePolicy:
    def run(self):
        """Template method - defines control loop structure"""
        for it in itertools.count():
            self.process_input()      # Hook 1
            self.update_state()        # Hook 2
            self.policy_action()       # Hook 3 (calls inference)
            self.post_step()           # Hook 4
            self.rate.sleep()

    def update_state(self):
        """Override in subclass"""
        pass

class LocomotionPolicy(BasePolicy):
    def update_state(self):
        """Concrete implementation: update gait phase"""
        self.update_phase_time()
```

---

## Implementation Details

### ONNX Runtime Optimization

**Why ONNX?**
- Cross-platform (Linux, Windows, MacOS)
- Hardware-agnostic (CPU, GPU, NPU)
- Optimized kernels (faster than PyTorch inference)
- Small binary size (~10MB vs >1GB for PyTorch)

**Exporting from PyTorch:**
```python
import torch

# Load trained policy
policy = torch.jit.load("policy.pt")
policy.eval()

# Create dummy input
dummy_obs = {"actor_obs": torch.zeros(1, 100)}

# Export to ONNX
torch.onnx.export(
    policy,
    (dummy_obs,),
    "policy.onnx",
    input_names=["actor_obs"],
    output_names=["actions"],
    dynamic_axes={"actor_obs": {0: "batch"}, "actions": {0: "batch"}},
    opset_version=17,
)
```

**Adding Metadata:**
```python
import onnx

model = onnx.load("policy.onnx")

# Store PD gains in metadata
meta_kp = model.metadata_props.add()
meta_kp.key = "motor_kp"
meta_kp.value = ",".join(map(str, kp_gains))

meta_kd = model.metadata_props.add()
meta_kd.key = "motor_kd"
meta_kd.value = ",".join(map(str, kd_gains))

onnx.save(model, "policy.onnx")
```

**Runtime Loading:**
```python
import onnxruntime as ort

session = ort.InferenceSession("policy.onnx")

# Extract metadata
metadata = session.get_modelmeta().custom_metadata_map
kp = np.array([float(x) for x in metadata["motor_kp"].split(",")])

# Run inference
obs = {"actor_obs": np.zeros((1, 100), dtype=np.float32)}
action = session.run(None, obs)[0]
```

### Latency Tracking

**Implementation:**
```python
class LatencyTracker:
    def __init__(self):
        self.stage_times = defaultdict(list)
        self.cycle_start = None

    def start_cycle(self):
        self.cycle_start = time.perf_counter()

    @contextmanager
    def measure(self, stage_name: str):
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        self.stage_times[stage_name].append(elapsed)

    def get_stats_str(self):
        lines = []
        for stage, times in self.stage_times.items():
            mean = np.mean(times[-50:])  # Last 50 samples
            std = np.std(times[-50:])
            lines.append(f"{stage}: {mean:.2f}±{std:.2f}ms")
        return "\n".join(lines)
```

**Usage:**
```python
with latency_tracker.measure("inference"):
    action = policy(obs)

# Output:
# inference: 1.23±0.15ms
# read_state: 0.45±0.08ms
# action_pub: 0.32±0.05ms
```

### Rate Limiting

**Naive Approach (Incorrect):**
```python
# BAD: Accumulates drift
dt = 1.0 / 50  # 20ms
while True:
    work()
    time.sleep(dt)  # Sleeps for dt, but work() also takes time!
```

**Correct Approach:**
```python
class RateLimiter:
    def __init__(self, rate_hz: float):
        self.period = 1.0 / rate_hz
        self.next_time = time.perf_counter()

    def sleep(self):
        now = time.perf_counter()
        sleep_time = self.next_time - now

        if sleep_time > 0:
            time.sleep(sleep_time)

        # Schedule next wake-up (accumulates correctly)
        self.next_time += self.period

        # Reset if we fall too far behind
        if self.next_time < now:
            self.next_time = now
```

**Key Insight:** `next_time` accumulates period, not actual time. This prevents drift even if individual iterations vary in duration.

---

## Comparison with Colosseum

| Aspect | Holosoma Inference | Colosseum Deployment |
|--------|-------------------|----------------------|
| **Architecture** | 3-layer (Policy, Interface, SDK) | 2-layer (Policy, Controller) |
| **Configuration** | Pydantic + Tyro (declarative) | Dataclasses (manual) |
| **PD Gains Source** | **ONNX metadata (single source of truth)** | Config files (duplicated) |
| **Robot Support** | Multi-robot (Booster, Unitree, ROS2) | Single robot (Booster T1) |
| **SDK Abstraction** | InterfaceWrapper + Factory Pattern | Direct SDK usage |
| **Observation Processing** | Config-driven (ObservationConfig) | Hardcoded in policy |
| **Policy Loading** | ONNX with metadata | ONNX or PyTorch |
| **Input Handling** | Keyboard + Joystick (evdev) | Keyboard + Joystick |
| **Gain Management** | Runtime adjustment (kp_level, kd_level) | Static (from config) |
| **Latency Tracking** | Per-stage breakdown | Total only |
| **Safety Features** | Gain scaling, position limits, mode switching | Position limits only |
| **Documentation** | Extensive (configs, CLI help) | Minimal (code comments) |

**Strengths of Holosoma Approach:**
1. **PD Gains in ONNX Metadata:** Single source of truth (training → ONNX → deployment), no duplication
2. **Robot-agnostic:** Easy to port to new robots
3. **Type-safe:** Pydantic catches config errors
4. **Observable:** Detailed latency tracking
5. **Flexible:** Runtime gain adjustment for safety
6. **Ergonomic:** Tyro CLI with auto-generated help

**Areas for Colosseum to Adopt:**
1. ⭐ **PD gains in ONNX metadata** (eliminates duplication, ensures consistency)
2. ✅ Config-driven observation processing
3. ✅ Factory pattern for SDK abstraction
4. ✅ Runtime gain adjustment
5. ✅ Per-stage latency tracking
6. ✅ Joystick support via evdev
7. 🔲 Multi-robot support (future)

---

## Summary

### Key Takeaways

1. **Robot-Agnostic Design is Paramount**
   - Abstract vendor SDKs behind unified interfaces
   - Use factory pattern for extensibility
   - Policy code should never import robot-specific modules

2. **Configuration Over Code**
   - All tunable parameters in configs (RobotConfig, ObservationConfig, TaskConfig)
   - Use type-safe validation (Pydantic)
   - Ergonomic CLI with Tyro

3. **Asynchronous Communication is Essential**
   - Robot state @ 500Hz, policy @ 50Hz → use callbacks + caching
   - Avoid blocking I/O in control loop
   - Thread safety via atomic reads, not locks

4. **Training-Deployment Consistency is Critical (ONNX Metadata Pattern)**
   - **Store PD gains in ONNX metadata** (single source of truth)
   - Training configs define gains → exported to ONNX → loaded at deployment
   - Eliminates config duplication and version mismatch issues
   - Observation processing must be identical
   - Action scaling stored in metadata
   - All training hyperparameters travel with the model

5. **Safety is Engineered, Not Assumed**
   - Runtime gain adjustment for instability mitigation
   - Position/velocity/torque limits enforced
   - Multiple control modes (init, manual, policy)
   - Emergency stop via keyboard/joystick

### Deployment Checklist

For successful deployment, ensure:
- [ ] PD gains match training (stored in ONNX metadata)
- [ ] Observation space identical (dims, scales, names)
- [ ] Action scaling matches training
- [ ] Default joint pose same
- [ ] Control frequency same (50 Hz typical)
- [ ] Network latency <5ms
- [ ] Policy tested in sim-to-sim first
- [ ] Gains start conservative (0.5x), increase gradually

---

**This document provides the complete technical understanding of holosoma_inference. Use it as reference when implementing or debugging deployment pipelines in Colosseum.**
