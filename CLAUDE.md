# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Colosseum** is a learning playground for humanoid robot research focused on robot soccer. It builds on top of **mjlab**, a GPU-accelerated reinforcement learning framework for robotics built on MuJoCo Warp.

Key characteristics:
- GPU-accelerated parallel training (4096+ environments)
- Manager-based declarative configuration (rewards, observations, actions)
- MuJoCo-native physics simulation
- Primary robot: Booster T1 humanoid (12 DOF locomotion / 23 DOF full body)

## Development Setup

### Environment Management

This project uses **Pixi** (a conda/pypi workspace manager) for environment and dependency management:

```bash
# Install dependencies and activate environment
pixi install

# Serve documentation locally
pixi run docs

# Build documentation
pixi run build-docs
```

The project uses:
- Python 3.12 (strict version)
- NVIDIA GPU required for training (MuJoCo Warp)
- `MUJOCO_GL=osmesa` environment variable (set automatically by Pixi)

**IMPORTANT: Always use `pixi run python` for Python commands:**
```bash
# Correct
pixi run python script.py

# Incorrect
python script.py  # Will fail with ModuleNotFoundError
```

This ensures the correct environment with all dependencies (mjlab, booster_robotics_sdk, etc.) is active.

### Code Quality

Ruff is configured with:
- Source directory: `src`
- Indent width: 2 spaces

## Architecture

### Three-Layer Structure

Colosseum follows mjlab's three-layer architecture:

1. **Simulation Layer** (lowest): GPU-accelerated physics via MuJoCo Warp
   - Batched parallel execution across thousands of environments
   - Direct `mjModel`/`mjData` access

2. **Scene Layer** (middle): Physical object organization
   - Entities (robots, terrain, obstacles)
   - Name → index mapping and resolution
   - Entity data access (poses, velocities, forces)

3. **Task Layer** (highest): Reinforcement learning MDP definition
   - Manager-based orchestration (Actions, Observations, Rewards, Terminations, Events)
   - Declarative term configuration
   - `gym.Env` interface for RL training

### Term-Based Pattern

The core design pattern across all layers:

1. **Declaration** (config time): Define terms declaratively using `*TermCfg` classes
2. **Resolution** (initialization): Map entity/component names → MuJoCo indices once
3. **Execution** (runtime): Fast computation using pre-resolved indices

Example term configuration:
```python
rewards = {
    "stand_reward": RewardTermCfg(
        func=mdp.body_height_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso")}
    )
}
```

During initialization, `SceneEntityCfg.resolve()` converts `body_names="torso"` → `body_ids=[3]` (global MuJoCo index). At runtime, term functions use these pre-resolved indices for efficient tensor operations.

### Three-Layer MDP Architecture (Training/Deployment Sharing)

Colosseum implements a novel three-layer architecture for MDP functions (observations, rewards, etc.) that eliminates code duplication between training and deployment:

**Layer 1: Universal MDP Functions** (`src/colosseum/mdp/`)
- Robot-agnostic, physics-based pure functions
- Examples: `compute_projected_gravity()`, `exponential_reward_kernel()`
- Dependencies: Only torch and math utilities
- Used by: All robots and tasks, both training and deployment

**Layer 2: Robot-Specific MDP Functions** (`src/colosseum/robots/*/mdp/`)
- Functions specific to one robot platform
- Examples: `compute_foot_contact_state()` (T1-specific sensor processing)
- Dependencies: Layer 1 functions, robot constants
- Used by: Multiple tasks using the same robot

**Layer 3: Task-Specific MDP Functions** (`src/colosseum/tasks/*/mdp/`)
- Functions specific to one task, split into two modules:
  - `observations.py`: **Pure functions** (shared between training and deployment)
  - `wrappers.py`: **Training wrappers** (mjlab interface adapters, training-only)
- Examples: `compute_velocity_commands()`, `compute_joint_pos_rel()`
- Dependencies: Layer 1 and Layer 2 functions
- Used by: Training configs (via wrappers) AND deployment policies (via pure functions)

**Key Benefits:**
- ✅ **Single Source of Truth**: Observation logic defined once in pure functions
- ✅ **Guaranteed Consistency**: Training and deployment use identical computation
- ✅ **Easy Maintenance**: Change observation → works everywhere automatically
- ✅ **Testable**: Pure functions are independently unit-testable
- ✅ **Clear Separation**: Computation (pure functions) vs. integration (wrappers)

**Example Flow:**
```python
# Layer 1: Universal
def compute_projected_gravity(quat: Tensor) -> Tensor:
    # Generic quaternion rotation (works for any robot)
    ...

# Layer 3: Task-specific pure function
def compute_base_ang_vel(robot_data) -> Tensor:
    return robot_data.root_ang_vel_b  # Works in training AND deployment

# Layer 3: Training wrapper (training-only)
def base_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Tensor:
    robot = env.scene[asset_cfg.name]
    return compute_base_ang_vel(robot.data)  # Calls pure function

# Deployment: Direct use of pure function
class VelocityPolicy(Policy):
    def compute_observation(self):
        obs = compute_base_ang_vel(self.robot.data)  # Same pure function!
```

See `docs/SHARED_OBSERVATIONS.md` for detailed implementation guide.

## Directory Structure

```
src/colosseum/
├── mdp/              # Layer 1: Universal MDP functions (robot-agnostic)
│   ├── observations.py  # Pure observation functions (e.g., projected_gravity)
│   └── rewards.py       # Pure reward functions (e.g., exponential_kernel)
│
├── robots/           # Robot definitions and constants
│   ├── booster_t1/   # Booster T1 humanoid
│   │   ├── xmls/     # MuJoCo XML models
│   │   │   ├── T1_12dof.xml   # 12-DOF locomotion model (DEPRECATED)
│   │   │   └── T1_23dof.xml   # 23-DOF full body model (unified)
│   │   ├── t1_actuators.py    # Motor specs and actuator configs
│   │   ├── t1_contacts.py     # Collision and contact sensor configs
│   │   ├── t1_constants.py    # Spec loaders and entity configs (training)
│   │   ├── mdp/               # Layer 2: T1-specific MDP functions
│   │   │   └── observations.py  # T1-specific observations (e.g., foot_contact)
│   │   └── deploy/            # Deployment configurations
│   │       └── robot_cfg.py     # T1 deployment configs (12/23 DOF)
│   └── cartpole/     # CartPole balancing demo
│
├── tasks/            # Task definitions (training + deployment shared code)
│   ├── cartpole/     # CartPole balancing task
│   │   ├── cartpole_scene.py  # Scene configuration
│   │   ├── cartpole_task.py   # MDP (actions, obs, rewards, terminations, events)
│   │   └── mdp_functions.py   # Custom term functions
│   └── velocity/     # Velocity tracking task
│       ├── config/   # Training configurations (robot-specific)
│       │   └── t1/
│       │       └── env_cfgs.py  # T1 velocity env config
│       ├── mdp/      # Layer 3: Task-specific MDP functions
│       │   ├── observations.py  # Pure observation functions (SHARED)
│       │   └── wrappers.py      # Training wrappers (mjlab-only)
│       └── rl/       # RL algorithm configs
│
├── deploy/           # Deployment infrastructure
│   ├── core/
│   │   ├── controllers/  # Base controller abstractions
│   │   └── utils/
│   │       ├── registry.py         # Task registry
│   │       └── robot_registry.py   # Robot registry (NEW)
│   └── tasks/        # Task deployment implementations
│       └── velocity/
│           ├── policy.py    # Robot-agnostic velocity policy
│           └── configs.py   # Example deployment configs
│
├── play/             # Evaluation/playback utilities
└── utils/            # Shared utilities (path helpers)

external/             # Git submodules (not managed by colosseum)
├── mjlab/            # Core RL framework
└── holosoma/         # Motion retargeting (optional)

tests/                # Test files
docs/                 # MkDocs documentation
```

## Creating Tasks

Tasks are registered via the `mjlab.tasks` entry point in `pyproject.toml`:

```toml
[project.entry-points."mjlab.tasks"]
cartpole = "colosseum.tasks.cartpole"
```

Each task module must:
1. Define scene configuration (`SceneCfg`)
2. Define MDP components (actions, observations, rewards, terminations, events)
3. Create `ManagerBasedRlEnvCfg` combining all components
4. Call `register_mjlab_task()` with a unique `task_id`

See `src/colosseum/train/tasks/cartpole/` for a complete minimal example.

## Working with Robots

### Booster T1 Humanoid

Located in `src/colosseum/robots/booster_t1/`:

#### XML Models

- **12 DOF model** ([T1_12dof.xml](src/colosseum/robots/booster_t1/xmls/T1_12dof.xml)): Legs only (6 DOF per leg), used for locomotion training
- **23 DOF model** ([T1_23dof.xml](src/colosseum/robots/booster_t1/xmls/T1_23dof.xml)): Full body including arms, waist, and neck, for deployment

#### Configuration Modules

The T1 configuration is organized into three modules:

**[t1_actuators.py](src/colosseum/robots/booster_t1/t1_actuators.py)**: Motor specifications and actuator configurations
- `MOTOR_SPECS`: Dictionary of motor specifications from manufacturer data (gear ratio, torque, speed, inertia)
- `compute_pd_gains()`: Computes PD controller gains using Unitree G1 method (natural frequency + damping ratio)
- Actuator configs for 12-DOF locomotion:
  - `T1_ACTUATOR_HIP_PITCH`, `T1_ACTUATOR_HIP_ROLL`, `T1_ACTUATOR_HIP_YAW`
  - `T1_ACTUATOR_KNEE`
  - `T1_ACTUATOR_ANKLE_PITCH`, `T1_ACTUATOR_ANKLE_ROLL`
- Actuator configs for 23-DOF full body:
  - `T1_ACTUATOR_NECK`, `T1_ACTUATOR_ARM`, `T1_ACTUATOR_WAIST`

**[t1_contacts.py](src/colosseum/robots/booster_t1/t1_contacts.py)**: Collision and contact sensor configurations
- Collision configs (modify geom properties):
  - `FEET_ONLY_COLLISION`: Only foot geoms collide (recommended for training)
  - `FULL_COLLISION_WITHOUT_SELF`: All parts collide with environment, no self-collision
  - `FULL_COLLISION`: Full collision including self-collision (most realistic)
  - `HANDS_FEET_COLLISION`: Only hands and feet collide (for manipulation)
- Contact sensor configs (for observation/reward):
  - `FEET_GROUND_CONTACT_SENSOR`: Tracks foot-ground contact with air time
  - `SELF_COLLISION_SENSOR`: Detects self-collisions
  - `HAND_CONTACT_SENSOR`: Tracks hand contact for manipulation
- `T1_FOOT_GEOM_NAMES`: Tuple of all foot geometry names for events

**[t1_constants.py](src/colosseum/robots/booster_t1/t1_constants.py)**: Spec loaders and entity configurations
- XML paths: `T1_12DOF_XML`, `T1_23DOF_XML`
- Spec loaders:
  - `get_t1_12dof_spec()`: Returns MjSpec for 12-DOF locomotion model
  - `get_t1_23dof_spec()`: Returns MjSpec for 23-DOF full body model
- Pre-configured entity configs:
  - `T1_12DOF_ENTITY_CFG`: Complete entity config for locomotion training
  - `T1_23DOF_ENTITY_CFG`: Complete entity config for full body deployment

#### PD Gain Computation

T1 uses the Unitree G1 method for computing actuator gains:

```python
# Natural frequency and damping ratio (same as G1)
natural_freq = 10.0 * 2π  # 10Hz in rad/s
damping_ratio = 2.0  # Overdamped (prevents oscillations)

# Compute gains from motor reflected inertia
stiffness = reflected_inertia × ω_n²
damping = 2 × ζ × reflected_inertia × ω_n
```

This ensures stable, overdamped control appropriate for each joint's mechanical properties.

### Entity Configuration

#### Using Pre-Configured Entities

The simplest way to use T1 is with the pre-configured entity configs:

```python
from colosseum.robots.booster_t1.t1_constants import T1_12DOF_ENTITY_CFG

# Use directly in scene config
entities = {"robot": T1_12DOF_ENTITY_CFG}
```

These configs include all actuators, collision settings, and sensors with manufacturer-accurate specifications.

#### Custom Entity Configuration

Entities are configured using `EntityCfg`:
- `spec_fn`: Function returning `mujoco.MjSpec` (e.g., `get_t1_12dof_spec`)
- `init_state`: Initial joint positions/velocities
- `actuators`: Dict of actuator configs applying to joint patterns
- `articulation_info`: Optional collision/armature settings using `EntityArticulationInfoCfg`:
  - `collision`: `CollisionCfg` object (e.g., `FEET_ONLY_COLLISION`)
  - `armature`: Joint armature values (reflected inertia)
- `sensors`: List of sensor configs (e.g., `[FEET_GROUND_CONTACT_SENSOR]`)

## Documentation

Documentation is built with MkDocs Material and includes:
- Colosseum-specific guides (Getting Started, CartPole tutorial)
- mjlab architecture documentation (3-layer system, term-based patterns)

The documentation covers:
- **Overview**: Architecture and design patterns
- **Task Layer**: Manager-based RL environment, term configuration
- **Scene Layer**: Entities, compilation, indexing
- **Simulation Layer**: MuJoCo Warp integration

## Key Concepts

### SceneEntityCfg Resolution

`SceneEntityCfg` is the bridge between managers and entities:
- Stores entity name + component patterns (e.g., `body_names=".*_knee"`)
- During `resolve()`, converts patterns to global MuJoCo indices
- Passed to term functions at runtime with pre-resolved indices

### MuJoCo Compilation Flow

1. Create `Scene` with entity configs
2. Each entity loads its `MjSpec` (XML representation)
3. Scene merges all specs and compiles → `MjModel` (MuJoCo assigns global indices)
4. Initialize entities with compiled model → creates `EntityIndexing`
5. Resolve `SceneEntityCfg` objects to map names → indices

### Custom Term Functions

Term functions follow this signature:
```python
def my_reward(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    data = entity.data.body_pos_w[:, asset_cfg.body_ids]  # Pre-resolved indices
    return compute_reward(data)
```

## Deployment System Architecture

Colosseum's deployment system provides a modular, robot-agnostic framework for deploying trained policies to both MuJoCo simulation (sim-to-sim testing) and real hardware.

**Architecture Documentation:**
- See `deployment_docs/DEPLOYMENT_FIX_COMPLETE.md` for recent stability fixes
- See `deployment_docs/GROUND_CONTACT_FIX.md` for initialization troubleshooting

### Directory Structure

```
src/colosseum/deploy/
├── config/                # Pydantic frozen dataclasses
│   ├── controller.py      # ControllerConfig (top-level orchestration)
│   ├── robot.py           # RobotConfig, PrepareStateConfig
│   ├── policy.py          # PolicyConfig, VelocityCommandConfig
│   └── backend.py         # MujocoConfig, BoosterConfig
│
├── core/                  # Core abstractions
│   ├── base_controller.py # BaseController (lifecycle + control loop)
│   ├── robot.py           # BoosterRobot, RobotData (state management)
│   ├── policy.py          # Policy (abstract base class)
│   ├── observation_spec.py # ObservationSpec (training-deployment contract)
│   ├── command.py         # VelocityCommand
│   └── registry.py        # TaskRegistry (auto-registration)
│
├── backends/              # Backend implementations
│   ├── mujoco.py          # MujocoController (sim-to-sim testing)
│   └── booster.py         # BoosterController (real robot, stub)
│
└── input/                 # User input handling
    ├── base.py            # BaseInputSource, InputState
    ├── joystick.py        # JoystickInputSource (evdev gamepad)
    └── keyboard.py        # KeyboardInputSource (terminal)

tasks/velocity/deploy/t1_23dof/  # Task-specific deployment
├── __init__.py            # @register_task decorators
├── config.py              # ControllerConfig presets
├── policy.py              # T1VelocityPolicy (observation + inference)
└── models/                # ONNX policy checkpoints

robots/booster_t1/
└── deploy_config.py       # T1_23DOF_ROBOT_CFG (centralized robot specs)
```

### Configuration System (Immutable Composition)

All deployment configs are **frozen dataclasses** (immutable after creation) with strict type validation:

```python
@dataclass(frozen=True)
class ControllerConfig:
    """Top-level deployment configuration"""
    policy_dt: float = 0.02                    # 50Hz policy execution
    robot: RobotConfig                         # Robot hardware specs
    policy: PolicyConfig                       # Policy checkpoint and scaling
    vel_command: Optional[VelocityCommandConfig] = None
    input: Optional[InputConfig] = None
    mujoco: MujocoConfig = field(default_factory=MujocoConfig)
    booster: BoosterConfig = field(default_factory=BoosterConfig)

    @computed_field
    @property
    def physics_dt(self) -> float:
        """Physics timestep = policy_dt / decimation"""
        return self.policy_dt / self.mujoco.decimation
```

**Key Configuration Objects:**

**RobotConfig** (hardware specifications):
```python
RobotConfig(
    joint_names=(23,)           # Real hardware order
    sim_joint_names=(23,)       # MuJoCo compiled order
    body_names=(24,)            # Body names for sensors
    joint_stiffness=(23,)       # Kp gains (MUST match training!)
    joint_damping=(23,)         # Kd gains (MUST match training!)
    default_joint_pos=(23,)     # HOME_QPOS (standing pose)
    effort_limit=(23,)          # Peak torques per joint
    mjcf_path=Path              # Path to robot MJCF/XML
    prepare_state=PrepareStateConfig  # Safe initialization pose
)
```

**PolicyConfig** (policy loading and scaling):
```python
PolicyConfig(
    task_name="velocity"                    # Registry lookup key
    checkpoint_path=Path("policy.onnx")     # ONNX or .pt model
    action_scale=0.25                       # MUST match training!
    use_onnx=True                           # ONNX runtime by default
)
```

**MujocoConfig** (sim-to-sim backend settings):
```python
MujocoConfig(
    init_pos=(0.0, 0.0, 0.70)  # CRITICAL: base height (see fixes below)
    init_quat=(1.0, 0.0, 0.0, 0.0)  # Identity quaternion (no rotation)
    decimation=4                     # 4 physics steps per policy step
    save_states=False                # State logging disabled by default
)
```

### Robot State Management

**RobotData** (runtime state container):
```python
class RobotData:
    """All joint-indexed tensors use REAL hardware order"""

    # Joint state (real hardware order)
    joint_pos: torch.Tensor         # (23,) joint positions [rad]
    joint_vel: torch.Tensor         # (23,) joint velocities [rad/s]
    feedback_torque: torch.Tensor   # (23,) measured torques [Nm]

    # Base state (IMU data in body frame)
    root_pos_w: torch.Tensor         # (3,) world position [m]
    root_quat_w: torch.Tensor        # (4,) world orientation (w,x,y,z)
    root_lin_vel_b: torch.Tensor     # (3,) linear velocity [m/s]
    root_ang_vel_b: torch.Tensor     # (3,) angular velocity [rad/s]
    projected_gravity_b: torch.Tensor # (3,) gravity vector in body frame

    # Joint order remapping (computed once at init)
    real2sim_joint_indexes: list[int]  # Maps hardware → policy order
    sim2real_joint_indexes: list[int]  # Maps policy → hardware order
```

**Joint Order Remapping:**

The T1 robot has **two different joint orderings**:
1. **Real hardware order** (`joint_names`): Hardware interface order
2. **MuJoCo simulation order** (`sim_joint_names`): Alphabetically sorted by MuJoCo

The system automatically computes bidirectional mappings:
```python
# Computed during RobotData.__init__
real2sim_joint_indexes = [cfg.joint_names.index(name)
                          for name in cfg.sim_joint_names]
sim2real_joint_indexes = [cfg.sim_joint_names.index(name)
                          for name in cfg.joint_names]

# Used in policy observation computation (remap to sim order)
joint_pos_sim = robot.data.joint_pos[robot.data.real2sim_joint_indexes]

# Used in action application (remap to hardware order)
joint_targets_real = action[robot.data.sim2real_joint_indexes]
```

### Policy Architecture

**Policy Base Class:**
```python
class Policy(ABC):
    """All deployment policies inherit from this"""

    def __init__(self, controller: BaseController):
        self.controller = controller
        self.robot = controller.robot
        self.config = controller.cfg.policy

        # Load ONNX or TorchScript model
        self._model = self._load_artifact(Path(self.config.checkpoint_path))

        # Compute per-joint action scaling (MUST match training!)
        # scale = action_scale * effort_limit / stiffness
        self.action_scale = (
            self.config.action_scale
            * self.robot.effort_limit
            / self.robot.joint_stiffness
        )

    @abstractmethod
    def reset(self) -> None:
        """Called at controller.start()"""

    @abstractmethod
    def inference(self) -> torch.Tensor:
        """Run policy, return joint targets in REAL hardware order"""
```

**T1VelocityPolicy Implementation:**
```python
class T1VelocityPolicy(Policy):
    def compute_observation(self) -> torch.Tensor:
        """Build observation matching VELOCITY_OBS_SPEC contract.

        CRITICAL: Remaps joint data from hardware order to simulation order!
        """
        real2sim = self.robot.data.real2sim_joint_indexes

        obs = torch.cat([
            self.robot.data.root_lin_vel_b,           # (3,) base lin vel
            self.robot.data.root_ang_vel_b,           # (3,) base ang vel
            self.robot.data.projected_gravity_b,      # (3,) proj gravity
            self.robot.data.joint_pos[real2sim],      # (23,) REMAPPED!
            self.robot.data.joint_vel[real2sim],      # (23,) REMAPPED!
            self.last_action,                         # (23,) in sim order
            self.vel_command.to_tensor(),             # (3,) velocity cmds
        ], dim=-1)  # Total: 3+3+3+23+23+23+3 = 82 for T1 23-DOF

        return obs.unsqueeze(0)  # (1, 82)

    def inference(self) -> torch.Tensor:
        """Execute policy and return joint targets.

        Flow: obs (sim order) → policy → action (sim order)
              → remap to hardware order → scale → add default pose
        """
        obs = self.compute_observation()      # (1, 82)
        action = self._model(obs).flatten()   # (23,) in sim order

        self.last_action = action  # Store for next step (keep in sim order)

        # Remap to hardware order and scale
        sim2real = self.robot.data.sim2real_joint_indexes
        joint_targets = (
            action[sim2real] * self.action_scale
            + self.robot.default_joint_pos
        )

        return joint_targets  # (23,) in REAL hardware order
```

### BaseController Execution Flow

**Lifecycle:**
```python
class BaseController(ABC):
    def start(self) -> None:
        """Initialize control session"""
        self._step_count = 0
        self._elapsed_s = 0.0
        self.is_running = True
        self.policy.reset()

    def policy_step(self) -> torch.Tensor:
        """Execute one policy inference step (50Hz)"""
        self._step_count += 1
        self._elapsed_s = self._step_count * self.cfg.policy_dt
        return self.policy.inference()

    def update_command(self) -> None:
        """Update velocity commands from joystick/keyboard input"""
        if self.input_source is None:
            return

        vx = self.input_source.get_vx_cmd()   # [-1, 1]
        vy = self.input_source.get_vy_cmd()
        vyaw = self.input_source.get_vyaw_cmd()

        # Scale by velocity limits
        self.vel_command.lin_vel_x = vx * self.vel_command.vx_max
        self.vel_command.lin_vel_y = vy * self.vel_command.vy_max
        self.vel_command.ang_vel_yaw = vyaw * self.vel_command.vyaw_max

    # Implemented by backends
    @abstractmethod
    def update_state(self) -> None:
        """Update robot.data from sensors/simulator"""

    @abstractmethod
    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        """Apply joint targets to actuators"""

    @abstractmethod
    def run(self) -> None:
        """Main control loop"""
```

**Typical Control Loop (MuJoCo Backend):**
```python
def run(self):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        self.update_state()  # Initial state
        self.start()         # Initialize policy

        while viewer.is_running() and self.is_running:
            # 1. Update robot state from MuJoCo sensors
            self.update_state()

            # 2. Update velocity commands from input
            self.update_command()

            # 3. Run policy inference (50Hz)
            joint_targets = self.policy_step()

            # 4. Apply controls (4x physics steps @ 200Hz)
            self.ctrl_step(joint_targets)

            # 5. Sync viewer and sleep
            viewer.sync()
            time.sleep(self.cfg.policy_dt)
```

### MuJoCo Backend (Sim-to-Sim Testing)

**Key Implementation Details:**

**Programmatic Actuator Creation:**
```python
class MujocoController(BaseController):
    def __init__(self, cfg: ControllerConfig):
        # Load MJCF and clear XML actuators
        spec = mujoco.MjSpec.from_file(self.robot.cfg.mjcf_path)
        spec.actuators.clear()

        # Add ground plane
        ground = spec.worldbody.add_geom()
        ground.type = mujoco.mjtGeom.mjGEOM_PLANE
        ground.friction[:] = [1.0, 0.005, 0.0001]

        # Add position actuators (one per joint, in MuJoCo order)
        for i, joint_name in enumerate(self.robot.cfg.sim_joint_names):
            actuator = spec.add_actuator()
            actuator.name = f"{joint_name}_actuator"
            actuator.joint = spec.find_joint(joint_name)
            actuator.gainprm[0] = self.robot.cfg.joint_stiffness[i]
            actuator.biasprm[2] = -self.robot.cfg.joint_damping[i]
            actuator.ctrlrange = [-1e9, 1e9]  # Position targets (unlimited)
            actuator.forcerange = [
                -self.robot.cfg.effort_limit[i],
                self.robot.cfg.effort_limit[i],
            ]

        # Compile and initialize
        self.mj_model = spec.compile()
        self.mj_model.opt.timestep = self.cfg.physics_dt  # 0.005s = 200Hz
        self.mj_data = mujoco.MjData(self.mj_model)

        # Set initial pose: [base_pos(3), base_quat(4), joint_pos(23)]
        self.mj_data.qpos[:] = np.concatenate([
            cfg.mujoco.init_pos,           # (0.0, 0.0, 0.70)
            cfg.mujoco.init_quat,          # (1.0, 0.0, 0.0, 0.0)
            self.robot.default_joint_pos,  # (23,) standing pose
        ])
```

**State Update (Sensors → RobotData):**
```python
def update_state(self) -> None:
    """Extract state from MuJoCo simulation"""
    # Joint state (skip first 7 qpos: 3 pos + 4 quat)
    self.robot.data.joint_pos = torch.from_numpy(self.mj_data.qpos[7:])
    self.robot.data.joint_vel = torch.from_numpy(self.mj_data.qvel[6:])

    # Base state from IMU sensors
    self.robot.data.root_pos_w = torch.from_numpy(self.mj_data.qpos[:3])
    self.robot.data.root_quat_w = torch.from_numpy(
        self.mj_data.sensor("orientation").data  # IMU quaternion
    )
    self.robot.data.root_lin_vel_b = torch.from_numpy(
        self.mj_data.sensor("imu_lin_vel").data
    )
    self.robot.data.root_ang_vel_b = torch.from_numpy(
        self.mj_data.sensor("imu_ang_vel").data
    )

    # Compute projected gravity from quaternion
    self.robot.data.projected_gravity_b = compute_projected_gravity(
        self.robot.data.root_quat_w
    )
```

**Control Step (Position Actuators):**
```python
def ctrl_step(self, joint_targets: torch.Tensor) -> None:
    """Apply joint position targets via MuJoCo position actuators.

    MuJoCo applies internal PD control:
        τ = Kp * (target - qpos) - Kd * qvel
        τ = clamp(τ, -effort_limit, effort_limit)
    """
    targets = joint_targets.cpu().numpy()

    # Run decimation steps (4x physics @ 200Hz per 1x policy @ 50Hz)
    for _ in range(self.cfg.mujoco.decimation):
        self.mj_data.ctrl[:] = targets  # Position targets
        mujoco.mj_step(self.mj_model, self.mj_data)
```

### Task Registry and Auto-Registration

**Registration System:**
```python
class TaskRegistry:
    def __init__(self):
        self._tasks: Dict[str, ControllerConfig] = {}
        self._policies: Dict[str, type[Policy]] = {}

    def register(self, task_name: str, config: ControllerConfig,
                 policy_class: type[Policy]) -> None:
        """Register complete task configuration"""
        self._tasks[task_name] = config
        self._policies[config.policy.task_name] = policy_class

    def get_config(self, task_name: str) -> ControllerConfig:
        return self._tasks[task_name]

    def get_policy(self, policy_type: str) -> type[Policy]:
        return self._policies[policy_type]

# Global registry
TASK_REGISTRY = TaskRegistry()
```

**Decorator-Based Registration:**
```python
# In tasks/velocity/deploy/t1_23dof/__init__.py
from colosseum.deploy.core.registry import register_task

@register_task("t1-velocity-rough")
def register_rough_task():
    return T1_23DOF_VELOCITY_ROUGH, T1VelocityPolicy

@register_task("t1-velocity-flat")
def register_flat_task():
    return T1_23DOF_VELOCITY_FLAT, T1VelocityPolicy
```

**Auto-Discovery:**
```python
def auto_register_tasks() -> None:
    """Scan and import all task deployment modules"""
    # Finds: tasks/{task_name}/deploy/{robot}/__init__.py
    # Example: tasks/velocity/deploy/t1_23dof/__init__.py
    for init_file in tasks_dir.glob("*/deploy/*/__init__.py"):
        module_path = convert_path_to_module(init_file)
        importlib.import_module(module_path)  # Triggers @register_task
```

### Input System (Joystick and Keyboard)

**Joystick Input (evdev-based):**
```python
class JoystickInputSource(BaseInputSource):
    """Threaded evdev gamepad input with deadzone filtering"""

    def _init_backend(self) -> None:
        # Auto-detect joystick with required axes
        self.device = find_joystick_device()

        # Start polling thread for low-latency input
        self.poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True
        )
        self.poll_thread.start()

    def _poll_loop(self) -> None:
        """Event loop running in dedicated thread"""
        while self._running:
            event = self.device.read_one()
            if event.type == evdev.ecodes.EV_ABS:
                self._handle_axis(event.code, event.value)
            elif event.type == evdev.ecodes.EV_KEY:
                self._handle_button(event.code, event.value)

    def get_vx_cmd(self) -> float:
        """Thread-safe getter, applies deadzone, returns [-1, 1]"""
        with self._lock:
            raw = self._state.left_stick_y
            if abs(raw) < self.config.control_threshold:
                return 0.0
            return raw
```

**Standard Gamepad Mapping:**
```python
# Default configuration (Xbox-compatible)
InputConfig(
    input_type="auto",           # Try joystick, fall back to keyboard
    control_threshold=0.1,       # 10% deadzone
    x_axis=ABS_Y,                # Left stick Y (forward/backward)
    y_axis=ABS_X,                # Left stick X (left/right strafe)
    yaw_axis=ABS_RX,             # Right stick X (yaw rotation)
    custom_mode_button=BTN_A,    # Button A
    rl_gait_button=BTN_B,        # Button B
)
```

### Recent Deployment Fixes and Stability Improvements

**Problem: Robot Falling and Oscillating**

Symptoms observed during initial sim-to-sim testing:
- Continuous falling: vertical velocity -5.7 m/s
- Head/arm oscillations: 2.8-1.3 rad/s angular velocities
- Feet not contacting ground properly
- Falls forward immediately on velocity commands

**Root Causes and Fixes:**

**Issue 1: PD Gains Mismatch** (Fixed in [deploy_config.py](src/colosseum/robots/booster_t1/deploy_config.py))

Original deployment config had incorrect PD gains that didn't match motor specifications from [t1_actuators.py](src/colosseum/robots/booster_t1/t1_actuators.py):

```python
# BEFORE (INCORRECT)
joint_stiffness=(
    7.11, 7.11,        # Head (WRONG - should be 15.99)
    111.54, ...,       # Arms (WRONG - should be 160.61)
    ...
)

# AFTER (CORRECTED to match motor specs)
joint_stiffness=(
    15.99, 15.99,      # Head @ 15Hz natural frequency
    160.61, 160.61, 160.61, 160.61,  # Left arm @ 12Hz
    160.61, 160.61, 160.61, 160.61,  # Right arm @ 12Hz
    188.76,            # Waist @ 10Hz
    206.83, 188.76, 188.76, 251.09, 134.05, 134.05,  # Left leg @ 10Hz
    206.83, 188.76, 188.76, 251.09, 134.05, 134.05,  # Right leg @ 10Hz
)

effort_limit=(
    7.0, 7.0,          # Head
    30.0, 30.0, 30.0, 30.0,  # Arms (FIXED from 18.0)
    30.0, 30.0, 30.0, 30.0,
    40.0,              # Waist (FIXED from 25.0)
    90.0, 40.0, 40.0, 118.0, 57.0, 57.0,  # Left leg (FIXED from 24.0)
    90.0, 40.0, 40.0, 118.0, 57.0, 57.0,  # Right leg
)
```

**Issue 2: Initialization Height Mismatch** (Fixed in [config.py](src/colosseum/tasks/velocity/deploy/t1_23dof/config.py))

The robot was initializing with feet floating 3.2cm above the ground:

```python
# Measured at init:
# - Base position: z = 0.665m
# - Foot position: z = 0.0318m (should be ~0)
# - Gap: 3.2cm - feet not touching ground!

# BEFORE
mujoco=MujocoConfig(
    init_pos=(0.0, 0.0, 0.665),  # Feet floating!
    ...
)

# AFTER
mujoco=MujocoConfig(
    init_pos=(0.0, 0.0, 0.70),  # Raised by 3.5cm → feet on ground
    ...
)
```

**Critical Lesson:** The initialization height fix was the PRIMARY solution. Early attempts to fix oscillations by tuning PD gains (increasing ankle damping, knee stiffness) actually made the problem WORSE because the root cause was improper ground contact, not control tuning.

**Files Modified:**
- [src/colosseum/robots/booster_t1/deploy_config.py](src/colosseum/robots/booster_t1/deploy_config.py): Lines 105-145 (PD gains and effort limits)
- [src/colosseum/tasks/velocity/deploy/t1_23dof/config.py](src/colosseum/tasks/velocity/deploy/t1_23dof/config.py): Lines 55, 106 (init_pos height)

**Documentation:**
- `deployment_docs/DEPLOYMENT_FIX_COMPLETE.md`: Complete fix summary
- `deployment_docs/GROUND_CONTACT_FIX.md`: Root cause analysis

### Training-Deployment Consistency Requirements

**Critical Consistency Checklist:**

1. **PD Gains** (Kp, Kd):
   - Training: Computed from motor specs in `t1_actuators.py` (Unitree G1 method)
   - Deployment: MUST use EXACT SAME VALUES in `deploy_config.py`
   - Mismatch causes: oscillations, instability, different control behavior

2. **Actuator Type**:
   - Training: `BuiltinPositionActuatorCfg` (MuJoCo applies internal PD)
   - Deployment: Position actuators with same gains (MuJoCo or robot firmware applies PD)
   - Mismatch causes: completely different control dynamics

3. **Action Scaling**:
   - Training: `policy_action_scale` in env config (typically 0.25)
   - Deployment: `action_scale` in PolicyConfig MUST MATCH
   - Per-joint scale: `action_scale * effort_limit / stiffness`
   - Mismatch causes: wrong joint range, clipping, unintended behavior

4. **Observation Space**:
   - Training: Defined by ObservationGroupCfg in task config
   - Deployment: Defined by ObservationSpec and policy.compute_observation()
   - MUST match: order, dimensions, scaling, normalization
   - Use `VelocityObservationSpec.describe(num_joints)` to verify

5. **Joint Ordering**:
   - Training: MuJoCo alphabetical order (from compiled model)
   - Deployment: Automatically remapped via `real2sim_joint_indexes`
   - Policy always receives/outputs simulation order
   - Hardware always receives real order

6. **Default Joint Positions** (HOME_QPOS):
   - Training: `default_joint_pos` in entity init_state
   - Deployment: `default_joint_pos` in RobotConfig
   - MUST be identical (used as action offset)

### Running Deployment

**Command-Line Interface:**
```bash
# List available registered tasks
pixi run python -m colosseum.deploy.core.registry

# Run MuJoCo sim-to-sim testing
pixi run python -m colosseum.deploy.backends.mujoco \
    --task t1-velocity-flat \
    --use-joystick

# Or programmatically
pixi run python
>>> from colosseum.deploy.core.registry import TASK_REGISTRY
>>> from colosseum.deploy.backends.mujoco import MujocoController
>>>
>>> cfg = TASK_REGISTRY.get_config("t1-velocity-flat")
>>> controller = MujocoController(cfg)
>>> controller.run()
```

**Creating Custom Configurations:**
```python
from colosseum.robots.booster_t1.deploy_config import T1_23DOF_ROBOT_CFG
from colosseum.deploy.config import ControllerConfig, PolicyConfig
from colosseum.deploy.backends.mujoco import MujocoController

cfg = ControllerConfig(
    robot=T1_23DOF_ROBOT_CFG,
    policy=PolicyConfig(
        task_name="velocity",
        checkpoint_path="path/to/model.onnx",
        action_scale=0.25,
    ),
    # Optional: add velocity commands and input
)

controller = MujocoController(cfg)
controller.run()
```

### Deployment Testing Workflow

**Recommended Testing Sequence:**

1. **Sim-to-Sim Testing (MuJoCo)**:
   - Load policy in MuJoCo using deployment controller
   - Validates: observation computation, action scaling, PD gains
   - Test with joystick input to verify velocity tracking
   - Check for: oscillations, falling, unexpected behavior
   - Iterate on config until stable

2. **Real Robot Deployment** (future):
   - Same ControllerConfig, different backend (BoosterController)
   - Adds: network latency, sensor noise, motor dynamics
   - Start with conservative velocity limits
   - Gradually increase aggressiveness

**Key Differences: Sim-to-Sim vs Real Robot**

| Aspect | Sim-to-Sim (MuJoCo) | Real Robot |
|--------|---------------------|------------|
| Physics | Deterministic MuJoCo | Real-world dynamics |
| Sensors | Perfect IMU, no noise | IMU drift, measurement noise |
| Actuators | Instant position control | Motor dynamics, backlash |
| Latency | Single-threaded, <1ms | Network (DDS), ~5-20ms |
| Safety | No hardware risk | Physical damage risk |
| Iteration | Instant restart | Manual intervention |

Both use the **same** Policy, RobotConfig, observation computation, and action scaling - ensuring consistent behavior.

## Common Patterns

### Path Resolution

Use utilities from `colosseum.utils`:
- `project_root()`: Returns repository root (looks for pyproject.toml)
- `src_dir()`: Returns `src/colosseum/` directory

### Robot XML Paths

Always use Path objects and verify existence:
```python
from colosseum.utils import src_dir
xml_path = src_dir() / "robots" / "my_robot" / "xmls" / "model.xml"
assert xml_path.exists(), f"XML not found: {xml_path}"
```

### Actuator Configuration

Group actuators by joint type with regex patterns:
```python
from colosseum.robots.booster_t1.t1_actuators import (
    T1_ACTUATOR_HIP_PITCH,
    T1_ACTUATOR_KNEE,
)

actuators = {
    "hip_pitch": T1_ACTUATOR_HIP_PITCH,  # Pre-configured with motor specs
    "knee": T1_ACTUATOR_KNEE,
}
```

Or create custom configs:
```python
from mjlab.actuator import BuiltinPositionActuatorCfg

actuators = {
    "hip_pitch": BuiltinPositionActuatorCfg(
        joint_names_expr=(".*Hip_Pitch",),  # Matches all hip pitch joints
        stiffness=50.0,
        damping=3.0,
        effort_limit=45.0,
    ),
}
```

### Collision and Contact Configuration

Use pre-configured collision settings for different scenarios:
```python
from colosseum.robots.booster_t1.t1_contacts import (
    FEET_ONLY_COLLISION,
    FEET_GROUND_CONTACT_SENSOR,
)
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

entity_cfg = EntityCfg(
    spec_fn=get_t1_12dof_spec,
    actuators={...},
    articulation_info=EntityArticulationInfoCfg(
        collision=FEET_ONLY_COLLISION,  # Only feet collide (stable training)
    ),
    sensors=[FEET_GROUND_CONTACT_SENSOR],  # Track foot-ground contact
)
```

Available collision presets:
- `FEET_ONLY_COLLISION`: Only feet collide (recommended for locomotion training)
- `FULL_COLLISION_WITHOUT_SELF`: All parts collide with environment
- `FULL_COLLISION`: Full collision including self-collision (most realistic)
- `HANDS_FEET_COLLISION`: Only hands and feet collide (for manipulation)

Available contact sensors:
- `FEET_GROUND_CONTACT_SENSOR`: Tracks foot-ground contact with air time
- `SELF_COLLISION_SENSOR`: Detects self-collisions
- `HAND_CONTACT_SENSOR`: Tracks hand contact for manipulation

## External Dependencies

- **mjlab** (Git submodule): Core RL framework, installed via `pyproject.toml`
- **holosoma** (Git submodule): Motion retargeting (optional, not used in core training)

These are managed in `external/` and installed as editable packages during `pixi install`.
