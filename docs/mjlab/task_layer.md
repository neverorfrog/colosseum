# Deep Dive: Task Layer

The Task Layer defines the reinforcement learning problem (MDP) through a collection of **managers**. Each manager handles a specific aspect of the MDP and orchestrates the execution of **terms** - small, composable functions or classes that implement specific behaviors.

---

## RewardManager

### Purpose

Computes scalar rewards at each timestep by evaluating multiple reward terms and combining them into a single value per environment.

### Key Characteristics

**Terms are functions:**
```python
def my_reward_term(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    # ... additional params
) -> torch.Tensor:
    """Returns reward for some behavior."""
    # Returns shape: (num_envs,)
    ...
```

**Returns:** `torch.Tensor` of shape `(num_envs,)` - one scalar reward per parallel environment

**Reward-per-second interpretation:** All reward functions should return values representing "reward per second". The manager automatically multiplies by `dt` to make rewards frame-rate independent.

### The Flow

RewardManager is called during every `env.step()`:

```python
def step(self, action):
    # 1. Process and apply action
    self.action_manager.process_action(action)
    for _ in range(self.cfg.decimation):
        self.action_manager.apply_action()
        self.sim.step()
    
    # 2. Compute reward ← RewardManager called here
    reward = self.reward_manager.compute(dt=self.step_dt)
    
    # 3. Compute observations, terminations, etc.
    ...
    return obs, reward, done, info
```

**Inside `RewardManager.compute(dt)`:**

```python
def compute(self, dt: float) -> torch.Tensor:
    self._reward_buf[:] = 0.0
    
    for name, term_cfg in zip(self._term_names, self._term_cfgs):
        if term_cfg.weight == 0.0:
            continue
            
        # Call term function
        value = term_cfg.func(self._env, **term_cfg.params)
        
        # Apply weight and dt
        value = value * term_cfg.weight * dt
        
        # Accumulate into total reward
        self._reward_buf += value
        
        # Track for logging
        self._episode_sums[name] += value
        
    return self._reward_buf  # shape: (num_envs,)
```

**Key insight:** The total reward is a weighted sum where each term contributes:
```
reward_i = Σ(term_func_i(env) × weight_i × dt)
```

### Configuration

Reward terms are defined in a dictionary mapping names to `RewardTermCfg` objects:

```python
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import mdp

rewards = {
    "track_velocity": RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        }
    ),
    "torque_penalty": RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-0.0002,  # Negative weight for penalties
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    ),
    "joint_limits": RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    ),
}
```

**RewardTermCfg fields:**
- `func`: Function reference that computes the reward
- `weight`: Multiplicative factor (use negative for penalties)
- `params`: Dictionary of arguments passed to the function (besides `env`)

### Example Computation

Assume `dt = 0.02` (50Hz control) and for environment 0 these values are returned:
- `track_velocity` → `0.8`
- `torque_penalty` → `100.0` (a cost)
- `joint_limits` → `0.0` (no violation)

Total reward for env 0:
```
reward = (0.8 × 1.0 × 0.02) + (100.0 × -0.0002 × 0.02) + (0.0 × -1.0 × 0.02)
       = 0.016 + (-0.0004) + 0.0
       = 0.0156
```

**Key design patterns:**

1. **Positive weights for rewards**, negative for penalties
2. **Functions return costs** (always positive), weight makes them penalties
3. **SceneEntityCfg** resolves string names → indices at init
4. **All terms return shape `(num_envs,)`** for parallel computation

### Writing Custom Reward Terms

**Function signature:**
```python
def my_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # ... additional params
) -> torch.Tensor:
    """
    Args:
        env: Environment instance (access to scene, sim, managers)
        asset_cfg: Resolved entity configuration (has .joint_ids, .body_ids, etc.)
        Additional params from RewardTermCfg.params
    
    Returns:
        Tensor of shape (num_envs,) with reward-per-second values
    """
```

**Accessing simulation data:**
```python
# Get entity by name (resolved during init)
asset = env.scene[asset_cfg.name]

# Access state using pre-resolved indices
joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
actuator_force = asset.data.actuator_force[:, asset_cfg.actuator_ids]
```

**Common patterns:**

**Tracking rewards** (exponential kernel):
```python
def track_target_exp(env, target, std, asset_cfg):
    asset = env.scene[asset_cfg.name]
    current = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error = torch.sum(torch.square(current - target), dim=-1)
    return torch.exp(-error / std**2)
```

**Penalty terms** (L2 regularization):
```python
def joint_acceleration_l2(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    joint_acc = asset.data.qacc[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_acc), dim=-1)
```

**Limit violations** (soft constraints):
```python
def joint_pos_limits(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # Get limits from asset
    soft_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    lower, upper = soft_limits[:, :, 0], soft_limits[:, :, 1]
    
    # Penalize violations
    violation = torch.sum(
        (joint_pos < lower).float() * torch.square(joint_pos - lower) +
        (joint_pos > upper).float() * torch.square(joint_pos - upper),
        dim=-1
    )
    return violation
```

**Best practices:**
- Keep functions pure (no side effects)
- Return reward-per-second values (manager handles dt)
- Use exponential/kernel functions for smooth gradients
- Default to `SceneEntityCfg` parameters for flexibility
- Document the expected range of return values

---

## ActionManager

### Purpose

Converts high-level policy outputs into low-level simulation commands. Handles the mapping from the action space (what the policy sees) to the actuator commands (what the simulation executes).

### Key Characteristics

**Terms are classes** (not functions):
```python
class JointPositionAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Initialize buffers, limits, etc.
    
    def process_actions(self, actions: torch.Tensor) -> None:
        # Store/preprocess actions
        
    def apply_actions(self) -> None:
        # Write to simulation
```

**Two-phase execution:**
1. **`process_actions()`**: Called **once** per control step to preprocess/store actions
2. **`apply_actions()`**: Called **`decimation` times** (once per physics step) to write to simulation

**Why two phases?** Actions are received at control frequency (e.g., 50Hz) but physics runs faster (e.g., 500Hz with decimation=10). The action is processed once, then repeatedly applied during multiple physics steps.

### The Flow

ActionManager participates in every `env.step()`:

```python
def step(self, action):
    # 1. Process action ONCE ← process_actions() called here
    self.action_manager.process_action(action)
    
    # 2. Apply action MULTIPLE times (decimation loop)
    for _ in range(self.cfg.decimation):
        self.action_manager.apply_action()  # ← apply_actions() called here
        self.sim.step()  # Physics step
    
    # 3. Compute observations, rewards, etc.
    ...
```

**Inside `ActionManager.process_action()`:**

```python
def process_action(self, action: torch.Tensor):
    """Split action vector and pass to each term."""
    # Action has shape (num_envs, total_action_dim)
    
    # Split action by term dimensions
    idx = 0
    for term in self._terms.values():
        term_action = action[:, idx:idx + term.action_dim]
        term.process_actions(term_action)  # Call term's process method
        idx += term.action_dim
    
    # Store for access (e.g., action_rate penalties)
    self._prev_action[:] = self._action
    self._action[:] = action
```

**Inside `ActionManager.apply_action()`:**

```python
def apply_action(self):
    """Call apply_actions() on each term."""
    for term in self._terms.values():
        term.apply_actions()  # Write to simulation
```

### Configuration

Action terms are defined in a dictionary mapping names to `ActionTermCfg` subclasses:

```python
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg

actions = {
    "joint_pos": JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],  # All joints
        scale=0.5,  # Scale policy output
        offset=0.0,  # Offset from default pose
    ),
}

# Or for velocity control:
actions = {
    "joint_vel": JointVelocityActionCfg(
        asset_name="robot",
        actuator_names=[".*"],  # All actuators
        scale=1.0,
    ),
}
```

**Common ActionTermCfg fields:**
- `asset_name`: Entity name in scene
- `joint_names` / `actuator_names`: Regex patterns for components
- `scale`: Multiplicative scaling of policy output
- `offset`: Additive offset (for position actions)
- `clip`: Optional (min, max) bounds on processed actions

### Common Action Types

**JointPositionAction:**
Converts policy output to target joint positions:

```python
def process_actions(self, actions):
    # Scale and offset
    self._processed_actions = actions * self.scale + self.offset + self._default_joint_pos
    
    # Clip to joint limits
    self._processed_actions = torch.clamp(
        self._processed_actions,
        self._joint_limits[..., 0],
        self._joint_limits[..., 1]
    )

def apply_actions(self):
    # Write to simulation control
    self._asset.set_joint_position_target(
        self._processed_actions,
        joint_ids=self._joint_ids
    )
```

**Key insight:** The policy outputs actions in a normalized space (e.g., [-1, 1]), which get scaled/offset to the robot's actual joint ranges.

**JointVelocityAction:**
Converts policy output to target joint velocities:

```python
def process_actions(self, actions):
    # Scale
    self._processed_actions = actions * self.scale
    
    # Clip to velocity limits
    self._processed_actions = torch.clamp(
        self._processed_actions,
        -self._velocity_limits,
        self._velocity_limits
    )

def apply_actions(self):
    # Write to simulation control
    self._asset.set_joint_velocity_target(
        self._processed_actions,
        joint_ids=self._joint_ids
    )
```

### Action Space Composition

When multiple action terms are defined, they're **concatenated** into a single action vector:

```python
actions = {
    "arm_pos": JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_arm_.*"],  # 7 DoF
    ),
    "gripper_pos": JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*_finger_.*"],  # 2 DoF
    ),
}

# Total action space: 7 + 2 = 9 dimensions
# Policy output: tensor of shape (num_envs, 9)
# ActionManager splits: [:, 0:7] → arm, [:, 7:9] → gripper
```

### Writing Custom Action Terms

Subclass `ActionTerm` and implement the required methods:

```python
from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.manager_term_config import ActionTermCfg

@dataclass
class MyCustomActionCfg(ActionTermCfg):
    scale: float = 1.0
    # ... other config fields

class MyCustomAction(ActionTerm):
    cfg: MyCustomActionCfg
    
    def __init__(self, cfg: MyCustomActionCfg, env):
        super().__init__(cfg, env)
        
        # Resolve asset and indices
        self._joint_ids = self._asset.find_joints(cfg.joint_names)[0]
        
        # Create buffers
        self._processed_actions = torch.zeros(
            (env.num_envs, len(self._joint_ids)),
            device=env.device
        )
    
    @property
    def action_dim(self) -> int:
        """Return dimensionality of this action term."""
        return len(self._joint_ids)
    
    @property
    def raw_action(self) -> torch.Tensor:
        """Return the raw processed actions (for logging)."""
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor) -> None:
        """Process actions from policy."""
        # Scale, clip, transform as needed
        self._processed_actions = actions * self.cfg.scale
    
    def apply_actions(self) -> None:
        """Apply actions to simulation."""
        # Write to sim (position, velocity, or torque targets)
        self._asset.set_joint_position_target(
            self._processed_actions,
            joint_ids=self._joint_ids
        )
```

**Best practices:**
- Resolve indices in `__init__()` (once)
- Keep `process_actions()` lightweight (called once per control step)
- Make `apply_actions()` very fast (called `decimation` times)
- Store processed actions in buffers for reuse
- Implement `raw_action` property for debugging/logging

### Key Differences from RewardManager

| Aspect | RewardManager | ActionManager |
|--------|---------------|---------------|
| **Terms** | Functions | Classes |
| **Instantiation** | No (functions called directly) | Yes (classes instantiated) |
| **Call frequency** | Once per control step | Two-phase (once + decimation times) |
| **Returns** | Tensor (rewards) | None (writes to sim) |
| **State** | Stateless (pure functions) | Stateful (buffers, limits) |

---

## ObservationManager

### Purpose

Extracts state information from the simulation and processes it into observation tensors that are fed to the policy. Supports multiple observation groups (policy, critic, privileged) and a processing pipeline for each observation term.

### Key Characteristics

**Terms are functions:**
```python
def joint_pos_obs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Returns joint positions."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]
```

**Returns:** `torch.Tensor` with shape `(num_envs, obs_dim)` where `obs_dim` depends on the data being observed

**Processing pipeline:** Each observation passes through an optional processing pipeline:
```
compute → noise → clip → scale → delay → history → flatten
```

**Observation groups:** Observations are organized into groups for different purposes:
- `"policy"`: Standard observations for the actor network
- `"critic"`: Observations for the critic network (can include additional info)
- `"privileged"`: Ground-truth information not available in deployment (terrain height, exact friction, etc.)

### The Flow

ObservationManager is called during every `env.step()` and `env.reset()`:

```python
def step(self, action):
    # 1. Process and apply actions
    self.action_manager.process_action(action)
    for _ in range(self.cfg.decimation):
        self.action_manager.apply_action()
        self.sim.step()
    
    # 2. Compute observations ← ObservationManager called here
    obs = self.observation_manager.compute()
    
    # 3. Compute rewards, terminations, etc.
    ...
    return obs, reward, done, info
```

**Inside `ObservationManager.compute()`:**

```python
def compute(self) -> dict[str, torch.Tensor]:
    """Compute observations for all groups."""
    observations = {}
    
    for group_name, group_cfg in self._group_obs_term_cfgs.items():
        # Compute each term in the group
        group_obs = []
        
        for term_name, term_cfg in group_cfg.items():
            # 1. Compute raw observation
            obs = term_cfg.func(self._env, **term_cfg.params)
            
            # 2. Apply processing pipeline
            obs = self._apply_noise(obs, term_cfg.noise)
            obs = self._apply_clip(obs, term_cfg.clip)
            obs = self._apply_scale(obs, term_cfg.scale)
            obs = self._apply_delay(obs, term_cfg.delay_cfg)
            obs = self._apply_history(obs, term_cfg.history_cfg)
            
            group_obs.append(obs)
        
        # 3. Concatenate all observations in group
        observations[group_name] = torch.cat(group_obs, dim=-1)
    
    return observations  # {"policy": (num_envs, N), "critic": (num_envs, M), ...}
```

### Configuration

Observations are organized into groups, with each group containing multiple observation terms:

```python
from mjlab.managers.manager_term_config import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import mdp

observations = ObservationGroupCfg(
    policy=ObservationTermCfg(
        # Terms for policy network
        joint_pos=ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
            clip=(-5.0, 5.0),
        ),
        joint_vel=ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.5),
        ),
        base_lin_vel=ObservationTermCfg(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        actions=ObservationTermCfg(
            func=mdp.last_action,
        ),
    ),
    critic=ObservationTermCfg(
        # Additional terms for critic (asymmetric actor-critic)
        **observations.policy.__dict__,  # Include all policy obs
        base_height=ObservationTermCfg(
            func=mdp.base_pos_z,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    ),
)
```

**ObservationTermCfg fields:**
- `func`: Function that computes the observation
- `params`: Dictionary of arguments passed to the function
- `noise`: Optional noise model to add to observations
- `clip`: Optional (min, max) bounds to clip values
- `scale`: Optional scaling factor(s)
- `delay_min_lag` / `delay_max_lag`: Sensor delay modeling
- `history_length`: Number of timesteps to stack

### Processing Pipeline

Each observation term passes through a pipeline of optional transformations:

**1. Compute:** Call the observation function
```python
obs = joint_pos_rel(env, asset_cfg)  # shape: (num_envs, 12)
```

**2. Noise:** Add sensor noise
```python
if noise_cfg is not None:
    obs = obs + torch.randn_like(obs) * noise_cfg.std
```

**3. Clip:** Bound values
```python
if clip is not None:
    obs = torch.clamp(obs, clip[0], clip[1])
```

**4. Scale:** Multiply by scaling factor
```python
if scale is not None:
    obs = obs * scale
```

**5. Delay:** Model sensor latency (stores past observations)
```python
# Simulates sensor with 20-60ms latency
if delay_cfg is not None:
    lag = random.randint(delay_cfg.min_lag, delay_cfg.max_lag)
    obs = self._obs_history[lag]  # Use observation from 'lag' steps ago
```

**6. History:** Stack multiple timesteps
```python
# Provides temporal context
if history_cfg is not None:
    obs = torch.cat([
        self._obs_history[0],   # Current
        self._obs_history[1],   # t-1
        self._obs_history[2],   # t-2
    ], dim=-1)  # shape: (num_envs, 12*3)
```

### Observation Groups

Different observation groups serve different purposes:

**Policy observations:**
- What the deployed policy will see
- Should be realistic (noisy, delayed)
- Example: joint positions, velocities, IMU data

**Critic observations:**
- Used only during training
- Can include privileged information
- Often superset of policy observations
- Example: policy obs + exact base height, foot contacts

**Privileged observations:**
- Ground-truth information not available at deployment
- Used for distillation or auxiliary tasks
- Example: terrain heightmap, exact friction coefficients, future commands

**Example configuration:**

```python
observations = ObservationGroupCfg(
    policy=ObservationTermCfg(
        # Noisy, realistic observations
        joint_pos=ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=GaussianNoiseCfg(std=0.01),
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        imu_acc=ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=GaussianNoiseCfg(std=0.05),
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    ),
    critic=ObservationTermCfg(
        # Includes policy obs + privileged info for training
        joint_pos=ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=GaussianNoiseCfg(std=0.01),
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        imu_acc=ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=GaussianNoiseCfg(std=0.05),
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        # Extra info for critic
        base_height=ObservationTermCfg(
            func=mdp.base_pos_z,  # Exact height (no noise)
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        foot_contacts=ObservationTermCfg(
            func=mdp.foot_contact_forces,  # Ground-truth contacts
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        ),
    ),
)
```

### Common Observation Functions

**Joint state:**
```python
def joint_pos_rel(env, asset_cfg):
    """Joint positions relative to default pose."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - \
           asset.data.default_joint_pos[:, asset_cfg.joint_ids]

def joint_vel_rel(env, asset_cfg):
    """Joint velocities relative to default."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]
```

**Base state:**
```python
def base_lin_vel(env, asset_cfg):
    """Linear velocity of base in base frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b  # shape: (num_envs, 3)

def base_ang_vel(env, asset_cfg):
    """Angular velocity of base in base frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b  # shape: (num_envs, 3)

def projected_gravity(env, asset_cfg):
    """Gravity vector projected into base frame (IMU-like)."""
    asset = env.scene[asset_cfg.name]
    return quat_rotate_inverse(
        asset.data.root_quat_w,
        torch.tensor([0, 0, -1], device=env.device)
    )  # shape: (num_envs, 3)
```

**Action history:**
```python
def last_action(env):
    """Previous action sent to simulation."""
    return env.action_manager.action  # shape: (num_envs, action_dim)
```

### Writing Custom Observation Functions

**Function signature:**
```python
def my_observation(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # ... additional params
) -> torch.Tensor:
    """
    Args:
        env: Environment instance
        asset_cfg: Resolved entity configuration
        Additional params from ObservationTermCfg.params
    
    Returns:
        Tensor of shape (num_envs, obs_dim)
    """
```

**Accessing data:**
```python
# Get entity
asset = env.scene[asset_cfg.name]

# Access various state
joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
root_quat = asset.data.root_quat_w

# Access commands (if CommandManager is used)
command = env.command_manager.get_command("base_velocity")

# Access previous actions
prev_action = env.action_manager.prev_action
```

**Example - Height above terrain:**
```python
def height_above_terrain(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Vertical distance from base to terrain."""
    asset = env.scene[asset_cfg.name]
    
    # Get base position in world frame
    base_pos_w = asset.data.root_link_pos_w  # (num_envs, 3)
    
    # Query terrain height at base x,y position
    terrain_height = env.scene.terrain.get_height_at_position(
        base_pos_w[:, :2]  # (num_envs, 2) - x,y only
    )  # Returns (num_envs,)
    
    # Compute vertical distance
    height = base_pos_w[:, 2] - terrain_height
    
    return height.unsqueeze(-1)  # (num_envs, 1)
```

### Best Practices

**For realistic sim-to-real transfer:**
- Add noise to all sensor observations
- Model sensor delays (especially for vision, expensive computations)
- Use relative observations when possible (invariant to initial conditions)
- Avoid privileged information in policy observations

**For observation design:**
- Normalize observations (use `scale` parameter)
- Clip large values to prevent outliers
- Use history for temporal context (velocity estimation, momentum)
- Group related observations together

**For debugging:**
- Start without noise/delay/history
- Add processing gradually
- Log observation statistics (mean, std, min, max)
- Visualize observation distributions

### Key Differences from Other Managers

| Aspect | ObservationManager | RewardManager | ActionManager |
|--------|-------------------|---------------|---------------|
| **Terms** | Functions | Functions | Classes |
| **Returns** | Dict of tensors per group | Single tensor | None |
| **Processing** | Multi-stage pipeline | Simple weighted sum | Two-phase execution |
| **Grouping** | Yes (policy/critic/privileged) | No | No |
| **Statefulness** | Yes (delay, history buffers) | No | Yes (action buffers) |

---

## TerminationManager

### Purpose

Checks episode termination conditions at each timestep. Determines when environments should reset based on failure conditions (robot fell over, joint limits violated) or natural episode endings (timeout, task success).

### Key Characteristics

**Terms are functions:**
```python
def base_height_termination(
    env: ManagerBasedRlEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate if base drops below minimum height."""
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    return base_height < minimum_height  # shape: (num_envs,), dtype: bool
```

**Returns:** `torch.Tensor` of shape `(num_envs,)` with `dtype=torch.bool`

**Time-out flag:** Each term has a `time_out` parameter:
- `time_out=False`: **Termination** - episode failed (robot fell, collision, etc.)
- `time_out=True`: **Truncation** - episode ended naturally (max steps reached, task completed)

**OR logic:** If **any** term returns `True` for an environment, that environment terminates/truncates.

### The Flow

TerminationManager is called during every `env.step()`:

```python
def step(self, action):
    # 1. Process actions and step physics
    self.action_manager.process_action(action)
    for _ in range(self.cfg.decimation):
        self.action_manager.apply_action()
        self.sim.step()
    
    # 2. Compute observations and rewards
    obs = self.observation_manager.compute()
    reward = self.reward_manager.compute(dt=self.step_dt)
    
    # 3. Check terminations ← TerminationManager called here
    dones = self.termination_manager.compute()
    
    # 4. Reset terminated environments
    if dones["terminated"].any() or dones["truncated"].any():
        reset_ids = torch.where(dones["terminated"] | dones["truncated"])[0]
        self._reset_idx(reset_ids)
    
    return obs, reward, dones, info
```

**Inside `TerminationManager.compute()`:**

```python
def compute(self) -> dict[str, torch.Tensor]:
    """Check all termination conditions."""
    self._terminated_buf[:] = False
    self._truncated_buf[:] = False
    
    for name, term_cfg in zip(self._term_names, self._term_cfgs):
        # Call term function
        value = term_cfg.func(self._env, **term_cfg.params)  # bool tensor
        
        # Accumulate using OR logic
        if term_cfg.time_out:
            self._truncated_buf |= value  # Episode timeout
        else:
            self._terminated_buf |= value  # Episode failure
    
    return {
        "terminated": self._terminated_buf,  # Failed episodes
        "truncated": self._truncated_buf,    # Natural endings
    }
```

**Key insight:** Termination uses **OR** logic - if any condition is `True`, the environment resets. This differs from rewards (weighted sum) and observations (concatenation).

### Configuration

Termination terms are defined in a dictionary mapping names to `TerminationTermCfg` objects:

```python
from mjlab.managers.manager_term_config import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import mdp

terminations = {
    "time_out": TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,  # This is a timeout, not a failure
    ),
    "base_contact": TerminationTermCfg(
        func=mdp.illegal_contact,
        time_out=False,  # This is a failure condition
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 1.0,
        }
    ),
    "joint_limits": TerminationTermCfg(
        func=mdp.joint_pos_out_of_limit,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    ),
}
```

**TerminationTermCfg fields:**
- `func`: Function that checks the termination condition
- `time_out`: Boolean indicating if this is a timeout (True) or failure (False)
- `params`: Dictionary of arguments passed to the function

### Common Termination Conditions

**Time-based:**
```python
def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Terminate episodes that exceed maximum duration."""
    return env.episode_length_buf >= env.max_episode_length
```

**Height-based:**
```python
def base_height_below_minimum(
    env: ManagerBasedRlEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate if base drops below threshold (robot fell)."""
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    return base_height < minimum_height
```

**Orientation-based:**
```python
def base_orientation_limit(
    env: ManagerBasedRlEnv,
    roll_threshold: float,
    pitch_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate if robot tilts beyond thresholds."""
    asset = env.scene[asset_cfg.name]
    
    # Convert quaternion to roll, pitch
    roll, pitch, _ = quat_to_euler_xyz(asset.data.root_quat_w)
    
    # Check if either angle exceeds threshold
    roll_violation = torch.abs(roll) > roll_threshold
    pitch_violation = torch.abs(pitch) > pitch_threshold
    
    return roll_violation | pitch_violation
```

**Contact-based:**
```python
def illegal_contact(
    env: ManagerBasedRlEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate if undesired body contacts exceed threshold."""
    # Assume contact sensor tracks specific bodies (e.g., torso, thighs)
    contact_forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w_norm
    
    # Any contact above threshold triggers termination
    return torch.any(contact_forces > threshold, dim=-1)
```

**Joint limit-based:**
```python
def joint_pos_out_of_limit(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate if joints exceed their position limits."""
    asset = env.scene[asset_cfg.name]
    
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    
    lower_limit = joint_limits[:, :, 0]
    upper_limit = joint_limits[:, :, 1]
    
    # Check if any joint violates limits
    out_of_limits = (joint_pos < lower_limit) | (joint_pos > upper_limit)
    return torch.any(out_of_limits, dim=-1)
```

**Velocity-based:**
```python
def joint_vel_limit(
    env: ManagerBasedRlEnv,
    max_velocity: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate if any joint velocity exceeds threshold."""
    asset = env.scene[asset_cfg.name]
    
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    
    # Check if any joint is too fast
    exceeds_limit = torch.abs(joint_vel) > max_velocity
    return torch.any(exceeds_limit, dim=-1)
```

### Writing Custom Termination Terms

**Function signature:**
```python
def my_termination(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # ... additional params
) -> torch.Tensor:
    """
    Args:
        env: Environment instance
        asset_cfg: Resolved entity configuration
        Additional params from TerminationTermCfg.params
    
    Returns:
        Bool tensor of shape (num_envs,) - True for envs that should terminate
    """
```

**Example - Position boundary:**
```python
def out_of_bounds(
    env: ManagerBasedRlEnv,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate if robot leaves allowed area."""
    asset = env.scene[asset_cfg.name]
    
    # Get base position
    pos = asset.data.root_link_pos_w[:, :2]  # (num_envs, 2) - x, y
    
    # Check boundaries
    x_violation = (pos[:, 0] < x_range[0]) | (pos[:, 0] > x_range[1])
    y_violation = (pos[:, 1] < y_range[0]) | (pos[:, 1] > y_range[1])
    
    return x_violation | y_violation
```

### Time-out vs Termination

The distinction between `time_out=True` and `time_out=False` is important for RL training:

**Termination (`time_out=False`):**
- Episode failed due to undesirable state
- Used for value function bootstrapping
- Critic should predict zero future reward
- Examples: robot fell, collision, constraint violation

**Truncation (`time_out=True`):**
- Episode ended but not due to failure
- Used for infinite-horizon tasks
- Critic should bootstrap from current state value
- Examples: max episode length reached, task completed successfully

**Configuration example:**
```python
terminations = {
    # Truncation - natural ending
    "time_out": TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,  # Don't treat as failure
    ),
    
    # Terminations - failures
    "base_contact": TerminationTermCfg(
        func=mdp.illegal_contact,
        time_out=False,  # This is a failure
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0}
    ),
    "fallen": TerminationTermCfg(
        func=mdp.base_height_below_minimum,
        time_out=False,  # This is a failure
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.3}
    ),
}
```

### Best Practices

**For termination design:**
- Use soft thresholds (not exact limits) to give agent warning
- Combine multiple conditions when appropriate (height AND orientation)
- Tune termination thresholds during training (too strict = no learning, too loose = dangerous behaviors)
- Always include `time_out` termination for finite-horizon control

**For debugging:**
- Log which termination condition triggered most often
- Visualize termination boundaries in simulation
- Test edge cases manually before training
- Monitor episode lengths over training

**For curriculum learning:**
- Start with loose termination conditions
- Gradually tighten thresholds as agent improves
- Use EventManager to adjust termination parameters dynamically

### Key Differences from Other Managers

| Aspect | TerminationManager | RewardManager | ObservationManager |
|--------|-------------------|---------------|-------------------|
| **Terms** | Functions | Functions | Functions |
| **Returns** | Bool tensor | Float tensor | Dict of float tensors |
| **Aggregation** | OR logic | Weighted sum | Concatenation per group |
| **Special flags** | time_out (timeout vs failure) | weight | noise, scale, delay, history |
| **Effect** | Resets environments | Guides learning | Feeds policy |

---

## EventManager

### Purpose

Applies events at specific times to modify simulation state. Events handle resets, domain randomization, periodic disturbances, and curriculum updates. Unlike other managers that compute values, EventManager **modifies** the simulation state directly.

### Key Characteristics

**Terms are functions:**
```python
def reset_joint_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Randomize joint positions when environment resets."""
    # Modifies simulation state directly
    ...
```

**Returns:** `None` (functions have side effects - they modify state)

**Three execution modes:**
1. **`"startup"`**: Runs **once** during environment initialization (domain randomization of fixed properties)
2. **`"reset"`**: Runs when environments reset (randomize initial conditions)
3. **`"interval"`**: Runs **periodically** during episodes (push robot, change terrain, etc.)

**Selective application:** Event functions receive `env_ids` parameter indicating which environments to modify.

### The Flow

EventManager is called at different times based on mode:

**Startup mode** (once at init):
```python
def __init__(self, cfg):
    # ... scene and simulation setup ...
    
    self.load_managers()  # Creates EventManager
    
    # Apply startup events ← EventManager called here
    if "startup" in self.event_manager.available_modes:
        self.event_manager.apply(
            mode="startup",
            env_ids=torch.arange(self.num_envs),
            dt=0.0
        )
```

**Reset mode** (when environments terminate):
```python
def step(self, action):
    # ... process actions, step physics, compute rewards ...
    
    # Check terminations
    dones = self.termination_manager.compute()
    
    # Reset terminated environments ← EventManager called here
    if dones["terminated"].any() or dones["truncated"].any():
        reset_ids = torch.where(dones["terminated"] | dones["truncated"])[0]
        
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=reset_ids,
                dt=self.step_dt
            )
```

**Interval mode** (periodic during episode):
```python
def step(self, action):
    # ... process actions, step physics ...
    
    # Apply interval events ← EventManager called here
    if "interval" in self.event_manager.available_modes:
        self.event_manager.apply(
            mode="interval",
            dt=self.step_dt
        )
    
    # ... compute observations, rewards, terminations ...
```

**Inside `EventManager.apply()`:**

```python
def apply(self, mode: str, env_ids: torch.Tensor | None = None, dt: float = 0.0):
    """Apply all events matching the specified mode."""
    for term_cfg in self._mode_term_cfgs[mode]:
        # For interval events, check if it's time to trigger
        if mode == "interval":
            if not self._should_trigger_interval(term_cfg, dt):
                continue
            
            # Sample which environments to affect
            env_ids = self._sample_interval_envs(term_cfg)
        
        # Call event function
        term_cfg.func(self._env, env_ids=env_ids, **term_cfg.params)
```

### Configuration

Event terms are defined in a dictionary with mode-specific configurations:

```python
from mjlab.managers.manager_term_config import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import mdp

events = {
    # Startup: randomize robot masses (once at init)
    "randomize_masses": EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),  # 80% to 120% of default
        }
    ),
    
    # Reset: randomize initial joint positions (every reset)
    "reset_joints": EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.2, 0.2),  # ±0.2 rad from default
            "velocity_range": (-0.1, 0.1),  # ±0.1 rad/s
        }
    ),
    
    # Interval: push robot randomly every 2-5 seconds
    "push_robot": EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 5.0),  # Random interval between pushes
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        }
    ),
}
```

**EventTermCfg fields:**
- `func`: Function that applies the event
- `mode`: When to trigger (`"startup"`, `"reset"`, or `"interval"`)
- `params`: Dictionary of arguments passed to the function
- `interval_range_s`: (min, max) seconds between triggers (interval mode only)
- `min_step_count_between_reset`: Minimum steps before retriggering (interval mode only)

### Common Event Functions

**Startup Events - Domain Randomization:**

**Randomize masses:**
```python
def randomize_rigid_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mass_distribution_params: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Randomize link masses (affects dynamics)."""
    asset = env.scene[asset_cfg.name]
    
    # Get default masses
    default_masses = asset.data.default_mass[env_ids, asset_cfg.body_ids]
    
    # Sample random scaling factors
    scale = torch.rand(len(env_ids), len(asset_cfg.body_ids), device=env.device)
    scale = scale * (mass_distribution_params[1] - mass_distribution_params[0]) + \
            mass_distribution_params[0]
    
    # Apply randomized masses
    asset.data.mass[env_ids, asset_cfg.body_ids] = default_masses * scale
```

**Randomize joint properties:**
```python
def randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    stiffness_range: tuple[float, float],
    damping_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Randomize PD controller gains."""
    asset = env.scene[asset_cfg.name]
    
    # Sample random gains
    stiffness = torch.rand(len(env_ids), len(asset_cfg.joint_ids), device=env.device)
    stiffness = stiffness * (stiffness_range[1] - stiffness_range[0]) + stiffness_range[0]
    
    damping = torch.rand(len(env_ids), len(asset_cfg.joint_ids), device=env.device)
    damping = damping * (damping_range[1] - damping_range[0]) + damping_range[0]
    
    # Apply to actuators
    asset.set_joint_stiffness(stiffness, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_damping(damping, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
```

**Reset Events - Initial Conditions:**

**Randomize joint positions:**
```python
def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Set random initial joint positions and velocities."""
    asset = env.scene[asset_cfg.name]
    
    # Sample random offsets from default pose
    pos_offset = torch.rand(len(env_ids), len(asset_cfg.joint_ids), device=env.device)
    pos_offset = pos_offset * (position_range[1] - position_range[0]) + position_range[0]
    
    vel = torch.rand(len(env_ids), len(asset_cfg.joint_ids), device=env.device)
    vel = vel * (velocity_range[1] - velocity_range[0]) + velocity_range[0]
    
    # Apply to simulation
    default_pos = asset.data.default_joint_pos[env_ids, asset_cfg.joint_ids]
    asset.write_joint_state_to_sim(
        position=default_pos + pos_offset,
        velocity=vel,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids
    )
```

**Randomize base pose:**
```python
def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Set random initial base position, orientation, and velocity."""
    asset = env.scene[asset_cfg.name]
    
    # Sample random pose
    pos = torch.zeros(len(env_ids), 3, device=env.device)
    pos[:, 0] = torch.rand(len(env_ids)) * (pose_range["x"][1] - pose_range["x"][0]) + pose_range["x"][0]
    pos[:, 1] = torch.rand(len(env_ids)) * (pose_range["y"][1] - pose_range["y"][0]) + pose_range["y"][0]
    pos[:, 2] = torch.rand(len(env_ids)) * (pose_range["z"][1] - pose_range["z"][0]) + pose_range["z"][0]
    
    # Random orientation (yaw only for simplicity)
    yaw = torch.rand(len(env_ids)) * (pose_range["yaw"][1] - pose_range["yaw"][0]) + pose_range["yaw"][0]
    quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    
    # Random velocity
    lin_vel = torch.zeros(len(env_ids), 3, device=env.device)
    lin_vel[:, 0] = torch.rand(len(env_ids)) * (velocity_range["x"][1] - velocity_range["x"][0]) + velocity_range["x"][0]
    
    # Apply to simulation
    asset.write_root_state_to_sim(
        root_state=torch.cat([pos, quat, lin_vel, torch.zeros(len(env_ids), 3)], dim=-1),
        env_ids=env_ids
    )
```

**Interval Events - Disturbances:**

**Push robot:**
```python
def push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Apply random velocity impulse to robot base."""
    asset = env.scene[asset_cfg.name]
    
    # Sample random velocity
    vel = torch.zeros(len(env_ids), 3, device=env.device)
    vel[:, 0] = torch.rand(len(env_ids)) * (velocity_range["x"][1] - velocity_range["x"][0]) + velocity_range["x"][0]
    vel[:, 1] = torch.rand(len(env_ids)) * (velocity_range["y"][1] - velocity_range["y"][0]) + velocity_range["y"][0]
    
    # Get current state and modify velocity
    root_state = asset.data.root_state_w[env_ids].clone()
    root_state[:, 7:10] += vel  # Add to existing velocity
    
    # Write back to simulation
    asset.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
```

### Interval Event Timing

For `mode="interval"` events, the timing is controlled by `interval_range_s`:

```python
"push_robot": EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(2.0, 5.0),  # Push every 2-5 seconds
    ...
)
```

**How it works:**
1. Each environment has an independent timer
2. When timer expires, the event triggers for that environment
3. A new interval is randomly sampled from the range
4. Timer resets and counts down again

**Per-environment vs global timing:**
```python
"push_robot": EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(2.0, 5.0),
    is_global_time=False,  # Each env has own timer (default)
    ...
)
```

- `is_global_time=False`: Each environment has independent timing (default)
- `is_global_time=True`: All environments trigger simultaneously

### Writing Custom Event Functions

**Function signature:**
```python
def my_event(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # ... additional params
) -> None:
    """
    Args:
        env: Environment instance
        env_ids: Tensor of environment indices to modify
        asset_cfg: Resolved entity configuration
        Additional params from EventTermCfg.params
    
    Returns:
        None (modifies simulation state as side effect)
    """
```

**Accessing and modifying state:**
```python
def my_event(env, env_ids, asset_cfg):
    asset = env.scene[asset_cfg.name]
    
    # Read current state
    joint_pos = asset.data.joint_pos[env_ids, asset_cfg.joint_ids]
    
    # Modify state
    new_joint_pos = joint_pos + torch.randn_like(joint_pos) * 0.1
    
    # Write back to simulation
    asset.write_joint_state_to_sim(
        position=new_joint_pos,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids
    )
```

### Best Practices

**For domain randomization:**
- Randomize parameters that affect real-world transfer (masses, friction, gains)
- Use startup events for fixed properties (mass, inertia)
- Use reset events for initial conditions (pose, velocity)
- Start with small randomization ranges, increase gradually

**For curriculum learning:**
- Use interval events to increase difficulty over time
- Combine with CurriculumManager for automatic progression
- Track success rate to determine when to increase difficulty

**For robustness:**
- Add push disturbances during training
- Randomize terrain properties
- Vary control latency using observation delays

**For debugging:**
- Log which events triggered and when
- Visualize randomized parameters
- Test events individually before combining
- Monitor if events make task impossible

### Key Differences from Other Managers

| Aspect | EventManager | RewardManager | TerminationManager |
|--------|--------------|---------------|-------------------|
| **Terms** | Functions | Functions | Functions |
| **Returns** | None (side effects) | Float tensor | Bool tensor |
| **Timing** | Mode-dependent | Every step | Every step |
| **Purpose** | Modify state | Guide learning | Determine resets |
| **env_ids** | Yes (selective) | No (all envs) | No (all envs) |

---

## CommandManager

### Purpose

Generates goal commands that specify what the agent should achieve. Commands are typically target velocities, positions, or trajectories that the agent tracks. Other managers (ObservationManager and RewardManager) reference these commands to provide goals to the policy and compute tracking rewards.

### Key Characteristics

**Terms are classes:**
```python
class UniformVelocityCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Initialize command buffers
    
    def compute(self, dt: float) -> torch.Tensor:
        # Generate/resample commands
        return self._command  # shape: (num_envs, command_dim)
```

**Returns:** `torch.Tensor` of shape `(num_envs, command_dim)` containing the current command for each environment

**Resampling:** Commands are automatically resampled at random intervals specified by `resampling_time_range`

**Referenced by other managers:**
- Observations: Include current command so policy knows the goal
- Rewards: Compute how well agent tracks the command

### The Flow

CommandManager is called during every `env.step()` and `env.reset()`:

```python
def step(self, action):
    # 1. Process actions and step physics
    self.action_manager.process_action(action)
    for _ in range(self.cfg.decimation):
        self.action_manager.apply_action()
        self.sim.step()
    
    # 2. Update commands ← CommandManager called here
    if hasattr(self, 'command_manager'):
        self.command_manager.compute(dt=self.step_dt)
    
    # 3. Compute observations (may include commands)
    obs = self.observation_manager.compute()
    
    # 4. Compute rewards (may use commands for tracking)
    reward = self.reward_manager.compute(dt=self.step_dt)
    
    ...
```

**Inside `CommandManager.compute()`:**

```python
def compute(self, dt: float) -> dict[str, torch.Tensor]:
    """Update and resample commands as needed."""
    commands = {}
    
    for name, term in self._terms.items():
        # Update resampling timer
        term._time_left -= dt
        
        # Check if it's time to resample
        env_ids = torch.where(term._time_left <= 0)[0]
        
        if len(env_ids) > 0:
            # Resample command for these environments
            term.resample(env_ids)
            
            # Reset timers with new random intervals
            term._time_left[env_ids] = torch.rand(len(env_ids)) * \
                (term.cfg.resampling_time_range[1] - term.cfg.resampling_time_range[0]) + \
                term.cfg.resampling_time_range[0]
        
        # Store current command
        commands[name] = term.command
    
    return commands  # {"base_velocity": (num_envs, 3), "heading": (num_envs, 1), ...}
```

### Configuration

Command terms are defined in a dictionary mapping names to `CommandTermCfg` subclasses:

```python
from mjlab.envs.mdp.commands import UniformVelocityCommandCfg
from mjlab.managers.manager_term_config import CommandTermCfg

commands = {
    "base_velocity": UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),  # Resample every 5-10 seconds
        ranges={
            "lin_vel_x": (-1.0, 1.0),  # m/s
            "lin_vel_y": (-0.5, 0.5),  # m/s
            "ang_vel_z": (-1.0, 1.0),  # rad/s
        },
    ),
}
```

**CommandTermCfg fields:**
- `class_type`: Command class to instantiate
- `resampling_time_range`: (min, max) seconds between resampling
- Additional fields specific to the command type (ranges, trajectories, etc.)

### Common Command Types

**Velocity commands** (for locomotion):
```python
class UniformVelocityCommand(CommandTerm):
    """Sample target velocities from uniform distribution."""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        
        # Command buffer: (num_envs, 3) for [vx, vy, omega_z]
        self._command = torch.zeros(env.num_envs, 3, device=env.device)
    
    def resample(self, env_ids: torch.Tensor):
        """Sample new velocity commands."""
        # Sample from uniform distribution
        self._command[env_ids, 0] = torch.rand(len(env_ids)) * \
            (self.cfg.ranges["lin_vel_x"][1] - self.cfg.ranges["lin_vel_x"][0]) + \
            self.cfg.ranges["lin_vel_x"][0]
        
        self._command[env_ids, 1] = torch.rand(len(env_ids)) * \
            (self.cfg.ranges["lin_vel_y"][1] - self.cfg.ranges["lin_vel_y"][0]) + \
            self.cfg.ranges["lin_vel_y"][0]
        
        self._command[env_ids, 2] = torch.rand(len(env_ids)) * \
            (self.cfg.ranges["ang_vel_z"][1] - self.cfg.ranges["ang_vel_z"][0]) + \
            self.cfg.ranges["ang_vel_z"][0]
    
    @property
    def command(self) -> torch.Tensor:
        return self._command
```

**Pose commands** (for manipulation):
```python
class UniformPoseCommand(CommandTerm):
    """Sample target end-effector poses."""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        
        # Command buffer: (num_envs, 7) for [x, y, z, qw, qx, qy, qz]
        self._command = torch.zeros(env.num_envs, 7, device=env.device)
        self._command[:, 3] = 1.0  # Initialize with identity quaternion
    
    def resample(self, env_ids: torch.Tensor):
        """Sample new pose commands within workspace."""
        # Sample position
        self._command[env_ids, 0] = torch.rand(len(env_ids)) * \
            (self.cfg.ranges["x"][1] - self.cfg.ranges["x"][0]) + self.cfg.ranges["x"][0]
        
        self._command[env_ids, 1] = torch.rand(len(env_ids)) * \
            (self.cfg.ranges["y"][1] - self.cfg.ranges["y"][0]) + self.cfg.ranges["y"][0]
        
        self._command[env_ids, 2] = torch.rand(len(env_ids)) * \
            (self.cfg.ranges["z"][1] - self.cfg.ranges["z"][0]) + self.cfg.ranges["z"][0]
        
        # Sample orientation (random quaternions)
        self._command[env_ids, 3:] = random_quaternion(len(env_ids), device=self.device)
```

### Using Commands in Other Managers

Commands are designed to be referenced by observations and rewards:

**In ObservationManager:**
```python
observations = {
    "policy": {
        # Include command so policy knows the goal
        "target_velocity": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},  # Reference command by name
        ),
        # ... other observations
    }
}
```

**In RewardManager:**
```python
rewards = {
    # Reward for tracking the commanded velocity
    "track_velocity": RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",  # Use command for comparison
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        }
    ),
}
```

**Command observation function:**
```python
def generated_commands(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Return current command from CommandManager."""
    return env.command_manager.get_command(command_name)
```

**Command tracking reward:**
```python
def track_lin_vel_xy_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for tracking commanded x-y velocity."""
    # Get current velocity
    asset = env.scene[asset_cfg.name]
    lin_vel_b = asset.data.root_lin_vel_b[:, :2]  # x, y in base frame
    
    # Get commanded velocity
    command = env.command_manager.get_command(command_name)
    command_vel = command[:, :2]  # x, y components
    
    # Compute tracking error
    error = torch.sum(torch.square(lin_vel_b - command_vel), dim=-1)
    
    # Exponential reward
    return torch.exp(-error / std**2)
```

### Command Resampling Strategy

Commands resample at random intervals to create curriculum:

**Fixed intervals:**
```python
commands = {
    "base_velocity": UniformVelocityCommandCfg(
        resampling_time_range=(10.0, 10.0),  # Exactly 10 seconds
        ...
    ),
}
```

**Variable intervals:**
```python
commands = {
    "base_velocity": UniformVelocityCommandCfg(
        resampling_time_range=(5.0, 15.0),  # Between 5-15 seconds
        ...
    ),
}
```

**Why random resampling?**
- Prevents agent from learning temporal patterns
- Creates diverse training scenarios
- Ensures agent can handle sudden command changes
- Simulates realistic command variations

### Writing Custom Command Terms

Subclass `CommandTerm` and implement the required methods:

```python
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg

@dataclass
class MyCustomCommandCfg(CommandTermCfg):
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    # ... other config fields

class MyCustomCommand(CommandTerm):
    cfg: MyCustomCommandCfg
    
    def __init__(self, cfg: MyCustomCommandCfg, env):
        super().__init__(cfg, env)
        
        # Initialize command buffer
        self._command = torch.zeros(
            env.num_envs,
            self.command_dim,
            device=env.device
        )
        
        # Resample for all environments initially
        self.resample(torch.arange(env.num_envs, device=env.device))
    
    @property
    def command_dim(self) -> int:
        """Dimensionality of the command."""
        return len(self.cfg.ranges)
    
    @property
    def command(self) -> torch.Tensor:
        """Return current command."""
        return self._command
    
    def resample(self, env_ids: torch.Tensor):
        """Generate new commands for specified environments."""
        # Sample from your distribution
        for i, (key, (low, high)) in enumerate(self.cfg.ranges.items()):
            self._command[env_ids, i] = torch.rand(len(env_ids), device=self.device) * \
                (high - low) + low
```

### Best Practices

**For command design:**
- Start with simple, achievable commands
- Gradually expand command ranges as agent improves (curriculum)
- Ensure commands are feasible given robot constraints
- Use realistic resampling times (not too fast/slow)

**For curriculum learning:**
- Begin with narrow ranges (easier goals)
- Expand ranges over training
- Track success rate to adjust difficulty
- Use CurriculumManager for automatic progression

**For multi-goal tasks:**
- Define multiple command generators
- Each can have different resampling strategies
- Combine in observations and rewards appropriately

**For debugging:**
- Visualize commanded vs achieved trajectories
- Log command distributions over training
- Check if commands fall within feasible workspace
- Monitor resampling frequency

### Key Differences from Other Managers

| Aspect | CommandManager | ObservationManager | ActionManager |
|--------|----------------|-------------------|---------------|
| **Terms** | Classes | Functions | Classes |
| **Returns** | Dict of float tensors | Dict of float tensors | None |
| **Purpose** | Generate goals | Extract state | Apply control |
| **Referenced by** | Observations, Rewards | Policy/Critic | Simulation |
| **Resampling** | Yes (time-based) | No | No |

---

## Summary

The Task Layer orchestrates the RL problem through six main managers:

1. **RewardManager**: Computes scalar rewards (weighted sum of functions)
2. **ActionManager**: Converts policy outputs to simulation commands (classes, two-phase)
3. **ObservationManager**: Extracts and processes state (functions, multi-stage pipeline, groups)
4. **TerminationManager**: Checks episode end conditions (functions, OR logic, timeout flag)
5. **EventManager**: Modifies simulation state (functions, three modes, domain randomization)
6. **CommandManager**: Generates goal commands (classes, resampling, referenced by others)

Each manager follows the term-based pattern:
- **Declaration**: Define terms in config dictionaries
- **Resolution**: String names → indices (once at init)
- **Execution**: Fast runtime computation (every step or event-driven)
