# Tutorial: Creating a CartPole Balancing Task

This tutorial walks through creating a complete CartPole balancing task in colosseum using mjlab. You'll learn how to apply the concepts from the Task Layer to build a working RL environment from scratch.

## Overview

**Goal**: Train a policy to balance a pole on a moving cart using velocity control.

**What you'll learn:**
- Setting up a scene with a simple robot
- Designing reward functions for a balancing task
- Configuring observations for pole balancing
- Defining termination conditions
- Adding domain randomization

---

## 1. The CartPole Robot

### Model Description

The CartPole consists of:
- **Cart**: Slides along the x-axis
- **Pole**: Hinges on the cart, rotates around the y-axis
- **Control**: Velocity actuator on the cart's slide joint

### XML Structure

```xml
<mujoco model="cartpole">
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slide" type="slide" axis="1 0 0" />
      <geom type="box" size="0.2 0.1 0.1" />
      
      <body name="pole" pos="0 0 0.1">
        <joint name="hinge" type="hinge" axis="0 1 0" />
        <geom type="capsule" fromto="0 0 0 0 0 1" size="0.05" />
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <velocity name="slide_velocity" joint="slide" ctrlrange="-20 20" kv="20"/>
  </actuator>
</mujoco>
```

**Key joints:**
- `slide`: Cart position along x-axis
- `hinge`: Pole angle from vertical

---

## 2. Designing the MDP

### Observations

The agent needs to observe:
1. **Cart position** - Where the cart is
2. **Cart velocity** - How fast the cart is moving
3. **Pole angle** - How tilted the pole is
4. **Pole angular velocity** - How fast the pole is rotating

```python
from mjlab.managers.manager_term_config import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import mdp

def create_cartpole_observations() -> ObservationGroupCfg:
    """Create CartPole observations."""
    return ObservationGroupCfg(
        policy=ObservationTermCfg(
            # Cart position (slide joint)
            cart_pos=ObservationTermCfg(
                func=mdp.joint_pos,
                params={"asset_cfg": SceneEntityCfg("cart", joint_names="slide")},
            ),
            # Cart velocity
            cart_vel=ObservationTermCfg(
                func=mdp.joint_vel,
                params={"asset_cfg": SceneEntityCfg("cart", joint_names="slide")},
            ),
            # Pole angle (hinge joint)
            pole_angle=ObservationTermCfg(
                func=mdp.joint_pos,
                params={"asset_cfg": SceneEntityCfg("cart", joint_names="hinge")},
            ),
            # Pole angular velocity
            pole_vel=ObservationTermCfg(
                func=mdp.joint_vel,
                params={"asset_cfg": SceneEntityCfg("cart", joint_names="hinge")},
            ),
        )
    )
```

**Observation space**: 4-dimensional vector `[cart_pos, cart_vel, pole_angle, pole_vel]`

### Actions

The policy outputs a single value controlling cart velocity:

```python
from mjlab.envs.mdp.actions import JointVelocityActionCfg

def create_cartpole_actions() -> dict[str, JointVelocityActionCfg]:
    """Create CartPole actions."""
    return {
        "cart_velocity": JointVelocityActionCfg(
            asset_name="cart",
            actuator_names=["slide_velocity"],
            scale=20.0,  # Scale [-1, 1] → [-20, 20] m/s
        ),
    }
```

**Action space**: 1-dimensional, policy outputs in range [-1, 1]

### Rewards

We design three reward terms:

#### 1. Stay Alive - Constant Positive Reward

Encourage the agent to keep the episode running:

```python
def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Constant reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)
```

**Intuition**: Every timestep the pole stays balanced earns reward.

#### 2. Pole Upright - Exponential Kernel

Reward for keeping the pole vertical using cosine of angle:

```python
def pole_upright_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward for keeping pole upright using cosine of angle."""
    asset = env.scene[asset_cfg.name]
    
    # Get pole angle from hinge joint
    pole_angle = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    
    # Use cosine: 1.0 when upright (0°), -1.0 when inverted (180°)
    alignment = torch.cos(pole_angle)
    
    # Apply exponential kernel for smooth gradient
    error = 1.0 - alignment  # 0 when upright, 2 when inverted
    reward = torch.exp(-error**2 / std**2)
    
    return reward  # shape: (num_envs,)
```

**Why cosine?**
- `cos(0°) = 1.0` → Pole perfectly upright (maximum reward)
- `cos(90°) = 0.0` → Pole horizontal
- `cos(180°) = -1.0` → Pole inverted

**Exponential kernel** provides smooth gradients for learning.

#### 3. Effort Penalty - Actuator Force

Penalize large control efforts to encourage energy-efficient solutions:

```python
def actuator_effort_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize actuator force magnitude (L2 norm)."""
    asset = env.scene[asset_cfg.name]
    
    # Sum squared forces across all actuators
    effort_squared = torch.sum(
        torch.square(asset.data.actuator_force), 
        dim=1
    )
    
    return effort_squared  # Returns cost (always positive)
```

**Note**: Function returns **cost** (positive), config uses **negative weight** to make it a penalty.

#### Reward Configuration

```python
from mjlab.managers.manager_term_config import RewardTermCfg

def create_cartpole_rewards() -> dict[str, RewardTermCfg]:
    """Create CartPole rewards."""
    return {
        "alive": RewardTermCfg(
            func=mdp.is_alive,
            weight=1.0,  # Positive: we want this
        ),
        "pole_upright": RewardTermCfg(
            func=pole_upright_reward,
            weight=1.0,  # Positive: we want this
            params={
                "asset_cfg": SceneEntityCfg("cart", joint_names="hinge"),
                "std": 0.5,
            }
        ),
        "effort": RewardTermCfg(
            func=actuator_effort_penalty,
            weight=-0.01,  # NEGATIVE: penalize high effort
            params={
                "asset_cfg": SceneEntityCfg("cart"),
            }
        ),
    }
```

**Reward computation example** (dt=0.02):
- `alive`: 1.0 × 1.0 × 0.02 = 0.02
- `pole_upright`: 0.9 × 1.0 × 0.02 = 0.018
- `effort`: 100.0 × (-0.01) × 0.02 = -0.02
- **Total**: 0.02 + 0.018 - 0.02 = 0.018

### Terminations

Define when episodes should end:

#### 1. Time Out - Maximum Episode Length

```python
def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Terminate episodes that exceed maximum duration."""
    return env.episode_length_buf >= env.max_episode_length
```

**Configuration**: `time_out=True` (this is a natural ending, not a failure)

#### 2. Pole Fallen - Angle Threshold

```python
def pole_angle_limit(
    env: ManagerBasedRlEnv,
    threshold: float,  # In radians
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when pole angle exceeds threshold."""
    asset = env.scene[asset_cfg.name]
    
    # Get pole angle from hinge joint
    pole_angle = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    
    # Check if absolute angle exceeds threshold
    fallen = torch.abs(pole_angle) > threshold
    
    return fallen  # shape: (num_envs,), dtype: bool
```

**Threshold**: π/4 radians (45°) - pole fallen if tilted beyond this

#### Termination Configuration

```python
from mjlab.managers.manager_term_config import TerminationTermCfg

def create_cartpole_terminations() -> dict[str, TerminationTermCfg]:
    """Create CartPole terminations."""
    return {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,  # Natural ending (not failure)
        ),
        "pole_fallen": TerminationTermCfg(
            func=pole_angle_limit,
            time_out=False,  # Failure condition
            params={
                "asset_cfg": SceneEntityCfg("cart", joint_names="hinge"),
                "threshold": 0.785,  # π/4 radians = 45°
            }
        ),
    }
```

---

## 3. Domain Randomization

Add robustness through randomized initial conditions:

### Reset Events

Randomize initial pole angle on each episode reset:

```python
def reset_pole_angle(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    angle_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Set random initial pole angle."""
    asset = env.scene[asset_cfg.name]
    
    # Sample random angles
    angles = torch.rand(len(env_ids), device=env.device)
    angles = angles * (angle_range[1] - angle_range[0]) + angle_range[0]
    
    # Apply to simulation
    asset.write_joint_state_to_sim(
        position=angles.unsqueeze(-1),
        velocity=torch.zeros(len(env_ids), 1, device=env.device),
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids
    )
```

#### Event Configuration

```python
from mjlab.managers.manager_term_config import EventTermCfg

def create_cartpole_events() -> dict[str, EventTermCfg]:
    """Create CartPole events."""
    return {
        "reset_pole": EventTermCfg(
            func=reset_pole_angle,
            mode="reset",  # Trigger on environment reset
            params={
                "asset_cfg": SceneEntityCfg("cart", joint_names="hinge"),
                "angle_range": (-0.1, 0.1),  # ±0.1 rad from vertical
            }
        ),
    }
```

**Effect**: Each episode starts with pole at slightly different angle, improving robustness.

---

## 4. Complete Environment Configuration

Putting it all together:

```python
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

# Scene setup
SCENE_CFG = SceneCfg(
    num_envs=64,
    extent=1.0,
    entities={"cart": get_cartpole_robot_cfg()},
)

# Simulation configuration
SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.02,  # 50Hz control frequency
        iterations=1,
    ),
)

# Environment configuration
CARTPOLE_ENV_CFG = ManagerBasedRlEnvCfg(
    # Scene and simulation
    scene=SCENE_CFG,
    sim=SIM_CFG,
    
    # MDP components
    observations=create_cartpole_observations(),
    actions=create_cartpole_actions(),
    rewards=create_cartpole_rewards(),
    terminations=create_cartpole_terminations(),
    events=create_cartpole_events(),
    
    # Episode settings
    decimation=1,  # No decimation (control freq = physics freq)
    episode_length_s=10.0,  # 10 second episodes
)
```

---

## 5. Key Design Decisions

### Why Velocity Control?

**Velocity actuators** are simpler than position control for this task:
- Direct control over cart speed
- No need for PD controller tuning
- Natural for continuous control

### Why These Rewards?

1. **Stay alive**: Sparse signal - encourages longer episodes
2. **Pole upright**: Dense signal - guides learning with smooth gradients
3. **Effort penalty**: Encourages efficiency - prevents aggressive movements

**Balance**: Dense rewards (pole upright) provide learning signal, sparse rewards (stay alive) provide task objective.

### Why 45° Termination?

- Too strict (e.g., 10°): Agent never learns, episodes too short
- Too loose (e.g., 80°): Agent learns bad policies, unsafe behaviors
- 45°: Sweet spot - allows exploration while preventing complete failures

---

## 6. Training Tips

### Hyperparameter Tuning

**Reward weights:**
- Start with `alive=1.0`, `pole_upright=1.0`, `effort=-0.01`
- If agent too aggressive: Increase effort penalty (e.g., -0.05)
- If agent too passive: Decrease effort penalty (e.g., -0.001)

**Termination threshold:**
- Start loose (60°), tighten during training (45° → 30°)
- Use curriculum learning to gradually increase difficulty

**Episode length:**
- Start short (5s), increase as agent improves (10s → 20s)
- Prevents long episodes of poor performance early in training

### Debugging

**Common issues:**

1. **Agent pushes cart to limits**: Increase effort penalty or add cart position penalty
2. **Pole oscillates**: Increase `std` in pole_upright reward for smoother gradients
3. **Episodes too short**: Loosen termination threshold or increase rewards
4. **No learning**: Check observation scaling, verify rewards are non-zero

---

## 7. Extensions

Once basic balancing works, try:

### Swing-Up Task

Start with pole hanging down, swing up and balance:

```python
"swing_up_reward": RewardTermCfg(
    func=pole_energy_reward,  # Reward kinetic + potential energy
    weight=0.5,
    params={"asset_cfg": SceneEntityCfg("cart", joint_names="hinge")},
)
```

### Moving Target

Add cart position command to track:

```python
"track_position": RewardTermCfg(
    func=mdp.track_position_exp,
    weight=0.5,
    params={
        "command_name": "target_pos",
        "asset_cfg": SceneEntityCfg("cart", joint_names="slide"),
    }
)
```

### Double Pole

Add second pole for increased difficulty:

```xml
<body name="pole2" pos="0 0 1">
  <joint name="hinge2" type="hinge" axis="0 1 0" />
  <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.03" />
</body>
```

---

## Summary

**What we built:**
- Simple 2-DoF robot (cart + pole)
- 4D observation space (positions + velocities)
- 1D action space (cart velocity)
- 3 reward terms (alive, upright, effort)
- 2 termination conditions (timeout, fallen)
- Reset randomization (initial pole angle)

**Key takeaways:**
1. **Reward design matters**: Balance dense and sparse rewards
2. **Terminations enable learning**: Not too strict, not too loose
3. **Domain randomization helps**: Randomize initial conditions
4. **Iterate and tune**: Start simple, add complexity gradually

**Next steps:**
- Train the policy using your preferred RL algorithm (PPO, SAC, etc.)
- Visualize learned behaviors
- Try the suggested extensions
- Apply these patterns to more complex robots!