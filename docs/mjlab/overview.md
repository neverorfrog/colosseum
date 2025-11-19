# mjlab Architecture Overview

## What is mjlab?

**mjlab** is a GPU-accelerated reinforcement learning environment framework for robotics, built on top of MuJoCo Warp. It provides Isaac Lab's proven manager-based API with MuJoCo's simplicity and performance.

**Key characteristics:**
- **Massive parallelization**: Run 4096+ environments simultaneously on GPU
- **Manager-based architecture**: Declarative configuration for rewards, observations, actions, and events
- **MuJoCo native**: Direct access to mjModel/mjData structures
- **Fast iteration**: Lightweight installation, quick startup, standard Python debugging

---

## The Three-Layer Architecture

mjlab is organized into three abstraction layers, each with a clear responsibility:

```
┌─────────────────────────────────────────┐
│     TASK / ENVIRONMENT LAYER            │  ← Highest abstraction
│  (ManagerBasedRlEnv)                    │     
│  - Defines MDP (rewards, obs, actions)  │     What to optimize
│  - Managers orchestrate term execution  │
│  - gym.Env interface for RL training    │
└──────────────────┬──────────────────────┘
                   │ uses
┌──────────────────▼──────────────────────┐
│        SCENE LAYER                      │  ← Middle abstraction
│  (Scene, Entity, Sensors)               │     
│  - Physical objects (robot, terrain)    │     What exists
│  - Name → index mapping                 │
│  - Entity data access (poses, vels)     │
└──────────────────┬──────────────────────┘
                   │ creates/updates
┌──────────────────▼──────────────────────┐
│      SIMULATION LAYER                   │  ← Lowest abstraction
│  (Simulation, MuJoCo Warp)              │     
│  - GPU-accelerated physics              │     How it moves
│  - Batched parallel execution           │
│  - Direct mjModel/mjData access         │
└─────────────────────────────────────────┘
```

### 1. Simulation Layer (Lowest)

**Purpose**: Run GPU-accelerated physics across thousands of parallel environments.

**Key components:**
- `Simulation`: Wrapper around MuJoCo Warp's batched simulator
- `mjwarp.Model`: Compiled physics model (immutable)
- `mjwarp.Data`: Simulation state arrays (qpos, qvel, xpos, etc.) with shape `(num_envs, ...)`

**What it does:**
```python
# Create 4096 parallel physics simulations
sim = Simulation(num_envs=4096, model=compiled_model, device="cuda")

# Step all environments in parallel (single GPU kernel)
sim.step()  # Updates positions, velocities, forces for all 4096 envs

# Access state (all envs simultaneously)
all_positions = sim.data.qpos  # shape: (4096, nq)
```

**Key insight**: This layer doesn't know about "robots" or "tasks" - it just runs physics on tensors.

---

### 2. Scene Layer (Middle)

**Purpose**: Organize physical objects and provide convenient access to their state.

**Key components:**

#### **Entity**
Represents a physical object (robot, terrain, obstacle). Provides:
- High-level data access: `robot.data.joint_pos`, `robot.data.root_link_pos_w`
- Coordinate frame transformations (world ↔ body, COM ↔ link)
- Name-based component access: `robot.find_joints(".*_knee")`

#### **EntityIndexing**
Maps component names to global MuJoCo indices:
```python
# After compilation and initialization
robot.joint_names  # ("hip_pitch", "knee", "ankle")
robot.find_joints("knee")  # Returns: ([5], ["robot/knee"])
#                                      ↑ Global MuJoCo index
```

#### **Scene**
Container for all entities and sensors:
```python
scene = Scene(SceneCfg(
    entities={
        "robot": robot_cfg,
        "terrain": terrain_cfg,
    }
))

# Access entities by name
robot = scene["robot"]  # Returns Entity object
```

**The compilation flow:**
```python
# 1. Create scene with entity configs
scene = Scene(cfg)

# 2. Each entity loads its MjSpec (MuJoCo XML representation)
robot_spec = mujoco.MjSpec.from_file("robot.xml")

# 3. Scene merges all specs and compiles → MjModel
mj_model = scene.compile()  # ← MuJoCo assigns global indices here

# 4. Initialize entities with the compiled model
scene.initialize(mj_model)  # ← EntityIndexing reads indices

# 5. Now entity knows its indices
robot.indexing.body_names  # ["robot/torso", "robot/head", ...]
#                              These ARE the global MuJoCo indices!
```

**Key insight**: After compilation, entity components have **global MuJoCo indices**. EntityIndexing stores these for fast name-based lookup.

---

### 3. Task / Environment Layer (Highest)

**Purpose**: Define the reinforcement learning problem (MDP) and orchestrate its execution.

**Key components:**

#### **ManagerBasedRlEnv**
The main environment class that implements `gym.Env`:
```python
env = ManagerBasedRlEnv(cfg, device="cuda")
obs = env.reset()
next_obs, reward, done, info = env.step(action)
```

#### **Managers**
Execute collections of related terms:
- `ActionManager`: Converts policy outputs → sim commands
- `ObservationManager`: Extracts state → policy inputs  
- `RewardManager`: Computes scalar rewards
- `TerminationManager`: Checks episode end conditions
- `EventManager`: Handles resets, domain randomization

**How it works:**
```python
# Define task declaratively
cfg = ManagerBasedRlEnvCfg(
    scene=scene_cfg,
    actions={"joint_pos": JointPositionActionCfg(...)},
    rewards={
        "stand_reward": RewardTermCfg(
            func=mdp.body_height_reward,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso")}
        )
    },
    observations={...},
)

# Env initialization resolves configs → executable terms
env = ManagerBasedRlEnv(cfg)

# Runtime: managers call term functions
reward = env.reward_manager.compute()  # Calls body_height_reward(env, asset_cfg)
```

---

## The Core Design Pattern: Term-Based Architecture

This is the **key abstraction** that ties everything together.

### Three Phases

#### **1. Declaration (Config Time)**
You write configs declaratively:
```python
rewards = {
    "stand_reward": RewardTermCfg(
        func=mdp.body_height_reward,  # Function reference
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",           # String name
                body_names="torso"      # String pattern
            )
        }
    )
}
```

#### **2. Resolution (Initialization Time)**
Manager resolves names → indices **once**:
```python
# In Manager.__init__()
for term_name, term_cfg in cfg.rewards.items():
    # Get the asset_cfg from params
    asset_cfg = term_cfg.params["asset_cfg"]
    
    # Resolve it against the scene
    asset_cfg.resolve(scene)
    
    # What resolve() does:
    entity = scene[asset_cfg.name]  # Get robot Entity
    indices, names = entity.find_bodies("torso")  # Returns ([3], ["robot/torso"])
    asset_cfg.body_ids = indices  # Store [3] for runtime
```

**After resolution:**
- `asset_cfg.name = "robot"` (still a string)
- `asset_cfg.body_names = ("torso",)` (still strings)
- `asset_cfg.body_ids = [3]` ← **Now contains the global MuJoCo index!**

#### **3. Execution (Runtime - Every Step)**
Term functions execute with resolved configs:
```python
def body_height_reward(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # Get entity (fast dict lookup)
    entity = env.scene[asset_cfg.name]
    
    # Use pre-resolved indices (no string matching!)
    body_pos = entity.data.body_pos_w[:, asset_cfg.body_ids]  # Direct indexing
    #                                      ↑ Already [3]
    
    return body_pos[:, 2]  # Z-height of torso
```

**Why this pattern?**
- ✅ **Efficient**: Name → index mapping happens once, not every step
- ✅ **Flexible**: Support regex patterns: `".*_knee"` → `[5, 12]`
- ✅ **Declarative**: Configs are readable, shareable, modifiable
- ✅ **Type-safe**: Term functions have clean signatures

---

## SceneEntityCfg: The Connection Point

`SceneEntityCfg` is the **glue** between managers and entities.

### What it stores:

```python
@dataclass
class SceneEntityCfg:
    # Entity reference
    name: str  # "robot"
    
    # Component selection (before resolve)
    body_names: str | tuple[str, ...] | None  # "torso" or ("head", "torso")
    body_ids: list[int] | slice  # slice(None) by default
    
    # Same pattern for joints, geoms, sites
    joint_names: ...
    joint_ids: ...
```

### What resolve() does:

```python
def resolve(self, scene: Scene) -> None:
    entity = scene[self.name]
    
    # If body_names provided, resolve to indices
    if self.body_names is not None:
        indices, names = entity.find_bodies(self.body_names)
        self.body_ids = indices  # Store for runtime
    
    # Repeat for joints, geoms, sites...
```

### Three resolution cases:

**Case 1: Only names provided**
```python
cfg = SceneEntityCfg(name="robot", body_names="torso")
cfg.resolve(scene)
# Result: cfg.body_ids = [3]
```

**Case 2: Only IDs provided**
```python
cfg = SceneEntityCfg(name="robot", body_ids=[3, 5])
cfg.resolve(scene)
# Result: cfg.body_names = ("torso", "head")  # Looks up names
```

**Case 3: Both provided**
```python
cfg = SceneEntityCfg(name="robot", body_names="torso", body_ids=[3])
cfg.resolve(scene)
# Result: Validates they match, raises error if inconsistent
```

---

## The Complete Initialization Flow

Here's how everything connects when you create an environment:

```python
# 1. USER: Define config
cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(entities={"robot": robot_cfg}),
    rewards={"stand": RewardTermCfg(func=..., params={"asset_cfg": SceneEntityCfg("robot", body_names="torso")})}
)

# 2. ENV: Initialize
env = ManagerBasedRlEnv(cfg, device="cuda")

    # 2a. Create scene
    scene = Scene(cfg.scene, device="cuda")
        # Each entity creates its MjSpec
        robot.spec = mujoco.MjSpec.from_file("robot.xml")
    
    # 2b. Compile scene → MjModel
    mj_model = scene.compile()
        # MuJoCo assigns global indices to all bodies, joints, etc.
        # mj_model.names().body_names = ["world", "robot/torso", "robot/head", ...]
        #                                   0          1              2
    
    # 2c. Create simulation
    sim = Simulation(num_envs=4096, model=mj_model, device="cuda")
    
    # 2d. Initialize scene with compiled model
    scene.initialize(mj_model, sim.model, sim.data)
        # Each entity creates EntityIndexing
        robot.indexing = EntityIndexing(mj_model, name_prefix="robot")
            # Stores: body_names = ["robot/torso", "robot/head"]
            #         These ARE the global MuJoCo indices [1, 2]
    
    # 2e. Load managers
    env.load_managers()
        reward_manager = RewardManager(cfg.rewards, env)
            # For each term, resolve SceneEntityCfg
            asset_cfg = term.params["asset_cfg"]
            asset_cfg.resolve(scene)
                entity = scene["robot"]
                indices, names = entity.find_bodies("torso")  # Returns ([1], ["robot/torso"])
                asset_cfg.body_ids = [1]  # STORED

# 3. RUNTIME: Step environment
obs, reward, done, info = env.step(action)
    # Reward manager computes rewards
    reward = reward_manager.compute()
        # Calls: body_height_reward(env, asset_cfg)
        # asset_cfg.body_ids is already [1] - no lookup needed!
```

---

## Key Takeaways

1. **Three layers** work together but stay decoupled:
   - Simulation: Physics (doesn't know about robots)
   - Scene: Objects (doesn't know about tasks)
   - Task: MDP (uses scene and sim)

2. **Compilation creates the index mapping**:
   - XML → MjSpec → `compile()` → MjModel (indices assigned)
   - EntityIndexing reads these indices from MjModel

3. **Term-based pattern** separates concerns:
   - Config: What to compute (declarative)
   - Resolution: Map names → indices (once at init)
   - Execution: Fast computation (every step)

4. **SceneEntityCfg** is the bridge:
   - Stores entity name + component patterns
   - Resolves to indices during init
   - Passed to term functions at runtime

---

## What's Next?

Now that you understand the architecture, you can dive deeper:

- **[Scene Layer](scene_layer.md)**: Entity, compilation, and indexing details
- **[Task Layer](task_layer.md)**: Managers, terms, and MDP configuration  
- **[Simulation Layer](simulation_layer.md)**: MuJoCo Warp and GPU parallelization

Or explore specific topics:
- How to create custom reward terms
- How domain randomization works
- How to add new sensors
- How to integrate mjlab with colosseum