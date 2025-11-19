# Deep Dive: Simulation Layer

Let me search for detailed information about the simulation layer and MuJoCo Warp integration.Perfect! Now I have comprehensive information about the simulation layer. Let me give you a detailed deep dive.

---

# 🔬 **Simulation Layerr - Deep Dive**

## **1. What is MuJoCo Warp?**

**MuJoCo Warp (mjwarp)** is a GPU-accelerated batched version of MuJoCo built on top of **NVIDIA Warp**. Here's what makes it special:

### **Key Concepts:**

**Standard MuJoCo:**
```python
# CPU-based, single environment
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)
mujoco.mj_step(model, data)  # One env, runs on CPU
```

**MuJoCo Warp:**
```python
# GPU-based, batched environments
wp_model = mjwarp.put_model(mj_model)  # Upload to GPU
wp_data = mjwarp.put_data(mj_model, mj_data, nworld=4096)  # 4096 parallel worlds
mjwarp.step(wp_model, wp_data)  # All 4096 envs step in parallel on GPU!
```

### **What is "nworld"?**
- **nworld** = number of parallel environments (worlds)
- Each world has its own state: `qpos`, `qvel`, `qacc`, `ctrl`, etc.
- All worlds share the **same model** (same robot, same physics parameters)
- Arrays are shaped: `(nworld, original_dim)`

Example:
```python
qpos.shape = (4096, nq)  # 4096 environments, nq joint positions each
qvel.shape = (4096, nv)  # 4096 environments, nv velocities each
```

---

## **2. Simulation Class Architecture**

Let's break down the `Simulation` class step by step:

### **Initialization Flow:**

```python
class Simulation:
    def __init__(self, num_envs: int, cfg: SimulationCfg, 
                 model: mujoco.MjModel, device: str):
```

**Step 1: Setup Device**
```python
self.device = device  # e.g., "cuda:0"
self.wp_device = wp.get_device(device)  # Warp device object
self.num_envs = num_envs  # Number of parallel environments
```

**Step 2: Create MuJoCo CPU Model (Template)**
```python
self._mj_model = model  # Standard MuJoCo model (CPU)
self._mj_data = mujoco.MjData(model)  # Single env data (CPU)
mujoco.mj_forward(self._mj_model, self._mj_data)  # Initialize
```
- This creates the "template" that will be replicated across all GPU environments

**Step 3: Upload to GPU with Warp**
```python
with wp.ScopedDevice(self.wp_device):
    # Upload model to GPU
    self._wp_model = mjwarp.put_model(self._mj_model)
    
    # Configure options
    self._wp_model.opt.ls_parallel = cfg.ls_parallel  # Parallel linesearch
    
    # Upload data and create nworld copies
    self._wp_data = mjwarp.put_data(
        self._mj_model,
        self._mj_data,
        nworld=self.num_envs,  # Create 4096 parallel worlds
        nconmax=cfg.nconmax,   # Max contacts per world
        njmax=cfg.njmax         # Max constraints per world
    )
```

**Step 4: Create Bridges (PyTorch Interop)**
```python
self._model_bridge = WarpBridge(self._wp_model, nworld=self.num_envs)
self._data_bridge = WarpBridge(self._wp_data)
```
- These wrap Warp arrays to make them behave like PyTorch tensors
- Enables zero-copy memory sharing between Warp and PyTorch

**Step 5: CUDA Graph Optimization**
```python
self.use_cuda_graph = (
    self.wp_device.is_cuda and 
    wp.is_mempool_enabled(self.wp_device)
)
self.create_graph()  # Pre-record GPU operations
```

**Step 6: NaN Guard**
```python
self.nan_guard = NanGuard(cfg.nan_guard, self.num_envs, self._mj_model)
```
- Debugging tool to detect and log NaN/Inf values

---

## **3. CUDA Graphs - Performance Optimization**

### **What are CUDA Graphs?**

CUDA graphs **record** a sequence of GPU operations once, then **replay** them with near-zero overhead.

**Without CUDA Graphs:**
```python
# Every step incurs kernel launch overhead
for _ in range(1000):
    mjwarp.step(model, data)  # Launch kernels, CPU overhead each time
```

**With CUDA Graphs:**
```python
# Record once
with wp.ScopedCapture() as capture:
    mjwarp.step(model, data)  # Record all kernels
step_graph = capture.graph

# Replay 1000x with minimal overhead
for _ in range(1000):
    wp.capture_launch(step_graph)  # Just replay, super fast!
```

**In mjlab:**
```python
def create_graph(self):
    if self.use_cuda_graph:
        # Record step graph
        with wp.ScopedCapture() as capture:
            mjwarp.step(self.wp_model, self.wp_data)
        self.step_graph = capture.graph
        
        # Record forward graph
        with wp.ScopedCapture() as capture:
            mjwarp.forward(self.wp_model, self.wp_data)
        self.forward_graph = capture.graph
```

**Performance Impact:**
- ~2-3x speedup for batched simulations
- Reduces CPU-GPU synchronization overhead
- Critical for high-throughput RL training

---

## **4. Core Simulation Methods**

### **forward() - Compute forward kinematics**
```python
def forward(self):
    with wp.ScopedDevice(self.wp_device):
        if self.use_cuda_graph and self.forward_graph is not None:
            wp.capture_launch(self.forward_graph)  # Use cached graph
        else:
            mjwarp.forward(self.wp_model, self.wp_data)  # Run directly
```

**What it does:**
- Computes derived quantities from `qpos`, `qvel`
- Updates: `xpos`, `xmat`, `xquat` (body positions/orientations)
- Updates: `xipos`, `ximat` (body inertial frame positions/orientations)
- Does NOT integrate physics or apply forces

**When to use:**
- After manually setting `qpos`/`qvel`
- To update visualization without stepping physics

### **step() - Integrate physics**
```python
def step(self):
    with wp.ScopedDevice(self.wp_device):
        with self.nan_guard.watch(self.data):  # Detect NaNs
            if self.use_cuda_graph and self.step_graph is not None:
                wp.capture_launch(self.step_graph)
            else:
                mjwarp.step(self.wp_model, self.wp_data)
```

**What it does:**
- **Complete physics step**: forward → inverse → forces → constraints → integration
- Updates `qpos`, `qvel` based on `ctrl` (control inputs)
- Resolves contacts, applies constraints
- Advances simulation by `dt` (timestep)

**The Full MuJoCo Pipeline:**
```
1. mj_forward()    - Kinematics from qpos/qvel
2. mj_inverse()    - Compute qacc from forces
3. mj_collision()  - Detect collisions
4. mj_sensor()     - Update sensor readings  
5. mj_integrate()  - Update qpos/qvel
```

---

## **5. WarpBridge - PyTorch Interoperability**

The **WarpBridge** and **TorchArray** classes enable seamless PyTorch operations on Warp data.

### **Problem:**
- Warp uses `wp.array` (GPU arrays)
- PyTorch uses `torch.Tensor`
- Need to share memory without copying

### **Solution: Zero-Copy Wrapping**

```python
class TorchArray:
    def __init__(self, wp_array: wp.array, nworld: int | None = None):
        self._wp_array = wp_array
        self._tensor = wp.to_torch(wp_array)  # Zero-copy conversion!
```

**Key Features:**

**1. Zero-Copy Memory Sharing:**
```python
# wp_data.qpos is a Warp array
# But accessing it returns a TorchArray that wraps a PyTorch tensor
qpos = sim.data.qpos  # TorchArray wrapping torch.Tensor
qpos[0, :] = new_positions  # Writes directly to GPU memory!
```

**2. Automatic Broadcasting for Domain Randomization:**
```python
# If nworld > 1 and first dim has stride 0 (shared across envs)
if self._tensor.stride(0) == 0 and self._tensor.shape[0] == 1:
    new_shape = (nworld,) + self._tensor.shape[1:]
    self._tensor = self._tensor.expand(new_shape)
```
This handles model fields that are shared initially but can be expanded for per-env randomization.

**3. PyTorch Operations Work Seamlessly:**
```python
# All PyTorch operations work!
qpos = sim.data.qpos  # TorchArray
mean_qpos = qpos.mean(dim=0)  # Works!
qpos_clipped = torch.clamp(qpos, min=-1, max=1)  # Works!
loss = (qpos - target).pow(2).sum()  # Works!
```

**4. Read-Only for CUDA Graph Safety:**
```python
# This is FORBIDDEN (would break CUDA graphs):
sim.data.qpos = new_tensor  # AttributeError!

# This is CORRECT (in-place modification):
sim.data.qpos[:] = new_tensor  # ✓ Memory address unchanged
```

### **WarpBridge Class:**

```python
class WarpBridge:
    """Wraps mjwarp objects to expose Warp arrays as PyTorch tensors."""
    
    def __getattr__(self, name: str):
        # Cache wrapped arrays
        if name in self._wrapped_cache:
            return self._wrapped_cache[name]
        
        val = getattr(self._struct, name)
        
        # Wrap Warp arrays as TorchArray
        if isinstance(val, wp.array):
            wrapped = TorchArray(val, nworld=self._nworld)
            self._wrapped_cache[name] = wrapped
            return wrapped
        
        return val
```

**Usage:**
```python
# Accessing data through bridge
sim.data.qpos  # Returns TorchArray (behaves like torch.Tensor)
sim.data.qvel  # Returns TorchArray
sim.data.ctrl  # Returns TorchArray

# All are zero-copy, directly accessing GPU memory!
```

---

## **6. Data Layout & Key Arrays**

### **Configuration Arrays (Model):**
```python
# Stored in sim.model (mjwarp.Model)
model.body_mass        # (nbody,) - mass of each body
model.body_pos         # (nbody, 3) - position in parent frame
model.jnt_range        # (nv, 2) - joint limits [min, max]
model.actuator_gainprm # (nu, 10) - actuator gain parameters
model.geom_friction    # (ngeom, 3) - friction coefficients
```

### **State Arrays (Data):**
```python
# Stored in sim.data (mjwarp.Data)
# All have shape (nworld, ...)

# --- Configuration Space ---
data.qpos    # (nworld, nq) - generalized positions
data.qvel    # (nworld, nv) - generalized velocities
data.qacc    # (nworld, nv) - generalized accelerations

# --- Control & Actuation ---
data.ctrl    # (nworld, nu) - control inputs to actuators
data.actuator_force  # (nworld, nu) - forces produced by actuators

# --- Cartesian Space ---
data.xpos    # (nworld, nbody, 3) - body positions (world frame)
data.xmat    # (nworld, nbody, 3, 3) - body orientations (rotation matrices)
data.xquat   # (nworld, nbody, 4) - body orientations (quaternions)
data.xipos   # (nworld, nbody, 3) - body inertial frame positions
data.cvel    # (nworld, nbody, 6) - body velocities (CoM frame)

# --- Geometry ---
data.geom_xpos  # (nworld, ngeom, 3) - geom positions (world)
data.geom_xmat  # (nworld, ngeom, 3, 3) - geom orientations

# --- Sites (markers/sensors) ---
data.site_xpos  # (nworld, nsite, 3) - site positions
data.site_xmat  # (nworld, nsite, 3, 3) - site orientations

# --- External Forces ---
data.xfrc_applied  # (nworld, nbody, 6) - external forces/torques

# --- Contacts ---
data.contact       # Contact information (complex structure)
data.ncon          # (nworld,) - number of contacts per world

# --- Other ---
data.subtree_com   # (nworld, nbody, 3) - subtree center of mass
data.qfrc_applied  # (nworld, nv) - applied forces in joint space
```

### **Understanding qpos vs joint_pos:**

**qpos** (generalized positions):
- Includes ALL degrees of freedom
- For floating base: `[root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, joint1, joint2, ...]`
- Shape: `(nworld, nq)` where `nq = 7 (if floating) + num_joints`

**joint_pos** (just articulated joints):
- Only the actuated/fixed joints
- Excludes the free joint (floating base)
- Shape: `(nworld, nv)` where `nv = num_joints`

**Accessing through Entity:**
```python
robot.data.joint_pos  # Just joints: (nworld, nv)
robot.data.root_link_pos_w  # Just root position: (nworld, 3)
robot.data.root_link_quat_w  # Just root orientation: (nworld, 4)
```

---

## **7. Domain Randomization**

One of mjlab's coolest features: **per-environment model parameters**!

### **How it Works:**

**Step 1: Identify Fields to Randomize**
```python
# EventManager collects fields that need per-env randomization
fields = ["geom_friction", "body_mass", "actuator_gainprm"]
```

**Step 2: Expand Model Fields**
```python
sim.expand_model_fields(fields)
```

**Step 3: The Magic - repeat_array_kernel**
```python
@wp.kernel
def repeat_array_kernel(src, nelems_per_world, dst):
    tid = wp.tid()
    src_idx = tid % nelems_per_world
    dst[tid] = src[src_idx]
```

**What happens:**
```python
# BEFORE expansion:
geom_friction.shape = (1, ngeom, 3)  # Shared across all envs
geom_friction.stride(0) = 0  # Stride-0 means shared

# AFTER expansion:
geom_friction.shape = (nworld, ngeom, 3)  # Per-env values!
geom_friction.stride(0) != 0  # Now can be modified per-env
```

**Step 4: Randomize Per Environment**
```python
# Now you can set different values for each environment!
for i in range(num_envs):
    sim.model.geom_friction[i, :, 0] = random.uniform(0.5, 1.5)
```

**Real Example from mjlab:**
```python
# In EventManager - randomize foot friction
def randomize_friction(env):
    robot = env.scene["robot"]
    foot_geom_ids = robot.indexing.geom_ids[foot_indices]
    
    # Generate random friction for each environment
    random_friction = torch.rand(num_envs, device=device) * 0.9 + 0.3  # [0.3, 1.2]
    
    # Apply to each environment
    env.sim.model.geom_friction[:, foot_geom_ids, 0] = random_friction.unsqueeze(-1)
```

---

## **8. Configuration - SimulationCfg**

### **MujocoCfg - Physics Parameters:**

```python
@dataclass
class MujocoCfg:
    # Time
    timestep: float = 0.002  # 2ms physics timestep (500 Hz)
    
    # Integration
    integrator: "euler" | "implicitfast" = "implicitfast"
    # - euler: explicit Euler (fast, less stable)
    # - implicitfast: semi-implicit (stable, recommended)
    
    # Solver
    solver: "newton" | "cg" | "pgs" = "newton"
    # - newton: Newton-Raphson (accurate, slower)
    # - cg: Conjugate Gradient (fast, good for many contacts)
    # - pgs: Projected Gauss-Seidel (fastest, less accurate)
    
    iterations: int = 100  # Max solver iterations
    tolerance: float = 1e-8  # Convergence tolerance
    
    ls_iterations: int = 50  # Line search iterations
    ls_tolerance: float = 0.01  # Line search tolerance
    
    # Friction
    cone: "pyramidal" | "elliptic" = "pyramidal"
    # - pyramidal: 4-sided friction cone (faster)
    # - elliptic: smooth friction cone (more accurate)
    
    impratio: float = 1.0  # Ratio of friction coefficients
    
    # Other
    gravity: tuple = (0, 0, -9.81)  # m/s²
```

### **SimulationCfg - Batch Settings:**

```python
@dataclass
class SimulationCfg:
    # Contact/constraint allocation per world
    nconmax: int | None = None  # Max contacts per env
    njmax: int | None = None    # Max constraints per env
    
    # Performance
    ls_parallel: bool = True  # Parallel line search (faster!)
    
    # Sensors
    contact_sensor_maxmatch: int = 64  # Max contact sensor pairs
    
    # Debugging
    nan_guard: NanGuardCfg = field(default_factory=NanGuardCfg)
```

**Why nconmax and njmax matter:**
- Contacts are stored in **heterogeneous arrays**
- Different environments can have different numbers of contacts
- But we need to allocate enough space for the worst case
- If `None`, mjlab uses heuristics based on model complexity

---

## **9. NaN Guard - Debugging Tool**

When training fails due to NaN, NaN Guard saves you:

### **How It Works:**

```python
class NanGuard:
    def __init__(self, cfg, num_envs, mj_model):
        self.buffer = deque(maxlen=cfg.buffer_size)  # Rolling buffer
        
    def capture(self, wp_data):
        """Save current state to buffer"""
        # Extract qpos, qvel, act for all envs
        states = np.empty((num_envs, state_size))
        for i in range(num_envs):
            mujoco.mj_getState(model, data, states[i], mjSTATE_PHYSICS)
        self.buffer.append({"step": step, "states": states})
    
    @contextmanager
    def watch(self, wp_data):
        """Capture before, check after"""
        self.capture(wp_data)  # Before step
        yield
        self.check_and_dump(wp_data)  # After step
```

**Usage in Simulation:**
```python
def step(self):
    with self.nan_guard.watch(self.data):  # Automatic capture & check
        mjwarp.step(self.wp_model, self.wp_data)
```

**When NaN Detected:**
```python
# Dumps to /tmp/mjlab/nan_dumps/nan_dump_20250103_123456.npz
{
    "states_step_000000": states[bad_envs],  # 100 steps before NaN
    "states_step_000001": states[bad_envs],
    ...
    "states_step_000099": states[bad_envs],  # Last good state
    "_metadata": {
        "nan_env_ids": [42, 103, ...],  # Which envs had NaN
        "detection_step": 12543,
        "model_file": "model_20250103_123456.mjb"
    }
}
```

**Replay NaN States:**
```bash
python scripts/nan_viz.py /tmp/mjlab/nan_dumps/nan_dump_20250103_123456.npz
```

---

## **10. Performance Characteristics**

### **Typical Setup:**
```python
num_envs = 4096
physics_dt = 0.002  # 500 Hz
control_dt = 0.01   # 100 Hz (decimation = 5)
```

### **Performance Benchmarks (RTX 4090):**

| Configuration | Physics Steps/sec | Speedup vs CPU |
|--------------|------------------|----------------|
| 1 env (CPU)  | ~5,000           | 1x             |
| 1024 envs (GPU) | ~2,000,000    | 400x           |
| 4096 envs (GPU) | ~6,000,000    | 1200x          |

**With CUDA Graphs:**
- Additional ~2-3x speedup
- ~15,000,000 steps/sec with 4096 envs!

### **Memory Usage:**
```python
# Rough estimate per environment:
memory_per_env = (
    nq * 8 +      # qpos (float64)
    nv * 8 +      # qvel (float64)
    nu * 8 +      # ctrl (float64)
    nbody * 24 +  # xpos (3 * float64)
    nbody * 72 +  # xmat (9 * float64)
    ...
)
# Typical humanoid: ~50 KB/env
# 4096 envs: ~200 MB
```

---

## **11. Key Insights**

### **Why is Warp + MuJoCo So Fast?**

1. **GPU Parallelism**: 4096 envs run simultaneously on thousands of CUDA cores
2. **Batched Operations**: Single kernel launch for all envs
3. **CUDA Graphs**: Zero kernel launch overhead
4. **Memory Locality**: All data in GPU memory (no CPU-GPU transfers)
5. **Optimized Solver**: Parallel line search, efficient constraint solving

### **Design Trade-offs:**

**Pros:**
- ✅ Massive throughput (millions of steps/sec)
- ✅ Perfect for RL training (need lots of samples)
- ✅ Zero-copy PyTorch integration
- ✅ Per-env domain randomization

**Cons:**
- ❌ All envs must have same model structure
- ❌ Memory overhead for batching
- ❌ Contact/constraint limits are per-env, must allocate for worst case
- ❌ CUDA graphs require fixed memory addresses (why WarpBridge is read-only)

---

## **12. Common Patterns**

### **Reading State:**
```python
# Through Entity (recommended - high-level)
qpos = robot.data.joint_pos  # (num_envs, nv)
qvel = robot.data.joint_vel  # (num_envs, nv)

# Through Simulation (low-level)
qpos = sim.data.qpos[:, robot.indexing.joint_q_adr]  # Manual indexing
```

### **Writing State:**
```python
# Through Entity
robot.write_joint_position_to_sim(new_qpos, joint_ids=None, env_ids=[0, 1, 2])

# Through Simulation (direct)
sim.data.qpos[:, joint_q_adr] = new_qpos  # In-place!
```

### **Stepping:**
```python
# High-level (through Environment)
env.step(action)  # Handles decimation, events, etc.

# Low-level (direct)
sim.data.ctrl[:] = control_inputs
sim.step()  # Single physics step
```

---

This is the foundation everything else builds on. The simulation layer handles **all** the heavy lifting of running thousands of parallel physics simulations on the GPU!

Want to dive into how the Entity layer works next? Or any specific aspect of the simulation layer you want to explore further?