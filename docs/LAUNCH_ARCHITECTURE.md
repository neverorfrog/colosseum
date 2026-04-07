# Launch Architecture: Application-Owned Stacks

## Background: How Circus Launches Robots

Circus is a MuJoCo physics simulator that manages one Docker container per robot. This is not optional — Docker is intrinsic to how Circus works. The container runs `supervisord`, which starts booster-motion, simbridge, and the application binary. All three processes share the container's network namespace, which is required because DDS (FastDDS) communicates over loopback (`127.0.0.1`).

```
Host
├── circus (Qt GUI, MuJoCo physics)          ← TCP server on port 5555
└── Docker container (per robot)
      └── supervisord
            ├── booster-motion               ← reads DDS, computes PD torques
            ├── simbridge_node               ← bridges TCP↔DDS
            └── <app binary>                 ← policy node / behavior framework
```

The container connects back to circus via `SERVER_IP=172.17.0.1` (Docker bridge gateway).

---

## Current Architecture (Problems)

Circus is currently the **entry point** and **orchestrator**:

1. Circus reads `resources/config/path_constants.yaml` — absolute paths to all sibling repos, **maintained manually**
2. Circus reads `resources/config/framework_config.yaml` — Docker image + volume binds, using `<repo>` placeholders substituted from path_constants
3. Circus reads `resources/scenes/*.yaml` — which robots to spawn
4. Circus creates one Docker container per robot, volume-mounting:
   - `<simbridge>/tools/booster_motion` → `/app/booster_motion`
   - `<simbridge>/.pixi/envs/default` → `/app/bridge`
   - `<maximus>/.pixi/envs/default` → `/app/maximus`
5. Container's `entrypoint.sh` + `booster.conf` are baked into the Docker image

**Problems:**
- Adding a new application requires modifying Circus's resources
- `path_constants.yaml` is manually maintained and machine-specific
- `booster.conf` and `entrypoint.sh` live in Circus, but their content depends entirely on which application is running
- Volume mounts span multiple repos (simbridge, maximus, colosseum) — fragile, hard to track
- The scene (robot count, positions) is in Circus, but it is scenario-specific to the application

---

## Proposed Architecture

**Each application owns its full launch stack.** The application's `main` binary is the entry point. Circus becomes a reusable simulator with no knowledge of who is using it.

### Dependency Model

Each application declares its full stack as **pixi/conda dependencies**:

```toml
# sim2sim/pixi.toml
[dependencies]
circus                  = "*"   # from spqr channel
simbridge               = "*"   # from spqr channel
booster_robotics_sdk    = "==1.5.0"   # from spqr channel
booster_robotics_sdk_ros2 = "*" # from spqr channel (provides booster-motion binary)
sim2sim                 = { path = "." }
```

After `pixi install`, every binary lives under one root:
```
sim2sim/.pixi/envs/default/
  bin/
    circus               ← Qt simulator
    simbridge_node       ← DDS bridge
    booster-motion       ← PD controller
    main                 ← policy node (sim2sim)
  lib/
    ...                  ← all shared libraries, rpaths resolved by conda
  share/
    sim2sim/configs/     ← installed application configs
```

This replaces the current fragmented volume model (four separate repo mounts) with **a single pixi environment**.

### Launch Flow

```
User
 └─► pixi run launch
       ├─ 1. Writes /tmp/path_constants_<pid>.yaml  (just sim2sim_env path)
       ├─ 2. Launches circus:
       │       circus --scene configs/scenes/training_1v1.yaml
       │              --simulation-config configs/simulation/training.yaml
       │              --framework-config configs/framework_config.yaml
       │              --path-constants /tmp/path_constants_<pid>.yaml
       └─ Circus reads configs, spawns Docker container per robot:
               └─► supervisord (from app's booster.conf)
                     ├─► booster-motion   /app/env/bin/booster-motion
                     ├─► simbridge_node   /app/env/bin/simbridge_node
                     └─► sim2sim main     /app/env/bin/main
```

### path_constants.yaml (Auto-Generated, Simplified)

Instead of mapping every sibling repo, the application generates a single-entry file:

```yaml
# /tmp/path_constants_<pid>.yaml  — written at runtime, never committed
sim2sim_env: /home/user/code/spqr/colosseum/src/sim2sim/.pixi/envs/default
```

Circus substitutes `<sim2sim_env>` in `framework_config.yaml` at runtime.

### framework_config.yaml (Single Volume Mount)

```yaml
# sim2sim/configs/framework_config.yaml
image: spqr:booster
volumes:
  - "<sim2sim_env>:/app/env"
  - "/dev/shm/circus_ipc:/dev/shm/circus_ipc"
```

One mount replaces the current four. All binaries and libraries are accessible at `/app/env/`.

---

## Per-Application Config Structure

### sim2sim (RL policy testing)

```
colosseum/src/sim2sim/
├── pixi.toml                          # declares circus, simbridge, booster_robotics_sdk,
│                                      # booster_robotics_sdk_ros2, sim2sim as dependencies
├── configs/
│   ├── framework_config.yaml          # image + single <sim2sim_env> volume
│   ├── booster.conf                   # supervisord: booster-motion, simbridge, sim2sim main
│   ├── entrypoint.sh                  # sets FASTRTPS_DEFAULT_PROFILES_FILE; starts supervisord
│   ├── scenes/
│   │   └── training_1v1.yaml          # 1 robot, RL spawn position
│   ├── simulation/
│   │   └── training.yaml              # no game phases, unlimited time
│   └── booster_motion/
│       ├── config_t1_rl.lua           # entry: loads graph + options below
│       ├── common_graph_define_t1_rl.lua
│       └── common_module_options_t1_rl.lua
└── src/
    └── main.cpp                       # generates path_constants, launches circus, then runs policy loop
```

**`booster.conf` for sim2sim:**
```ini
[program:booster-motion]
directory=/app/env
command=/app/env/bin/booster-motion -mode sim -config /app/env/share/sim2sim/configs/booster_motion/config_t1_rl.lua
priority=1
autostart=true

[program:delayed-starter]
command=/app/delayed_start.sh
priority=2
autostart=true
autorestart=false

[program:simbridge]
command=/app/env/bin/simbridge_node
autostart=false
autorestart=true

[program:sim2sim]
command=/app/env/bin/main
environment=FASTRTPS_DEFAULT_PROFILES_FILE="/app/env/share/booster_motion/fastdds_profile.xml"
autostart=false
autorestart=true
```

**`entrypoint.sh` for sim2sim:**
```bash
#!/bin/bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/app/env/share/booster_motion/fastdds_profile.xml
/usr/bin/supervisord -n -c /app/env/share/sim2sim/configs/booster.conf
```

**`training_1v1.yaml` scene:**
```yaml
simulation_config: training
teams:
  red:
    - type: Booster-T1
      number: 1
      position: [0.0, 0.0, 0.68]
      orientation: [0.0, 0.0, 0.0]
```

**`training.yaml` simulation config:**
```yaml
simulation:
  max_simulation_time: -1
game:
  field: fieldAdultSize
  game_duration: -1
  automatic_restart: true
```

---

### maximus (behavior framework testing)

```
spqrbooster2026/
├── pixi.toml                          # declares circus, simbridge, booster_robotics_sdk,
│                                      # booster_robotics_sdk_ros2, maximus as dependencies
├── configs/
│   ├── framework_config.yaml          # image + <maximus_env> volume + TensorRT volume
│   ├── booster.conf                   # supervisord: booster-motion, simbridge, maximus
│   ├── entrypoint.sh                  # sets FASTRTPS, SPQR_CONFIG_ROOT, SPQR_BEHAVIOR_TREE_PATH
│   ├── scenes/
│   │   ├── 1v1.yaml
│   │   └── 5v5.yaml
│   ├── simulation/
│   │   └── default.yaml               # full game settings
│   └── booster_motion/
│       └── config_isaac.lua           # standard behavior graph
└── src/app/
    └── main.cpp                       # generates path_constants, launches circus
```

**`framework_config.yaml` for maximus:**
```yaml
image: spqr:booster
volumes:
  - "<maximus_env>:/app/env"
  - "<maximus_env>/tools/vision/tensorrt:/app/tensorrt"
  - "/dev/shm/circus_ipc:/dev/shm/circus_ipc"
```

---

## Docker Image Strategy

The `spqr:booster` base image becomes **minimal**: only ROS2 Humble, system libraries, and `supervisord`. It no longer contains booster SDK builds (those come from the pixi env volume mount).

```dockerfile
FROM ros:humble
RUN apt-get update && apt-get install -y supervisor libpulse0 ... && rm -rf /var/lib/apt/lists/*
# No booster SDK builds — everything comes from the volume-mounted pixi env
COPY delayed_start.sh /app/delayed_start.sh
```

The `entrypoint.sh` and `booster.conf` are no longer baked into the image — they are owned by the application and referenced from the mounted volume or passed via a separate bind.

---

## Implementation Steps

### Step 1 — Circus accepts external config paths via CLI

```bash
circus \
  --scene /path/to/app/configs/scenes/training_1v1.yaml \
  --simulation-config /path/to/app/configs/simulation/training.yaml \
  --framework-config /path/to/app/configs/framework_config.yaml \
  --path-constants /tmp/path_constants_<pid>.yaml
```

Changes in Circus C++:
- `Constants.h`: replace hardcoded resource paths with runtime values from CLI
- `SceneParser`: accept scene file path as constructor argument
- `RobotManager::startContainers()`: accept framework config and path constants paths

### Step 2 — Application generates path_constants.yaml at startup

```cpp
// In sim2sim main.cpp
std::filesystem::path env_prefix = get_conda_prefix();  // or derive from argv[0]

YAML::Node paths;
paths["sim2sim_env"] = env_prefix.string();

std::string out = "/tmp/path_constants_" + std::to_string(getpid()) + ".yaml";
write_yaml(paths, out);
```

### Step 3 — Application launches circus as subprocess

```cpp
std::string cmd = env_prefix / "bin/circus";
cmd += " --scene " + config_dir + "/scenes/training_1v1.yaml";
cmd += " --framework-config " + config_dir + "/framework_config.yaml";
cmd += " --path-constants " + path_constants_file;
std::system(cmd.c_str());  // or fork/exec with proper signal handling
```

### Step 4 — Package conda packages for the spqr channel

| Package | Content |
|---------|---------|
| `circus` | circus binary + MuJoCo assets + field configs |
| `simbridge` | `simbridge_node` binary + booster_interface ROS2 msgs |
| `booster_robotics_sdk` | C++ headers + static lib (already exists on spqr channel) |
| `booster_robotics_sdk_ros2` | `booster-motion` binary + Lua configs + FastDDS profile |

### Step 5 — Move configs into application repos

| File | From | To |
|------|------|----|
| `framework_config.yaml` | `circus/resources/config/` | `sim2sim/configs/` and `maximus/configs/` |
| `booster.conf` | `circus/dockerfiles/` | `sim2sim/configs/` and `maximus/configs/` |
| `entrypoint.sh` | `circus/dockerfiles/` | `sim2sim/configs/` and `maximus/configs/` |
| `scenes/*.yaml` | `circus/resources/scenes/` | `sim2sim/configs/scenes/` and `maximus/configs/scenes/` |
| `simulation_configs/` | `circus/resources/config/` | `sim2sim/configs/simulation/` and `maximus/configs/simulation/` |
| RL Lua configs | `simbridge/tools/booster_motion/configs/` | `sim2sim/configs/booster_motion/` |

---

## What Circus Loses

After this change, `circus/resources/` contains only:
- `config/fields/*.yaml` — standard RoboCup field geometries (fixed by league rules)
- `robots/*/` — MuJoCo robot XML assets

Everything else moves to the application repos. Circus becomes a reusable simulator binary.

---

## Migration Checklist

- [ ] Add CLI args to Circus: `--scene`, `--simulation-config`, `--framework-config`, `--path-constants`
- [ ] Package `circus` as conda package (spqr channel)
- [ ] Package `simbridge` as conda package (spqr channel)
- [ ] Package `booster_robotics_sdk_ros2` as conda package (spqr channel, ships `booster-motion`)
- [ ] Add `circus`, `simbridge`, `booster_robotics_sdk_ros2` to sim2sim `pixi.toml`
- [ ] Add `circus`, `simbridge`, `booster_robotics_sdk_ros2` to maximus `pixi.toml`
- [ ] Create `sim2sim/configs/` with `framework_config.yaml`, `booster.conf`, `entrypoint.sh`
- [ ] Create `sim2sim/configs/scenes/training_1v1.yaml`
- [ ] Create `sim2sim/configs/simulation/training.yaml`
- [ ] Move RL Lua configs to `sim2sim/configs/booster_motion/`
- [ ] Add path_constants generation + circus launch to `sim2sim/src/main.cpp`
- [ ] Create `maximus/configs/` with equivalent files
- [ ] Slim down `spqr:booster` Dockerfile (remove SDK builds, keep only system deps + supervisord)
- [ ] Remove `framework_config.yaml`, `booster.conf`, `entrypoint.sh` from Circus repo
- [ ] Remove `scenes/` and `simulation_configs/` from Circus repo
- [ ] Keep `path_constants.yaml` in Circus only as a fallback default (for standalone use)
