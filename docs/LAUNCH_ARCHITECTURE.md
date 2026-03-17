# Launch Architecture: Application-Owned Stacks

## Current Architecture

Circus is the **entry point** for the entire SPQR simulation stack. The user starts circus, which:

1. Reads its own configs (`path_constants.yaml`, `framework_config.yaml`, `scenes/*.yaml`, `simulation_configs/*.yaml`)
2. Launches a Docker container **per robot**, using volume mounts derived from `path_constants.yaml`
3. Each container runs **supervisord** (`booster.conf`), which starts:
   - `booster-motion` (priority 1, always on)
   - `simbridge` (connects container back to circus via TCP:5555)
   - `colosseum` or `maximus` (optional, disabled by default)

```
User
 └─► circus (Qt GUI, host)
       ├─ reads: path_constants.yaml, framework_config.yaml
       ├─ reads: scenes/1v1.yaml → field + robot list
       └─► docker run spqr:booster  (per robot)
             └─► supervisord (booster.conf)
                   ├─► booster-motion -config config_isaac.lua
                   ├─► simbridge_node
                   └─► colosseum/bin/main  OR  maximus/bin/maximus_main
```

**Problems with this model:**

- Circus owns knowledge of where every other repo lives (`path_constants.yaml` is a manual file in circus's resources)
- `framework_config.yaml` (Docker volumes) is also in circus — but which volumes are needed depends entirely on which application is running
- `booster.conf` enables/disables processes by commenting them out manually
- Adding a new application means modifying circus's configs
- Circus is the infrastructure, but it acts as the orchestrator


## Proposed Architecture

**Each application owns its own launch stack.** The application binary is the entry point. Circus becomes a reusable simulator that any application can start.

```
User
 └─► sim2sim/bin/main  OR  maximus/bin/maximus_main
       ├─ auto-generates: path_constants.yaml  (from own binary path)
       ├─ owns: framework_config.yaml          (its own volume needs)
       ├─ owns: booster.conf                   (its own process list)
       ├─ owns: entrypoint.sh                  (its own env setup)
       ├─ owns: scenes/                        (its own scenario definitions)
       ├─ owns: simulation_configs/            (its own game settings)
       └─► docker run <app-image>  (per robot)
             └─► supervisord (app's booster.conf)
                   ├─► circus (Qt GUI, inside container or on host)
                   ├─► booster-motion -config <app-specific>.lua
                   ├─► simbridge_node
                   └─► <app binary>
```

Anyone can build a new application that uses circus as a simulator by providing their own config set. Circus has no knowledge of who is using it.

---

## Config Classification

### Configs that stay in Circus (circus-internal, static)

These are **intrinsic to the simulator** and do not depend on which application is running:

| File | Reason |
|------|--------|
| `resources/config/fields/*.yaml` | Standard RoboCup field geometries — fixed by league rules |
| Core MuJoCo XML assets | Physics model definitions, not application-specific |

Circus should expose these via CLI arguments so applications can reference or override them.

### Configs that move to the Application

These depend on **which application is running** and should live with the application:

| File | Currently In | Moves To | Reason |
|------|-------------|----------|--------|
| `path_constants.yaml` | circus/resources/config/ | **generated at runtime** | Only the app knows where it lives |
| `framework_config.yaml` | circus/resources/config/ | app/configs/ | Docker volumes depend on which app is running |
| `booster.conf` | circus/dockerfiles/ | app/configs/ | Which processes run depends on the app |
| `entrypoint.sh` | circus/dockerfiles/ | app/configs/ | Env vars depend on the app (TensorRT, SPQR_CONFIG_ROOT, etc.) |
| `simulation_configs/*.yaml` | circus/resources/config/ | app/configs/simulation/ | Game settings are scenario-specific, not simulator-specific |
| `scenes/*.yaml` | circus/resources/scenes/ | app/configs/scenes/ | Robot count, position, team are entirely app-defined |
| `booster_motion/configs/*.lua` | simbridge/tools/ | app/configs/booster_motion/ | Which motion graph runs depends on the app (RL vs behavior) |

---

## Per-Application Config Structure

### sim2sim (Colosseum RL testing)

```
colosseum/src/sim2sim/
├── configs/
│   ├── framework_config.yaml       # volumes: simbridge + sim2sim binary
│   ├── booster.conf                # programs: booster-motion, simbridge, sim2sim (no maximus)
│   ├── entrypoint.sh               # no TensorRT, no SPQR_CONFIG_ROOT
│   ├── scenes/
│   │   └── training_1v1.yaml       # 1 robot, specific spawn position for training
│   ├── simulation/
│   │   └── training.yaml           # no game phases, unlimited time, auto-restart
│   └── booster_motion/
│       └── config_t1_rl.lua        # RL-specific motion graph (no behavior tree)
└── bin/
    └── main                        # Entry point — generates path_constants.yaml, launches stack
```

**`framework_config.yaml` for sim2sim:**
```yaml
image: spqr:booster
volumes:
  - "<simbridge>/tools/booster_motion:/app/booster_motion"
  - "<simbridge>/.pixi/envs/default:/app/bridge"
  - "<colosseum>/src/sim2sim/.pixi/envs/default:/app/colosseum"
  - "/dev/shm/circus_ipc:/dev/shm/circus_ipc"
```
No maximus volumes. No TensorRT.

**`booster.conf` for sim2sim:**
```ini
[program:booster-motion]
command=/app/booster_motion/booster-motion -mode sim -config ./configs/config_t1_rl.lua
priority=1

[program:delayed-starter]
command=/app/delayed_start.sh
priority=2

[program:simbridge]
command=/app/bridge/bin/simbridge_node
autostart=false

[program:colosseum]
command=/app/colosseum/bin/main
autostart=false
```

---

### maximus (behavior framework testing)

```
spqrbooster2026/
├── configs/
│   ├── framework_config.yaml       # volumes: simbridge + maximus + TensorRT
│   ├── booster.conf                # programs: booster-motion, simbridge, maximus (no colosseum)
│   ├── entrypoint.sh               # TensorRT symlink, SPQR_CONFIG_ROOT, behavior tree path
│   ├── scenes/
│   │   ├── 1v1.yaml
│   │   └── 5v5.yaml
│   ├── simulation/
│   │   └── default.yaml            # full game settings, 10-minute matches
│   └── booster_motion/
│       └── config_isaac.lua        # standard behavior graph
└── src/app/
    └── main.cpp                    # Entry point — generates path_constants.yaml, launches stack
```

---

## Implementation Steps

### Step 1 — Circus accepts external config paths via CLI

Circus needs to stop hardcoding its internal config paths. Instead, it should accept CLI arguments:

```bash
circus \
  --scene /path/to/app/configs/scenes/training_1v1.yaml \
  --simulation-config /path/to/app/configs/simulation/training.yaml \
  --framework-config /path/to/app/configs/framework_config.yaml \
  --path-constants /path/to/generated/path_constants.yaml
```

Changes required in circus C++:
- `Constants.h`: Replace hardcoded paths with runtime values set from CLI args
- `SceneParser`: Accept scene file path as constructor argument (already partially done)
- `RobotManager::startContainers()`: Accept framework config and path constants paths as arguments

Circus keeps its internal configs as **defaults** (for when it's run standalone), but always prefers externally provided paths.

### Step 2 — Each application generates `path_constants.yaml` at startup

The application binary knows its own location at runtime. From there it can derive all sibling repo paths:

```cpp
// In sim2sim main.cpp
std::filesystem::path self = std::filesystem::canonical("/proc/self/exe");
std::filesystem::path spqr_root = self.parent_path().parent_path().parent_path(); // up from bin/

// Write path_constants.yaml
YAML::Node paths;
paths["circus"]      = (spqr_root / "circus").string();
paths["simbridge"]   = (spqr_root / "simbridge").string();
paths["colosseum"]   = (spqr_root / "colosseum").string();
paths["maximus"]     = (spqr_root / "spqrbooster2026").string();

// Write to a temp location or pass directly to circus via CLI
std::string out_path = "/tmp/spqr_path_constants.yaml";
write_yaml(paths, out_path);
```

No more manually maintained `path_constants.yaml`. It is always auto-generated.

### Step 3 — Each application owns its Docker configs

Move `framework_config.yaml`, `booster.conf`, and `entrypoint.sh` into each application's repo under `configs/`. These files are already written — they just live in the wrong place.

The application passes its own `framework_config.yaml` to circus (via CLI, Step 1) so circus knows which volumes to mount when creating containers.

### Step 4 — Each application owns its scenes and simulation configs

Move scene files and simulation config files from circus's resources into the application. The application passes the scene file path to circus via CLI (Step 1).

Circus's `resources/config/simulation_configs/` and `resources/scenes/` directories are removed (or kept only as examples/defaults).

### Step 5 — Each application owns its booster_motion Lua configs

The RL-specific Lua configs (`config_t1_rl_isaac.lua`, `common_graph_define_t1_rl_isaac.lua`, etc.) move from `simbridge/tools/booster_motion/configs/` into the application repo. The Docker volume mount for booster_motion already allows injecting configs — the application just ships its own variants.

### Step 6 — Application binary becomes the launch entry point

The application's `main()` orchestrates startup:

```
1. Parse args (scene, robot count, etc.)
2. Generate path_constants.yaml → /tmp/spqr_path_constants_<pid>.yaml
3. Launch circus: circus --scene configs/scenes/training_1v1.yaml
                         --framework-config configs/framework_config.yaml
                         --path-constants /tmp/spqr_path_constants_<pid>.yaml
4. Wait for circus TCP server to be ready (port 5555)
5. Application's own main loop starts (RL training, behavior execution, etc.)
```

Circus handles the Docker container lifecycle (launching one container per robot in the scene). The containers use the application's `booster.conf` and `entrypoint.sh` via volume mount or baked into a shared base Docker image.

---

## Docker Image Strategy

Currently there is one image: `spqr:booster`. Two options going forward:

**Option A — Shared base image, app-specific configs via volumes**
Keep `spqr:booster` as a base with booster-motion, simbridge, and ROS2 pre-installed.
Each application mounts its own `booster.conf` and `entrypoint.sh` into the container:
```yaml
volumes:
  - "<app>/configs/booster.conf:/etc/supervisor/conf.d/booster.conf"
  - "<app>/configs/entrypoint.sh:/app/entrypoint.sh"
```
Pros: one image to maintain. Cons: entrypoint override is slightly awkward.

**Option B — Per-application Docker images**
`spqr:sim2sim` and `spqr:maximus` each extend the base image and bake in their own supervisord config.
Pros: fully self-contained. Cons: more images to build and maintain.

**Recommendation: Option A** initially, since the only difference between applications is which optional programs run in supervisord and which env vars are set. Volume-mounting the conf is simple and avoids image proliferation.

---

## What Circus Loses

After this change, circus's `resources/config/` directory contains only:
- `fields/*.yaml` (standard RoboCup field definitions)

Everything else is either deleted or moved to the application repos. Circus becomes a reusable simulator binary with no opinions about who runs it or how.

---

## Migration Checklist

- [ ] Add CLI args to circus for `--scene`, `--simulation-config`, `--framework-config`, `--path-constants`
- [ ] Move `framework_config.yaml` to sim2sim and maximus repos
- [ ] Move `booster.conf` and `entrypoint.sh` to sim2sim and maximus repos
- [ ] Move `simulation_configs/` to sim2sim and maximus repos
- [ ] Move `scenes/` to sim2sim and maximus repos
- [ ] Move RL-specific Lua configs to colosseum/sim2sim repo
- [ ] Add path_constants.yaml generation to sim2sim `main.cpp`
- [ ] Add path_constants.yaml generation to maximus `main.cpp`
- [ ] Add circus launch logic to sim2sim `main.cpp`
- [ ] Add circus launch logic to maximus `main.cpp`
- [ ] Remove `path_constants.yaml` from circus repo
- [ ] Verify circus still works standalone with its own defaults (for development/debugging)
