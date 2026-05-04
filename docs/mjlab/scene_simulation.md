# Scene & Simulation

The scene and simulation layers are documented in full in the
**[mjlab documentation](https://mujocolab.github.io/mjlab/main/index.html)**.

Colosseum does not extend these layers — it uses them as-is. The pages below
cover everything you need:

| Topic | mjlab docs |
|---|---|
| Entity (robot, terrain, sensors) | [Entity](https://mujocolab.github.io/mjlab/main/source/entity/index.html) · [Entity Data](https://mujocolab.github.io/mjlab/main/source/entity/entity_data.html) |
| Actuators | [Actuators](https://mujocolab.github.io/mjlab/main/source/actuators.html) |
| Sensors (contact, RayCast, RGB-D) | [Sensors](https://mujocolab.github.io/mjlab/main/source/sensors/index.html) |
| Scene configuration | [Scene](https://mujocolab.github.io/mjlab/main/source/scene.html) |
| Terrain | [Terrain](https://mujocolab.github.io/mjlab/main/source/terrain.html) |
| GPU-accelerated physics (MuJoCo Warp) | [Architecture Overview](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html) |
| Heterogeneous worlds (per-world mesh) | [Heterogeneous Worlds](https://mujocolab.github.io/mjlab/main/source/entity/per_world_mesh.html) |

---

## Quick orientation

The three-layer structure is summarised in the [mjlab Overview](overview.md).
In short:

- **Simulation layer** — MuJoCo Warp runs 4096+ physics environments in
  parallel on GPU. Colosseum sets `timestep=0.005` (200 Hz) and uses
  `decimation=4`, so the policy runs at 50 Hz.
- **Scene layer** — `SceneCfg` declares entities, terrain, and sensors by
  name. At compile time MuJoCo assigns global indices; `SceneEntityCfg`
  resolves string patterns (e.g. `body_names=".*_knee"`) to those indices
  once at init.
- **Task layer** — Managers (`RewardManager`, `ObservationManager`, …)
  use pre-resolved indices for fast per-step tensor operations. See
  [Task Layer](task_layer.md) for the colosseum view of this layer, or the
  [mjlab manager docs](https://mujocolab.github.io/mjlab/main/source/observations.html)
  for the complete API.

---

## Colosseum-specific scene setup

The T1 scene used by all humanoid tasks is assembled in
`src/colosseum/robots/t1_23dof/constants.py`. The key call is
`get_robot_cfg()`, which returns an `EntityCfg` with:

- **XML**: `robots/booster_t1/T1_23dof.xml`
- **Actuators**: PDActuator specs from `t1_actuators.py`
- **Sensors**: selectable contact and height-scan sensors from `t1_sensors.py`
- **Articulation info**: `CollisionCfg` and armature values

The dribbling scene additionally includes a soccer ball entity and up to
three obstacle entities (see `tasks/dribbling/config/t1_23dof/t1_dribbling_cfg.py`).
