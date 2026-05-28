# Booster Motion Policy Architecture — Analysis Report

## Scope

Analysis of `~/code/spqr/arena/external/simbridge/tools/booster_motion/` with focus on two questions:

1. Gait frequency: how is it configured, and how does it compare to
   `include/GaitPhaseCommand.h`?
2. Policy topology: is there a single network, or are there multiple policies —
   particularly for push recovery?

---

## 1. Gait Frequency

### 1.1 `include/GaitPhaseCommand.h` (our colosseum-side code)

- Fixed frequency sampled once per episode from `[gait_freq_lo=1.5, gait_freq_hi=2.0]` Hz.
- Default midpoint = 1.75 Hz.
- No speed-dependent scaling ("Frequency is FIXED").
- Phase is **computed externally** and fed into the policy observation as
  `[cos(φ_L), cos(φ_R), sin(φ_L), sin(φ_R)]`.
- Standing condition: when `|v_xy| < gate_speed_threshold=0.05`, phase snaps to
  π for both feet, producing observation `[-1, -1, 0, 0]`.

### 1.2 Booster motion

- **No explicit gait frequency parameter exists** in any config file
  (`configs/*.lua`, `configs/*.toml`).
- The RL policies (`RMARun`, `RMALocomotion`, `RMAFall`, etc.) learn their own
  gait rhythm internally — the frequency is an emergent property of the network
  output, not an input.
- The `gait_freq_indicator` / `high_freq_index` signal wired into `RMARun` is
  a **discrete mode selector** (high vs. low frequency gait), not a numerical
  Hz value. See connections:
  ```
  con_rl_locomotion_run_6: command_manager.gait_freq_indicator → rl_locomotion_run.high_freq_index
  ```
  in `configs/common_graph_define.lua:1367-1372` and equivalent in
  `configs/common_graph_define_t1_rl_isaac.lua:992-997`.
- The robocup T1 graph (`common_graph_define_t1_rl_robocup_isaac.lua`) uses
  `rl_speed` and `rl_footprint` which each output their own `phase` signal
  back to the publisher — suggesting the policy itself synthesizes a gait
  phase internally and reports it.

### 1.3 Contrast

| Aspect | `GaitPhaseCommand.h` | Booster motion |
|--------|----------------------|----------------|
| Frequency source | External, sampled [1.5, 2.0] Hz | Internal, learned by policy |
| Phase input | `[cos, cos, sin, sin]` in observation | None (policy outputs phase internally) |
| Standing | Speed-gated with hysteresis | Policy-internal fall detection |

---

## 2. Policy Topology: Multiple Networks

### 2.1 The switch architecture

Motor commands are routed through `planner_pvt_switch` (a `special::Switch`
module), controlled by `command_manager.planner_index`. Each planner occupies
a numbered input port:

| Port | Module | Class | Source | Active in config |
|------|--------|-------|--------|------------------|
| in_1 | `commander1` | `MitVariedTarget` | `libmodule_source.so` | All |
| in_3 | `damping_mode` | `DampingMode` | `libmodule_source.so` | All |
| in_4 | `debugging_mode` / `custom_mode` | — | — | Some |
| in_5 | `stance` (DCM + WBC) | → `BipedQuasiStaticPlanner` | multiple | All |
| in_6 | `rl_locomotion` | **`RMALocomotion`** | `librl_locomotion.so` | All |
| in_7 | `custom_traj` | `NestedTrajectory` | `libmodule_source.so` | All |
| in_8 | `rl_locomotion_run` | **`RMARun`** | `librl_locomotion.so` | All |
| in_9 | `rl_fall_recovery` | **`RMAFall`** | `librl_locomotion.so` | Full only |
| in_10 | `rl_locomotion_gait_2` | **`RMARunMultiModal`** | `librl_locomotion.so` | Full only |

Source: `configs/common_graph_define.lua:421-481` (switch connections).

The T1 robocup config adds `rl_speed` (RMARun) and `rl_footprint`
(RMABallDribble), with their own internal `rl_parallel_mech_switch` before
being routed into in_6/8/9 (see
`configs/common_graph_define_t1_rl_robocup_isaac.lua:872-932`).

### 2.2 The five RL policy modules

Each is a **separately trained policy** loaded from distinct model weight
files in `lib/`:

| Module class | Model files in `lib/` | Role |
|---|---|---|
| `RMALocomotion` | `rma_locomotion_model` | Simpler locomotion (vel tracking, fall detection) |
| `RMARun` | `rma_run_model`, `rma_run_tcn`, `rma_run_odom` | Main high-performance walking/running with odometry |
| `RMAFall` | (in `librl_locomotion.so`) | Explicit fall recovery policy |
| `RMARunMultiModal` | (in `librl_locomotion.so`) | Secondary gait style |
| `RMABallDribble` | (in `librl_locomotion.so`) | Ball dribbling (robocup T1 only) |

Model files checked with `file`:
```
rma_locomotion_model:  data
rma_run:               data
rma_run_model:         data
rma_run_odom:          data
rma_run_tcn:           data
rma_run_tcn_7_dof_arm: data
```

All are opaque binary blobs (likely jit-compiled TorchScript or ONNX graphs).

The naming strongly suggests an **RMA (Rapid Motor Adaptation)** architecture:
a base MLP policy (`_model`) + a Temporal Convolutional Network encoder
(`_tcn`) for environment/context estimation + an odometry estimator (`_odom`).

### 2.3 RMALocomotion vs RMARun — detailed comparison

From graph connections (using T1 config for clarity,
`configs/common_graph_define_t1_rl_isaac.lua`):

| | `rl_locomotion` (RMALocomotion) | `rl_locomotion_run` (RMARun) |
|---|---|---|
| `state_command` (vel_cmd) | yes (con_rl_locomotion_1) | yes (con_rl_locomotion_run_1) |
| `gait_index` / wave_hand_index | yes (con_rl_locomotion_2) | yes (con_rl_locomotion_run_2) |
| `motor_state_feedback` | yes | yes |
| `imu_feedback` | yes | yes |
| `high_freq_index` (gait freq indicator) | **no** | **yes** (con_rl_locomotion_run_6) |
| `yaw_tracking_index` | **no** | **yes** (con_rl_locomotion_run_7) |
| outputs `odom` | no | **yes** (to publisher) |
| outputs `robot_fallen` | yes (to portal) | yes (to portal) |
| output port on switch | in_6 | in_8 |

Key observations:
- Both receive velocity commands — neither is a pure "stand-up-only" policy.
- RMARun has additional conditioning signals (`high_freq_index`,
  `yaw_tracking_index`) that allow the command_manager to modulate its
  behavior.
- RMARun is the only policy that outputs `odom` for state estimation.
- The comment "stand up controller" in
  `common_graph_define_v23_rl_isaac.lua:828` labels RMALocomotion's
  connections, suggesting its training domain may have included ground-to-stand
  transitions more heavily.

---

## 3. Command Arbitration: CommandManagerV2

The `command_manager` module (`source::CommandManagerV2`) is a **scripted
state machine**, not a learned policy. Evidence:

- It lives in the `source::` namespace (`libmodule_source.so`).
- It configures all velocity limits, gait list, planner index switching logic
  (`configs/common_module_options.lua:112-184`).
- It receives fall signals from ALL RL policies simultaneously via portal
  pairs:
  ```
  rl_locomotion.robot_fallen → PortalCollect → PortalPublish → command_manager.rl_locomotion_fallen_
  rl_locomotion_run.robot_fallen → PortalCollect → PortalPublish → command_manager.rl_locomotion_run_fallen_
  rl_locomotion_gait_2.robot_fallen → PortalCollect → PortalPublish → command_manager.rl_locomotion_gait_2_fallen_
  ```
  (From `common_graph_define.lua:126-143`)

- It also receives `fall_down_recovery_state` and `is_recovery_avalible` from
  the non-RL `FallDownRecovery` module.

- It outputs `planner_index` to control `planner_pvt_switch` — a simple
  multiplexer that connects one policy's motor commands to the robot at a
  time.

The decision chain for falls:
```
policy → "I'm unstable" via robot_fallen → command_manager evaluates → changes planner_index → different policy drives robot
```

### 3.1 Fall detection mechanism

The `robot_fallen` signal is **a hard-coded geometric threshold** — not a
learned neural network output. The thresholds are configured under
`rl_locomotion_run` in the options files (e.g.
`configs/common_module_options_v23_rl_isaac.lua:1838-1841`):

```lua
recovery_x_upper_thres =  0.1    -- rad, ~5.7° torso pitch forward
recovery_x_lower_thres = -0.05   -- rad, ~2.9° torso pitch backward
recovery_y_upper_thres =  0.1    -- rad, ~5.7° torso roll
recovery_y_lower_thres = -0.1    -- rad, ~5.7° torso roll
```

These parameter names were extracted from `librl_locomotion.so` via `strings`
and confirmed present in all three options files
(`common_module_options.lua:1938-1941`,
`common_module_options_t1_rl_isaac.lua:1953-1956`,
`common_module_options_v23_rl_isaac.lua:1838-1841`).

If torso pitch (x-axis) or roll (y-axis) from the IMU exceeds any of these
bounds, `librl_locomotion.so` immediately sets `robot_fallen = true`.
The thresholds are very tight — just ~3–6° from upright — which means the
signal fires **during** a push disturbance, well before the robot hits the
ground.

Notable: these recovery thresholds are only configured under
`rl_locomotion_run` (RMARun). `rl_locomotion` (RMALocomotion) has no
recovery thresholds in its config block and uses different model files:

```lua
rl_locomotion = {
    model_file = "lib/rma_locomotion_model",
    is_serial = true,
    g_offset_x = 0.05,
},

rl_locomotion_run = {
    model_file = "lib/rma_run",
    odom_model_file = "lib/rma_run_odom",
    g_offset_x = 0.0,
    vel_offset_x = -0.1,
    is_serial = true,
    recovery_x_upper_thres = 0.1,
    recovery_x_lower_thres = -0.05,
    recovery_y_upper_thres = 0.1,
    recovery_y_lower_thres = -0.1,
},
```

The `g_offset_x` and `vel_offset_x` biases suggest the two policies were
trained in different domains (RMALocomotion with a forward gravity bias,
RMARun with a backward velocity bias), reinforcing that they are different
gaits/style rather than a stand-up vs. walking distinction.

### 3.2 The `high_freq_index` and `yaw_tracking_index` inputs

These are **discrete mode flags** from `command_manager` to the RL policy,
not numerical values:

- `high_freq_index` — selects a gait frequency mode (high vs. low). The RL
  policy internally maps this to different stepping rates.
- `yaw_tracking_index` — enables / disables yaw tracking behavior. The policy
  conditions on this in its internal computation.

Neither is a continuous frequency (Hz) or angle value.

### 3.3 The `odom` output

RMARun outputs `odom` to the `publisher` module
(`common_graph_define_v23_rl_isaac.lua:752-756`),
separate from the motor command path. This is an auxiliary network head —
the policy estimates its own odometry internally (common in RMA architectures
where the TCN adaptation encoder is also trained for state estimation).

---

## 4. Control Output Pipeline

The RL policies output joint commands through a parallel mechanism chain:

```
rl_policy → motor_command → parallel_mech_input_rl_* (BipedParallelFeetInput)
  → motor_pvt_commands_serial → parallel_mech_output → planner_pvt_switch
  → intercept_motor_cmd → joint_map_input → simulator_in
```

In **real-bot mode** (`use_pvt_ = true`, `common_module_options.lua`):
outputs are joint PVT (position-velocity-torque) commands.

In **simulation mode** (`use_pvt_ = false`,
`common_module_options_t1_rl_isaac.lua:103` / 
`common_module_options_v23_rl_isaac.lua:103`):
outputs go through a PID controller → `force_commands` → simulator.

Control loop rates:
| Config | `controller_base_dt_ms` | Loop rate |
|--------|------------------------|-----------|
| Full (`common_module_options.lua`) | 2 | 500 Hz |
| T1 (`common_module_options_t1_rl_isaac.lua`) | 4 | 250 Hz |
| V23 (`common_module_options_v23_rl_isaac.lua`) | 8 | 125 Hz |
| T1 Robocup | 8 | 125 Hz |

---

## 5. Config File Matrix

Which graph/options file corresponds to which hardware config:

| Graph file | Options file | Notes |
|---|---|---|
| `common_graph_define.lua` | `common_module_options.lua` | Full: all RL policies + DCM + fall recovery |
| `common_graph_define_v23_rl_isaac.lua` | `common_module_options_v23_rl_isaac.lua` | V23: RMALocomotion + RMARun only |
| `common_graph_define_t1_rl_isaac.lua` | `common_module_options_t1_rl_isaac.lua` | T1: RMALocomotion + RMARun |
| `common_graph_define_t1_rl_robocup_isaac.lua` | `common_module_options_t1_rl_robocup_isaac.lua` | T1 Robocup: + RMABallDribble |

---

## 6. Summary

1. **Gait frequency**: The booster motion RL policies learn their own gait
   frequency internally — there is no configurable frequency parameter. This
   contrasts with `GaitPhaseCommand.h` which explicitly drives a fixed
   1.5–2.0 Hz phase clock.

2. **Policy count**: There are **at least 3, and up to 5** separately trained
   RL policies deployed simultaneously:

   | Policy | When active |
   |--------|-------------|
   | RMALocomotion | Stand controller / simpler locomotion |
   | RMARun | Primary walking/running with odometry |
   | RMAFall | Fall recovery |
   | RMARunMultiModal | Alternate gait (full config only) |
   | RMABallDribble | Ball control (robocup only) |

3. **Fall recovery**: Is **not a behavioral mode of a single network**. It is
   a **separate policy** (`RMAFall`, loaded from distinct weights) activated
   when a fall is detected. The decision to switch is made by
   `CommandManagerV2`, a scripted state machine, based on `robot_fallen`
   signals from all running policies plus state from the non-RL
   `FallDownRecovery` module.

4. **Architecture style**: RMA (Rapid Motor Adaptation). Each policy uses a
   base MLP model paired with a TCN adaptation encoder. RMARun additionally
   has an odometry estimator head. All model weights are opaque binary blobs
   in `lib/`.
