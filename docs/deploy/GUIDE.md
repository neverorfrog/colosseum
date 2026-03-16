# Deployment Guide

## Pipeline Overview

```
Train (mjlab, GPU)
  → Export ONNX (pixi run export-onnx)
  → Test Python sim2sim (MuJoCo backend, fast iteration)
  → Test C++ sim2sim (Circus + booster-motion, realistic)
  → Deploy to real robot (same C++ code, swap hardware for sim)
```

## Step 1: Train

```bash
pixi run -e train python -m colosseum.scripts.train task:t1-velocity-flat
```

Produces checkpoints in `logs/`.

## Step 2: Export ONNX

```bash
pixi run -e train export-onnx task:t1-velocity-flat
```

Output: `models/t1-velocity-flat_ppo_latest.onnx`

The exported model includes the observation normalizer baked in:
- Input: `"obs"` shape `(1, 82)` — raw observation vector
- Output: `"actions"` shape `(1, 23)` — normalized actions in [-1, 1]

## Step 3: Test with Python MuJoCo Backend

Quick local validation. No external dependencies needed.

```bash
pixi run python -m colosseum.scripts.deploy --task t1-velocity-flat --mujoco
```

Verify:
- Robot stands and balances
- Joystick commands produce expected motion
- No oscillations or divergence

## Step 4: Test with C++ Sim2Sim

Realistic pipeline using Circus + SimBridge + booster-motion.

```bash
# Terminal 1: Start Circus simulator
cd external/circus && pixi run circus

# Terminal 2: Start SimBridge + booster-motion (Docker)
cd external/simbridge && ./run.sh

# Terminal 3: Run policy
cd src/sim2sim && ./build/sim2sim --model ../../models/t1-velocity-flat_ppo_latest.onnx
```

This exercises the full DDS communication path. The policy reads `rt/low_state` and writes `rt/joint_ctrl`, exactly as it would on the real robot.

## Step 5: Deploy to Real Robot

Same sim2sim executable, but booster-motion reads from real hardware instead of SimBridge:

```bash
# On the robot (or networked computer)
./sim2sim --model models/t1-velocity-flat_ppo_latest.onnx
```

Start conservatively:
1. Begin with zero velocity commands
2. Gradually increase speed
3. Monitor for instability

## Configuration

All deployment parameters must match training exactly:

| Parameter | Where to find | Typical value |
|-----------|--------------|---------------|
| `action_scale` | `robots/t1_23dof/constants.py` | 0.25 |
| `default_joint_pos` | `robots/t1_23dof/deploy.py` | per-joint (rad) |
| `joint_stiffness` (kp) | `robots/t1_23dof/deploy.py` | per-joint |
| `joint_damping` (kd) | `robots/t1_23dof/deploy.py` | per-joint |
| `policy_dt` | task config | 0.02s (50Hz) |
| `observation_order` | `tasks/velocity/mdp/observation_spec.py` | see below |

## Observation Vector (Velocity Task, 82 dims)

Concatenated in this exact order:

| Component | Size | Source |
|-----------|------|--------|
| `base_lin_vel` | 3 | Body-frame linear velocity |
| `base_ang_vel` | 3 | `imu_state.gyro` |
| `projected_gravity` | 3 | Computed from `imu_state.rpy` → quat → rotate [0,0,-1] |
| `joint_pos_rel` | 23 | `motor_state[i].q - default_pos[i]` |
| `joint_vel` | 23 | `motor_state[i].dq` |
| `last_action` | 23 | Previous normalized action output |
| `velocity_commands` | 3 | `[vx, vy, vyaw]` scaled by max velocities |

The observation normalizer is baked into the ONNX model, so raw values go in.

## Troubleshooting

**Robot oscillates or is unstable:**
- Kp/Kd mismatch between training and deployment. Check values match exactly.
- Wrong `action_scale`.

**Policy produces nonsensical actions:**
- Observation order or content doesn't match training. Verify each component.
- Joint order mismatch. Verify motor_state_serial ordering matches training joint order.

**Robot doesn't respond (C++ sim2sim):**
- Check booster-motion is running and publishing `rt/low_state`.
- Check FastDDS profile restricts to loopback (`127.0.0.1`) when running locally.
- Verify robot is in `kCustom` mode.
