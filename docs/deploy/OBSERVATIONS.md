# Observation Contract

Training and deployment compute observations independently but must produce identical vectors. The observation specification defines this contract.

## The Pattern

```python
# Define once: tasks/velocity/mdp/observation_spec.py
class VelocityObservationSpec(ObservationSpec):
    observation_names = [
        "base_lin_vel",       # (3,)
        "base_ang_vel",       # (3,)
        "projected_gravity",  # (3,)
        "joint_pos_rel",      # (num_joints,)
        "joint_vel",          # (num_joints,)
        "last_action",        # (num_joints,)
        "velocity_commands",  # (3,)
    ]

# Training: extract from simulation state
def base_ang_vel(env, asset_cfg):
    return env.scene[asset_cfg.name].data.root_ang_vel_b

# Python deployment: compute from sensor data
def compute_observation(self):
    obs = torch.cat([
        self.robot.data.root_lin_vel_b,
        self.robot.data.root_ang_vel_b,
        self.robot.data.projected_gravity_b,
        self.robot.data.joint_pos[real2sim] - default_pos,
        self.robot.data.joint_vel[real2sim],
        self.last_action,
        self.vel_command.to_tensor(),
    ])
    return obs  # (82,) for 23-DOF

# C++ deployment: same computation, from rt/low_state
void build_observation(const RobotData& state, ...) {
    // Same order: lin_vel, ang_vel, proj_gravity,
    //             joint_pos_rel, joint_vel, last_action, vel_cmd
}
```

Both sides produce the same 82-dim vector. The ONNX model has the normalizer baked in, so raw values are expected.

## Validation

The `ObservationSpec` base class provides runtime validation:

```python
VELOCITY_OBS_SPEC.validate_observation(obs, num_joints=23)
# Raises ValueError if size doesn't match

components = VELOCITY_OBS_SPEC.split_observation(obs, num_joints=23)
# Returns dict: {"base_ang_vel": tensor(3,), "joint_pos_rel": tensor(23,), ...}

print(VELOCITY_OBS_SPEC.describe(num_joints=23))
# Observation Structure:
#   Total size: 82
#   [  0:  3] base_lin_vel        (size: 3)
#   [  3:  6] base_ang_vel        (size: 3)
#   ...
```

## Joint Order

The T1 23-DOF has identical joint ordering between hardware and MuJoCo simulation (no remapping needed). If this changes for a different robot, use `real2sim_joint_indexes` / `sim2real_joint_indexes` mappings.

Observations use **simulation order** (the order the policy was trained with).
Actions output in **simulation order**, then remapped to hardware order before sending.

## Adding New Observations

1. Add to the spec's `observation_names` and `get_component_size`
2. Add the training extraction function
3. Add the deployment computation (both Python and C++)
4. Retrain the policy (observation size changed)
