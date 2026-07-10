from dataclasses import replace

from mjlab.envs.mdp.observations import (
  builtin_sensor,
  generated_commands,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
  projected_gravity,
)
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.observations import (
  foot_contact,
  foot_contact_forces,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from colosseum.mdp.observations import (
  base_external_force,
  base_external_torque,
  base_mass_com_offset,
  terrain_clearance,
)
from colosseum.mdp.symmetry import (
  MirrorableObservationTermCfg,
  mirror_ang_vel,
  mirror_base_lin_vel,
  mirror_gait_phase,
  mirror_projected_gravity,
  mirror_velocity_command,
)
from colosseum.robots.t1.constants import BASE_BODY_NAME
from colosseum.robots.t1.mdp.symmetry import mirror_actions, mirror_joints

# ---------------------------------------------------------------------------
# Actor / critic terms
# ---------------------------------------------------------------------------

# Persistent per-episode observation latency on the sensor-derived actor terms,
# matching the real robot's FastDDS transport delay (~2-3 control steps). The high
# update_period holds the sampled lag for ~one episode (30s/20ms ~= 1500 steps)
# so the policy learns phase margin against a STEADY delay, not per-step jitter.
# Applied to actor only; the critic stays clean/privileged (see critic_terms).

actor_terms = {
  "base_ang_vel": MirrorableObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_ang_vel"},
    noise=Unoise(n_min=-0.1, n_max=0.1),
    mirror_fn=mirror_ang_vel,
    delay_min_lag=0,
    delay_max_lag=3,
    delay_update_period=1500,
  ),
  "projected_gravity": MirrorableObservationTermCfg(
    func=projected_gravity,
    noise=Unoise(n_min=-0.05, n_max=0.05),
    mirror_fn=mirror_projected_gravity,
    delay_min_lag=0,
    delay_max_lag=3,
    delay_update_period=1500,
  ),
  "joint_pos": MirrorableObservationTermCfg(
    func=joint_pos_rel,
    noise=Unoise(n_min=-0.01, n_max=0.01),
    mirror_fn=mirror_joints,
    delay_min_lag=0,
    delay_max_lag=3,
    delay_update_period=1500,
  ),
  "joint_vel": MirrorableObservationTermCfg(
    func=joint_vel_rel,
    noise=Unoise(n_min=-0.1, n_max=0.1),
    mirror_fn=mirror_joints,
    delay_min_lag=0,
    delay_max_lag=3,
    delay_update_period=1500,
  ),
  "actions": MirrorableObservationTermCfg(
    func=last_action,
    mirror_fn=mirror_actions,
  ),
  "command": MirrorableObservationTermCfg(
    func=generated_commands,
    params={"command_name": "twist"},
    mirror_fn=mirror_velocity_command,
  ),
  "gait_phase": MirrorableObservationTermCfg(
    func=generated_commands,
    params={"command_name": "gait_phase"},
    mirror_fn=mirror_gait_phase,
  ),
}

# Critic is privileged: same proprio terms but WITHOUT the actor's obs delay
# (replace() makes fresh copies, so mutating delay here can't alias the actor).
critic_terms = {
  **{
    k: replace(v, delay_min_lag=0, delay_max_lag=0, delay_update_period=0)
    for k, v in actor_terms.items()
  },
  "base_lin_vel": MirrorableObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_lin_vel"},
    noise=Unoise(n_min=-0.1, n_max=0.1),
    mirror_fn=mirror_base_lin_vel,
  ),
  "base_height": ObservationTermCfg(
    func=terrain_clearance,
    params={"sensor_name": "base_height_scan"},
    noise=Unoise(n_min=-0.02, n_max=0.02),
  ),
  "foot_height": ObservationTermCfg(
    func=terrain_clearance,
    params={"sensor_name": "foot_height_scan"},
  ),
  # Privileged physics — mirror the DR events; critic-only.
  "base_mass_com_offset": ObservationTermCfg(
    func=base_mass_com_offset,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME)},
  ),  # [N,4]
  "base_external_force": ObservationTermCfg(
    func=base_external_force,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME)},
  ),  # [N,3]
  "base_external_torque": ObservationTermCfg(
    func=base_external_torque,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME)},
  ),  # [N,3]
  "foot_contact": ObservationTermCfg(
    func=foot_contact,
    params={"sensor_name": "feet_ground_contact"},
  ),
  "foot_contact_forces": ObservationTermCfg(
    func=foot_contact_forces,
    params={"sensor_name": "feet_ground_contact"},
  ),
}

observations = {
  "actor": ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    enable_corruption=True,
  ),
  "critic": ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  ),
}
