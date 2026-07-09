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
from mjlab.tasks.velocity.mdp.observations import (
  foot_contact,
  foot_contact_forces,
  foot_height,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from colosseum.mdp.amp_obs import AMP_BODY_JOINTS, amp_basic_obs_group
from colosseum.mdp.observations import base_height
from colosseum.mdp.symmetry import (
  MirrorableObservationTermCfg,
  mirror_ang_vel,
  mirror_base_lin_vel,
  mirror_gait_phase,
  mirror_projected_gravity,
  mirror_velocity_command,
)
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
    func=base_height,
    noise=Unoise(n_min=-0.02, n_max=0.02),
  ),
  "foot_height": ObservationTermCfg(
    func=foot_height,
    params={"sensor_name": "foot_height_scan"},
  ),
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
  # AMP discriminator state: joints-only [joint_pos_rel(21), joint_vel(21)] = 42-dim,
  # speed-agnostic (no base velocities) so a single slow clip doesn't fight the
  # commanded speed range. Head excluded (randomly perturbed in training, static in
  # the clip → a trivial separator). Must match AmpMotionDataset
  # (include_base_vel=False, joint_ids mirroring AMP_BODY_JOINTS).
  "amp": amp_basic_obs_group(joint_names=AMP_BODY_JOINTS),
}
