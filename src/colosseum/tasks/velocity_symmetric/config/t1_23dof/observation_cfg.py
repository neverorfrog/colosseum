"""Observation config for T1 velocity task with symmetry mirroring.

Uses MirrorableObservationTermCfg so the PPO algorithm can apply
left-right mirror augmentation and symmetry loss during training.
"""

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
  foot_air_time,
  foot_contact,
  foot_contact_forces,
  foot_height,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from colosseum.mdp.observations import base_height
from colosseum.mdp.symmetry import (
  MirrorableObservationTermCfg,
  mirror_ang_vel,
  mirror_base_lin_vel,
  mirror_gait_phase,
  mirror_projected_gravity,
  mirror_velocity_command,
)
from colosseum.robots.t1_23dof.mdp.symmetry import mirror_joints

actor_terms = {
  "base_ang_vel": MirrorableObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_ang_vel"},
    noise=Unoise(n_min=-0.2, n_max=0.2),
    mirror_fn=mirror_ang_vel,
  ),
  "projected_gravity": MirrorableObservationTermCfg(
    func=projected_gravity,
    noise=Unoise(n_min=-0.05, n_max=0.05),
    mirror_fn=mirror_projected_gravity,
  ),
  "joint_pos": MirrorableObservationTermCfg(
    func=joint_pos_rel,
    noise=Unoise(n_min=-0.01, n_max=0.01),
    mirror_fn=mirror_joints,
  ),
  "joint_vel": MirrorableObservationTermCfg(
    func=joint_vel_rel,
    noise=Unoise(n_min=-0.05, n_max=0.05),
    mirror_fn=mirror_joints,
  ),
  "actions": MirrorableObservationTermCfg(
    func=last_action,
    mirror_fn=mirror_joints,
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

critic_terms = {
  **actor_terms,
  "base_lin_vel": MirrorableObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_lin_vel"},
    noise=Unoise(n_min=-0.5, n_max=0.5),
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
}
