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
from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, FOOT_GEOM_NAMES
from colosseum.robots.t1_23dof.mdp.symmetry import mirror_joints
from colosseum.tasks.velocity.mdp.observations import (
  actuator_kd_scale_obs,
  actuator_kp_scale_obs,
  base_com_obs,
  effort_limit_scale_obs,
  foot_friction_obs,
  joint_damping_scale_obs,
  link_mass_scale_obs,
)

# ---------------------------------------------------------------------------
# Actor / critic terms
# ---------------------------------------------------------------------------

actor_terms = {
  "base_ang_vel": MirrorableObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_ang_vel"},
    noise=Unoise(n_min=-0.1, n_max=0.1),
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
    noise=Unoise(n_min=-0.1, n_max=0.1),
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

# ---------------------------------------------------------------------------
# Privileged env_params group (9D)
# ---------------------------------------------------------------------------

_foot_cfg = SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES)
_base_cfg = SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,))
_actuator_cfg = SceneEntityCfg("robot")
_all_bodies_cfg = SceneEntityCfg("robot", body_names=(".*",))
_all_joints_cfg = SceneEntityCfg("robot", joint_names=(".*",))

env_params_terms = {
  "foot_friction": ObservationTermCfg(
    func=foot_friction_obs,
    params={"asset_cfg": _foot_cfg},
  ),
  "base_com": ObservationTermCfg(
    func=base_com_obs,
    params={"asset_cfg": _base_cfg},
  ),
  "kp_scale": ObservationTermCfg(
    func=actuator_kp_scale_obs,
    params={"asset_cfg": _actuator_cfg},
  ),
  "kd_scale": ObservationTermCfg(
    func=actuator_kd_scale_obs,
    params={"asset_cfg": _actuator_cfg},
  ),
  "link_mass_scale": ObservationTermCfg(
    func=link_mass_scale_obs,
    params={"asset_cfg": _all_bodies_cfg},
  ),
  "joint_damping_scale": ObservationTermCfg(
    func=joint_damping_scale_obs,
    params={"asset_cfg": _all_joints_cfg},
  ),
  "effort_limit_scale": ObservationTermCfg(
    func=effort_limit_scale_obs,
    params={"asset_cfg": _actuator_cfg},
  ),
}


odom_terms = {
  "base_lin_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_lin_vel"},
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
  "env_params": ObservationGroupCfg(
    terms=env_params_terms,
    concatenate_terms=True,
    enable_corruption=False,
  ),
  "odom": ObservationGroupCfg(
    terms=odom_terms,
    concatenate_terms=True,
    enable_corruption=False,
  ),
}
