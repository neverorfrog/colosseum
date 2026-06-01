from mjlab.envs.mdp.observations import builtin_sensor
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, FOOT_GEOM_NAMES
from colosseum.tasks.velocity.config.t1_23dof.observation_cfg import (
  actor_terms,
  critic_terms,
)
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
