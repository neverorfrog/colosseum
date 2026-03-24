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
  foot_air_time,
  foot_contact,
  foot_contact_forces,
  foot_height,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from colosseum.robots.t1_23dof.constants import FOOT_SITE_NAMES
from colosseum.tasks.dribbling.mdp.observations import ball_position

actor_terms = {
  "base_ang_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_ang_vel"},
    noise=Unoise(n_min=-0.2, n_max=0.2),
  ),
  "projected_gravity": ObservationTermCfg(
    func=projected_gravity,
    noise=Unoise(n_min=-0.05, n_max=0.05),
  ),
  "joint_pos": ObservationTermCfg(
    func=joint_pos_rel,
    noise=Unoise(n_min=-0.01, n_max=0.01),
  ),
  "joint_vel": ObservationTermCfg(
    func=joint_vel_rel,
    noise=Unoise(n_min=-1.5, n_max=1.5),
  ),
  "actions": ObservationTermCfg(func=last_action),
  "command": ObservationTermCfg(
    func=generated_commands,
    params={"command_name": "twist"},
  ),
  "ball_pos": ObservationTermCfg(
    func=ball_position,
    noise=Unoise(n_min=-0.01, n_max=0.01),
  ),
}

critic_terms = {
  **actor_terms,
  "base_lin_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_lin_vel"},
    noise=Unoise(n_min=-0.5, n_max=0.5),
  ),
  "foot_height": ObservationTermCfg(
    func=foot_height,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES)},
  ),
  "foot_air_time": ObservationTermCfg(
    func=foot_air_time,
    params={"sensor_name": "feet_ground_contact"},
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
