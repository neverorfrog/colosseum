"""Observations for the booster-mimic Waist+legs task.

Identical layout to the 12-DOF task, but the joint blocks span the 13 actuated
joints (Waist + 12 legs) instead of 12.

Actor (50): proj_gravity(3) ang_vel(3) command(3) gait[cos,sin](2)
            dof_pos-default(13) 0.1*dof_vel(13) actions(13).
Critic (64) = actor terms + booster's full 14-dim privileged vector, in order:
            base_mass_scaled(4) base_lin_vel(3) base_height(1)
            push_force(3) push_torque(3).
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
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from colosseum.mdp.observations import (
  base_external_force,
  base_external_torque,
  base_height,
  base_mass_com_offset,
  gait_clock,
)
from colosseum.mdp.symmetry import (
  MirrorableObservationTermCfg,
  mirror_ang_vel,
  mirror_base_lin_vel,
  mirror_base_mass_com,
  mirror_gait_clock,
  mirror_projected_gravity,
  mirror_velocity_command,
)
from colosseum.robots.t1_23dof.mdp.symmetry import mirror_legs_waist

# Waist + 12 legs, in MuJoCo model index order (Waist first). Restricts the
# joint_pos/joint_vel observations to the actuated set; without this the 23-DOF
# model would emit all 23 joints.
_ACTUATED = SceneEntityCfg(
  "robot", joint_names=("Waist", ".*Hip.*", ".*Knee.*", ".*Ankle.*")
)

# Noise half-widths from booster T1.yaml (gravity 0.01, ang_vel 0.1, dof_pos 0.01,
# dof_vel 0.1, lin_vel 0.05, height 0.02).
actor_terms = {
  "projected_gravity": MirrorableObservationTermCfg(
    func=projected_gravity,
    noise=Unoise(n_min=-0.01, n_max=0.01),
    mirror_fn=mirror_projected_gravity,
  ),
  "base_ang_vel": MirrorableObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_ang_vel"},
    noise=Unoise(n_min=-0.1, n_max=0.1),
    mirror_fn=mirror_ang_vel,
  ),
  "command": MirrorableObservationTermCfg(
    func=generated_commands,
    params={"command_name": "twist"},
    mirror_fn=mirror_velocity_command,
  ),
  "gait_phase": MirrorableObservationTermCfg(
    func=gait_clock,
    params={"command_name": "gait_phase", "twist_command_name": "twist"},
    mirror_fn=mirror_gait_clock,
  ),
  "joint_pos": MirrorableObservationTermCfg(
    func=joint_pos_rel,
    params={"asset_cfg": _ACTUATED},
    noise=Unoise(n_min=-0.01, n_max=0.01),
    mirror_fn=mirror_legs_waist,
  ),
  "joint_vel": MirrorableObservationTermCfg(
    func=joint_vel_rel,
    params={"asset_cfg": _ACTUATED},
    noise=Unoise(n_min=-0.1, n_max=0.1),
    scale=0.1,  # booster dof_vel normalization
    mirror_fn=mirror_legs_waist,
  ),
  "actions": MirrorableObservationTermCfg(
    func=last_action,
    mirror_fn=mirror_legs_waist,
  ),
}

# Critic is privileged: actor proprio + booster's 14-dim privileged vector,
# in booster's exact order (base_mass_scaled, base_lin_vel, height, push_*).
_BASE = SceneEntityCfg("robot", body_names="Trunk")
critic_terms = {
  **actor_terms,
  "base_mass_scaled": MirrorableObservationTermCfg(
    func=base_mass_com_offset,
    params={"asset_cfg": _BASE},
    mirror_fn=mirror_base_mass_com,
  ),
  "base_lin_vel": MirrorableObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_lin_vel"},
    noise=Unoise(n_min=-0.05, n_max=0.05),
    mirror_fn=mirror_base_lin_vel,
  ),
  "base_height": ObservationTermCfg(
    func=base_height,
    noise=Unoise(n_min=-0.02, n_max=0.02),
  ),
  # booster push_force normalization 0.1, push_torque 0.5.
  "push_force": MirrorableObservationTermCfg(
    func=base_external_force,
    params={"asset_cfg": _BASE},
    scale=0.1,
    mirror_fn=mirror_base_lin_vel,
  ),
  "push_torque": MirrorableObservationTermCfg(
    func=base_external_torque,
    params={"asset_cfg": _BASE},
    scale=0.5,
    mirror_fn=mirror_ang_vel,
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
