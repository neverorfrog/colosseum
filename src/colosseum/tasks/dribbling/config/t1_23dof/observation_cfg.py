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

from colosseum.assets.ball.ball_spec import BALL_FRICTION, BALL_MASS
from colosseum.tasks.dribbling.mdp.observations import (
  ball_friction,
  ball_mass,
  ball_position,
  ball_vel_command_body,
  ball_velocity,
  ball_velocity_xy,
  base_height,
  foot_ball_contact_force,
)

# ---------------------------------------------------------------------------
# Actor: proprioceptive only.
# Encoder latents are concatenated by RmaPPO, not via the obs manager.
# ---------------------------------------------------------------------------

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
  "gait_phase": ObservationTermCfg(
    func=generated_commands,
    params={"command_name": "gait_phase"},
  ),
  "command": ObservationTermCfg(
    func=ball_vel_command_body,
    params={"command_name": "ball_vel"},
  ),
}

# ---------------------------------------------------------------------------
# Privileged ball: encoder input, 4D = ball_pos_xy(2) + ball_vel_xy(2).
# Also included wholesale in the critic so it sees the same values.
# ---------------------------------------------------------------------------

privileged_ball_terms = {
  "ball_pos": ObservationTermCfg(func=ball_position),  # (N, 2)
  "ball_vel_xy": ObservationTermCfg(func=ball_velocity_xy),  # (N, 2)
}

# ---------------------------------------------------------------------------
# Critic: actor + privileged ball + remaining GT terms + foot extras.
# Asymmetric actor-critic: critic sees everything, actor sees only proprio.
# ---------------------------------------------------------------------------

critic_terms = {
  **actor_terms,
  **privileged_ball_terms,
  "base_height": ObservationTermCfg(func=base_height),
  "ball_mass": ObservationTermCfg(
    func=ball_mass,
    params={"ball_mass": BALL_MASS},
  ),
  "foot_ball_contact_force": ObservationTermCfg(
    func=foot_ball_contact_force,
    params={"sensor_name": "foot_ball_contact"},
  ),
  "ball_friction": ObservationTermCfg(
    func=ball_friction,
    params={"ball_friction": BALL_FRICTION},
  ),
  "base_lin_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_lin_vel"},
    noise=Unoise(n_min=-0.5, n_max=0.5),
  ),
  "foot_height": ObservationTermCfg(
    func=foot_height,
    params={"sensor_name": "foot_height_scan"},
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

# ---------------------------------------------------------------------------
# Observation groups
# ---------------------------------------------------------------------------

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
  "privileged_ball": ObservationGroupCfg(
    terms=privileged_ball_terms,
    concatenate_terms=True,
    enable_corruption=False,
  ),
}
