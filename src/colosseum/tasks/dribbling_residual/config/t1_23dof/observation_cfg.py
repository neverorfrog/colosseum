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

from colosseum.mdp.observations import base_height
from colosseum.mdp.symmetry import (
  MirrorableObservationTermCfg,
  mirror_ang_vel,
  mirror_base_lin_vel,
  mirror_gait_phase,
  mirror_projected_gravity,
  mirror_velocity_command,
)
from colosseum.robots.t1_23dof.mdp.symmetry import mirror_actions, mirror_joints
from colosseum.tasks.dribbling.mdp.observations import (
  # ball_position,
  ball_vel_command_body,
  # ball_velocity_xy,
)
from colosseum.tasks.dribbling_residual.mdp.ball_perception import (
  BallPerceptionModel,
  ball_state_gt,
  ball_velocity_xy,
)

# ---------------------------------------------------------------------------
# Shared proprioception (both the frozen walk branch and the residual branch
# see these). Order here is part of the frozen walk checkpoint's input layout,
# so do not reorder without retraining the walk policy.
# ---------------------------------------------------------------------------

proprio_terms = {
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
    mirror_fn=mirror_actions,
  ),
}

command_term = MirrorableObservationTermCfg(
  func=generated_commands,
  params={"command_name": "twist"},
  mirror_fn=mirror_velocity_command,
)
gait_phase_term = MirrorableObservationTermCfg(
  func=generated_commands,
  params={"command_name": "gait_phase"},
  mirror_fn=mirror_gait_phase,
)

ball_terms = {
  # "ball_pos": ObservationTermCfg(func=ball_position),
  # "ball_vel_xy": ObservationTermCfg(func=ball_velocity_xy),
  "ball_state": ObservationTermCfg(
    func=BallPerceptionModel,
  ),  # (N, 4)
}

# ---------------------------------------------------------------------------
# Skill actor groups
# ---------------------------------------------------------------------------

# Frozen walk branch: exactly the velocity policy's actor layout
# (proprio + twist command + gait phase). The twist is BallTwistCommand, so the
# walk naturally steers toward the ball without a separate chase skill.
loco_actor_terms = {
  **proprio_terms,
  "command": command_term,
  "gait_phase": gait_phase_term,
}

# Trainable residual branch: proprio + gait phase + ball state + dribble
# setpoint, no twist command. The dribble command key must differ from loco's
# "command" so the orchestrator union keeps both.
dribble_actor_terms = {
  **proprio_terms,
  "gait_phase": gait_phase_term,
  **ball_terms,
  "ball_vel_command": ObservationTermCfg(
    func=ball_vel_command_body,
    params={"command_name": "ball_vel"},
  ),
}

# Orchestrator (gating net): deduplicated union of all skill observations.
# Dict merge collapses shared term names, so each term appears once.
orchestrator_terms = {**loco_actor_terms, **dribble_actor_terms}

# ---------------------------------------------------------------------------
# Critic: actor terms + privileged ground-truth state (asymmetric actor-critic).
# ---------------------------------------------------------------------------

critic_terms = {
  **orchestrator_terms,
  # Override the actor's noisy ball estimate with clean GT (keeps the union's
  # slot position, swaps only the value). The critic is train-only and may use
  # privileged truth for lower-variance value estimates.
  "ball_state": ObservationTermCfg(func=ball_state_gt),
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
  "loco_actor": ObservationGroupCfg(
    terms=loco_actor_terms,
    concatenate_terms=True,
    enable_corruption=True,
  ),
  "dribble_actor": ObservationGroupCfg(
    terms=dribble_actor_terms,
    concatenate_terms=True,
    enable_corruption=True,
  ),
  "orchestrator": ObservationGroupCfg(
    terms=orchestrator_terms,
    concatenate_terms=True,
    enable_corruption=False,
  ),
  "critic": ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  ),
}
