from mjlab.managers.curriculum_manager import CurriculumTermCfg

from colosseum.mdp.curriculums import (
  command_vel_curriculum,
  penalty_curriculum,
  push_force_curriculum_by_transitions,
  standing_curriculum_by_transitions,
)

curriculum = {
  "command_vel": CurriculumTermCfg(
    func=command_vel_curriculum,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {
          "transitions": 0,
          "lin_vel_x": (-0.2, 0.5),
          "lin_vel_y": (-0.2, 0.2),
          "ang_vel_z": (-0.5, 0.5),
        },
        {
          "transitions": 100_000_000,
          "lin_vel_x": (-0.3, 1.0),
          "lin_vel_y": (-0.5, 0.5),
          "ang_vel_z": (-0.75, 0.75),
        },
        {
          "transitions": 200_000_000,
          "lin_vel_x": (-1.0, 1.5),
          "lin_vel_y": (-1.0, 1.0),
          "ang_vel_z": (-1.0, 1.0),
        },
      ],
    },
  ),
  "standing_curriculum": CurriculumTermCfg(
    func=standing_curriculum_by_transitions,
    params={
      "command_name": "twist",
      "stages": [
        {"transitions": 0, "rel_standing_envs": 0.5},
        {"transitions": 50_000_000, "rel_standing_envs": 0.3},
        {"transitions": 100_000_000, "rel_standing_envs": 0.2},
        {"transitions": 150_000_000, "rel_standing_envs": 0.1},
      ],
    },
  ),
  "push_curriculum": CurriculumTermCfg(
    func=push_force_curriculum_by_transitions,
    params={
      "event_name": "push_robot",
      "stages": [
        # Reference-matched: gaussian std 10N/2Nm ≈ uniform ±17/±3.5 (a = std·√3).
        {
          "transitions": 0,
          "force_range": (-17.0, 17.0),
          "torque_range": (-3.5, 3.5),
        },
        # Headroom beyond the reference's gaussian tail for extra robustness.
        {
          "transitions": 200_000_000,
          "force_range": (-25.0, 25.0),
          "torque_range": (-5.0, 5.0),
        },
      ],
    },
  ),
  "penalty_curriculum": CurriculumTermCfg(
    func=penalty_curriculum,
    params={
      "reward_names": [
        "penalty_body_ang_vel",
        "penalty_orientation",
        "penalty_action_rate",
        "penalty_feet_distance",
        "penalty_feet_ori",
        "penalty_pose_deviation",
        "penalty_landing",
        "penalty_dof_vel",
        "penalty_dof_acc",
        "feet_slip",
      ],
      "initial_scale": 0.1,
      "min_scale": 0.01,
      "max_scale": 1.0,
      "level_down_threshold": 150,
      "level_up_threshold": 750,
      "degree": 0.00025,
    },
  ),
}
