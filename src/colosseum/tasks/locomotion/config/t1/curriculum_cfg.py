from mjlab.managers.curriculum_manager import CurriculumTermCfg

from colosseum.mdp.curriculums import (
  command_vel_curriculum,
  penalty_curriculum,
  push_force_curriculum_by_transitions,
)

# The command-range and standing distributions are now owned by the grid-based,
# performance-gated CurriculumVelocityCommand (per-env, in cat_cfg.py), so the
# old time-based command_vel / standing curricula are removed here.
curriculum = {
  "push_curriculum": CurriculumTermCfg(
    func=push_force_curriculum_by_transitions,
    params={
      "event_name": "push_robot",
      "stages": [
        {
          "transitions": 0,
          "force_range": (-5.0, 5.0),
          "torque_range": (-1.0, 1.0),
        },
        {
          "transitions": 50_000_000,
          "force_range": (-10.0, 10.0),
          "torque_range": (-2.0, 2.0),
        },
        {
          "transitions": 100_000_000,
          "force_range": (-20.0, 20.0),
          "torque_range": (-3.0, 3.0),
        },
        {
          "transitions": 150_000_000,
          "force_range": (-50.0, 50.0),
          "torque_range": (-10.0, 10.0),
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
        "penalty_landing",
        "penalty_lin_vel_z",
        "penalty_feet_yaw_mean",
        "penalty_feet_yaw_diff",
        "penalty_feet_slip",
        "penalty_pose_deviation",
      ],
      "initial_scale": 0.1,
      "min_scale": 0.01,
      "max_scale": 1.0,
      "level_down_threshold": 150,
      "level_up_threshold": 750,
      "degree": 0.00025,
    },
  ),
  "command_vel_curriculum": CurriculumTermCfg(
    func=command_vel_curriculum,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {
          "transitions": 0,
          "lin_vel_x": (-0.5, 0.5),
          "lin_vel_y": (-0.5, 0.5),
          "ang_vel_z": (-0.5, 0.5),
        },
        {
          "transitions": 50_000_000,
          "lin_vel_x": (-0.75, 0.75),
          "lin_vel_y": (-0.75, 0.75),
          "ang_vel_z": (-0.75, 0.75),
        },
        {
          "transitions": 100_000_000,
          "lin_vel_x": (-1.0, 1.0),
          "lin_vel_y": (-0.75, 0.75),
          "ang_vel_z": (-1.0, 1.0),
        },
        {
          "transitions": 150_000_000,
          "lin_vel_x": (-1.5, 1.5),
          "lin_vel_y": (-1.25, 1.25),
          "ang_vel_z": (-1.5, 1.5),
        },
      ],
    },
  ),
}
