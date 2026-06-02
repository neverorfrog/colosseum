from mjlab.managers.curriculum_manager import CurriculumTermCfg

from colosseum.mdp.curriculums import (
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
          "transitions": 100_000_000,
          "force_range": (-10.0, 10.0),
          "torque_range": (-2.0, 2.0),
        },
        {
          "transitions": 200_000_000,
          "force_range": (-20.0, 20.0),
          "torque_range": (-3.0, 3.0),
        },
        {
          "transitions": 400_000_000,
          "force_range": (-30.0, 30.0),
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
