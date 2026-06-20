from mjlab.managers.curriculum_manager import CurriculumTermCfg

from colosseum.mdp.curriculums import (
  push_force_curriculum_by_transitions,
  reward_curriculum_by_transitions,
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
  "ball_speed_kick_curriculum": CurriculumTermCfg(
    func=reward_curriculum_by_transitions,
    params={
      "reward_name": "ball_speed_kick",
      "stages": [
        {"transitions": 0, "weight": 2.0, "params": {"min_contact_force": 2.0}},
        {"transitions": 100_000_000, "weight": 4.0, "params": {"min_contact_force": 4.0}},
        {"transitions": 200_000_000, "weight": 5.0, "params": {"min_contact_force": 6.0}},
      ],
    },
  ),
  "ball_vel_angle_curriculum": CurriculumTermCfg(
    func=reward_curriculum_by_transitions,
    params={
      "reward_name": "ball_vel_angle",
      "stages": [
        {"transitions": 0, "weight": 2.0, "params": {"min_speed": 0.5}},
        {"transitions": 100_000_000, "weight": 1.5, "params": {"min_speed": 1.0}},
        {"transitions": 200_000_000, "weight": 1.0, "params": {"min_speed": 1.0}},
      ],
    },
  ),
}
