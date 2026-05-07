from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel

from colosseum.mdp.curriculums import (
  penalty_curriculum,
  standing_curriculum_by_transitions,
)

curriculum = {
  "command_vel": CurriculumTermCfg(
    func=commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {
          "step": 0,
          "lin_vel_x": (-1.0, 1.0),
          "lin_vel_y": (-0.5, 0.5),
          "ang_vel_z": (-0.5, 0.5),
        },
        {
          "step": 5000 * 24,
          "lin_vel_x": (-1.5, 2.0),
          "lin_vel_y": (-1.0, 1.0),
          "ang_vel_z": (-0.7, 0.7),
        },
        {
          "step": 10000 * 24,
          "lin_vel_x": (-2.0, 3.0),
          "lin_vel_y": (-2.0, 2.0),
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
        {"transitions": 0, "rel_standing_envs": 0.3},
        {"transitions": 50_000_000, "rel_standing_envs": 0.1},
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
      ],
      "initial_scale": 0.1,
      "min_scale": 0.0,
      "max_scale": 1.0,
      "level_down_threshold": 150,
      "level_up_threshold": 750,
      "degree": 0.00025,
    },
  ),
}
