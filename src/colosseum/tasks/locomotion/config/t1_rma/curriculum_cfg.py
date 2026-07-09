from mjlab.managers.curriculum_manager import CurriculumTermCfg

from colosseum.mdp.curriculums import (
  penalty_curriculum,
)

curriculum = {
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
}
