import math

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from colosseum.tasks.dribbling_symmetric.mdp.curriculum import (
  obstacle_curriculum,
  push_ball_curriculum,
  yaw_reset_curriculum,
)

curriculum = {
  "yaw_reset": CurriculumTermCfg(
    func=yaw_reset_curriculum,
    params={
      "event_name": "reset_base",
      "stages": [
        {"transitions": 0, "half_range": 0.0},  # always forward
        {"transitions": 50_000_000, "half_range": math.pi / 6},  # ±30°
        {"transitions": 100_000_000, "half_range": math.pi / 3},  # ±60°
        {"transitions": 150_000_000, "half_range": math.pi / 2},  # ±90°
        {"transitions": 200_000_000, "half_range": math.pi},  # ±180° (full)
      ],
    },
  ),
  "push_ball": CurriculumTermCfg(
    func=push_ball_curriculum,
    params={
      "event_name": "push_ball",
      "stages": [
        {"transitions": 0, "max_speed": 0.3},
        {"transitions": 100_000_000, "max_speed": 0.6},
        {"transitions": 200_000_000, "max_speed": 1.0},
      ],
    },
  ),
  "obstacle": CurriculumTermCfg(
    func=obstacle_curriculum,
    params={
      "command_name": "adversary",
      "stages": [
        {
          "step": 0,
          "num_active": 0,
          "behavior": "none",
          "max_speed": 0.0,
        },
      ],
    },
  ),
}
