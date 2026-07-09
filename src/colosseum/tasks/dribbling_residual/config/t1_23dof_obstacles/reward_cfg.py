from mjlab.managers import RewardTermCfg

from colosseum.tasks.dribbling.mdp.rewards import robot_obstacle_collision

from ..t1.reward_cfg import rewards as _base

rewards = dict(_base)
rewards["robot_obstacle_collision"] = RewardTermCfg(
  func=robot_obstacle_collision,
  weight=-3.0,
  params={
    "command_name": "adversary",
    "collision_near_distance": 0.5,
    "collision_far_distance": 2.0,
  },
)