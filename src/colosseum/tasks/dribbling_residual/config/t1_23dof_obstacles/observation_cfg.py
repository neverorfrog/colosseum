"""Observation groups for t1-dribbling-residual-obstacles.

Carries forward the base groups (loco_actor consumed by the frozen RMA walk,
dribble_actor, orchestrator, critic) and adds obstacle perception to the
trainable branches:
  - ``obstacle_residual``: the dribble_actor group (proprio + gait + ball) plus
    the obstacle estimate, so the single residual can both dribble and avoid.
  - ``obstacle_orchestrator``: the orchestrator union plus the obstacle estimate.
The critic adds clean GT obstacle_state_gt.
"""

from copy import deepcopy

from mjlab.managers import ObservationGroupCfg, ObservationTermCfg

from colosseum.tasks.dribbling.mdp.obstacle_perception import (
  ObstaclePerceptionModel,
  obstacle_state_gt,
)

from ..t1.observation_cfg import (
  dribble_actor_terms,
  observations as _base_observations,
  orchestrator_terms as _base_orchestrator_terms,
)

observations = {group: deepcopy(cfg) for group, cfg in _base_observations.items()}

obstacle_perception_term = ObservationTermCfg(
  func=ObstaclePerceptionModel,
  params={
    "command_name": "adversary",
    "sigma_pos_base": 0.03,
    "sigma_pos_per_m": 0.05,
    "sigma_pos_scale_range": (0.5, 2.0),
    "sigma_vel_range": (0.1, 0.4),
    "vel_filter_alpha": 0.3,
    "coast_vel_decay": 0.98,
    "p_miss_range": (0.02, 0.2),
    "max_range": 6.0,
    "latency_steps": 2,
  },
)

obstacle_residual_terms = {
  **dribble_actor_terms,
  "obstacle_state": obstacle_perception_term,
}
observations["obstacle_residual"] = ObservationGroupCfg(
  terms=obstacle_residual_terms,
  concatenate_terms=True,
  enable_corruption=True,
)

obstacle_orchestrator_terms = {
  **_base_orchestrator_terms,
  "obstacle_state": obstacle_perception_term,
}
observations["obstacle_orchestrator"] = ObservationGroupCfg(
  terms=obstacle_orchestrator_terms,
  concatenate_terms=True,
  enable_corruption=False,
)

observations["critic"].terms["obstacle_state_gt"] = ObservationTermCfg(
  func=obstacle_state_gt,
  params={"command_name": "adversary"},
)
