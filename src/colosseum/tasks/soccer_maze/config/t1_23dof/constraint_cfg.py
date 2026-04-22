"""CaT constraint terms for the T1 soccer-maze task."""

from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.managers import ConstraintTermCfg
from colosseum.robots.t1_23dof.constants import FOOT_SITE_NAMES
from colosseum.tasks.dribbling.mdp.rewards import feet_distance_penalty

constraints: dict[str, ConstraintTermCfg] = {
  # Feet crossing: stochastic termination proportional to severity.
  # max_p=0.3 means up to 30% per-step termination probability when feet
  # are fully overlapping; mild crossings get proportionally less.
  "feet_distance": ConstraintTermCfg(
    func=feet_distance_penalty,
    max_p=0.25,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "min_dist": 0.05,
    },
  ),
}
