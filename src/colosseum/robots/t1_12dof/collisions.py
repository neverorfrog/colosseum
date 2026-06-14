"""Booster T1 12-DOF collision configurations (locomotion only)."""

from mjlab.utils.spec_config import CollisionCfg

# Feet-only collision (recommended for flat-ground locomotion training).
# Only the foot capsules collide with the terrain; no self-collision.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=4,
  priority=1,
  friction=(1.0, 0.04),
)

# Full collision without self-collision (for rough terrain, where a foot sphere
# slipping a tile seam still catches on the shin/knee cylinders).
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(r".+",),  # Named geoms only; visual geoms are unnamed.
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot.*": 3, r".*": 1},
  priority={r"^(left|right)_foot.*": 1},
  friction={r"^(left|right)_foot.*": (0.6,)},
)
