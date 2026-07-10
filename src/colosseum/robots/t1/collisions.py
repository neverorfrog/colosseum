"""Booster T1 contact and collision configurations."""

from mjlab.utils.spec_config import CollisionCfg

##
# Collision Configurations (for physics simulation)
# These modify geom collision properties in the XML
##

# Feet-only collision (recommended for locomotion training)
# - Each foot collides via a single flat box (booster_gym foot, full fidelity):
#   rigid vertical-normal contacts at 4 corners instead of rockable capsule ends.
# - The sole capsules and the vertical inner-face capsule (foot5) are disabled
#   here (disable_other_geoms); they remain for dribbling (inside-foot ball touch).
# - condim=3 like booster_gym: yaw is resisted geometrically by the box corners,
#   no torsional-friction term.
# - No self-collisions; most stable for training.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(
    r"^(left|right)_foot[1-4]_collision$",
  ),  # Only match foot collision capsules
  contype=0,
  conaffinity=1,
  condim=4,
  priority=1,
  friction=(1.0, 0.08),
)

DRIBBLING_FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(
    r"^(left|right)_foot[1-5]_collision$",
  ),  # Only match foot collision capsules
  contype=0,
  conaffinity=1,
  condim=4,
  priority=1,
  friction=(1.0, 0.08),
)

# Feet collision with inter-foot contacts enabled (for dribbling)
# - Foot geoms collide with environment (bit 0) AND each other (bit 1)
# - contype=3 (bits 0+1), conaffinity=3 (bits 0+1)
# - Ground must have contype=1 (default) so bit 0 matches
FEET_SELF_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=3,
  conaffinity=3,
  condim=3,
  priority=1,
  friction=(1.0,),
)

# Full collision without self-collision
# - All body parts collide with environment
# - No self-collisions between robot parts
# - Good for general tasks
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(r".+",),  # Named geoms only; visual geoms are unnamed in this XML
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot.*": 3, r".*": 1},
  priority={r"^(left|right)_foot.*": 1},
  friction={r"^(left|right)_foot.*": (0.6,)},
)

# Full collision with self-collision
# - All body parts collide with environment
# - Self-collisions enabled between robot parts
# - Most realistic but can be unstable for training
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".+",),  # Named geoms only; visual geoms are unnamed in this XML
  condim={r"^(left|right)_foot.*": 3, r".*": 1},
  priority={r"^(left|right)_foot.*": 1},
  friction={r"^(left|right)_foot.*": (0.6,)},
)
