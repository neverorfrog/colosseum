"""Booster T1 contact and collision configurations."""

from mjlab.utils.spec_config import CollisionCfg

##
# Collision Configurations (for physics simulation)
# These modify geom collision properties in the XML
##

# Feet-only collision (recommended for locomotion training)
# - Only foot geoms collide with environment
# - No self-collisions
# - Most stable for training
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot_sphere.*link$",),  # Only match sphere links
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

# Feet collision with inter-foot contacts enabled (for dribbling)
# - Foot geoms collide with environment (bit 0) AND each other (bit 1)
# - contype=3 (bits 0+1), conaffinity=3 (bits 0+1)
# - Ground must have contype=1 (default) so bit 0 matches
FEET_SELF_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot_sphere.*link$",),
  contype=3,
  conaffinity=3,
  condim=3,
  priority=1,
  friction=(0.6,),
)

# Full collision without self-collision
# - All body parts collide with environment
# - No self-collisions between robot parts
# - Good for general tasks
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot.*": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot.*": 1},
  friction={r"^(left|right)_foot.*": (0.6,)},
)

# Full collision with self-collision
# - All body parts collide with environment
# - Self-collisions enabled between robot parts
# - Most realistic but can be unstable for training
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot.*": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot.*": 1},
  friction={r"^(left|right)_foot.*": (0.6,)},
)

# Hands and feet collision (for manipulation tasks)
HANDS_FEET_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_(foot|hand).*",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)
