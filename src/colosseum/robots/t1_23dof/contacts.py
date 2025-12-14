"""Booster T1 contact and collision configurations."""

from mjlab.sensor import ContactMatch, ContactSensorCfg
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

##
# Contact Sensor Configurations (for observation/reward)
# These create MuJoCo contact sensors dynamically (NOT in XML)
##

# Foot-ground contact sensor
# Tracks aggregate contact between each foot body and terrain
# Returns 2 contact points (left_foot, right_foot)
# Includes air time tracking for landing/takeoff detection
FEET_GROUND_CONTACT_SENSOR = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
        mode="body",
        pattern=r"^(left_foot_link|right_foot_link)$",
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
)

# Self-collision sensor
# Detects collisions between robot body parts
# Useful for penalty/safety rewards
SELF_COLLISION_SENSOR = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
)

# Hand contact sensor (for manipulation)
# Tracks contact between hand sphere end-effectors and any objects
HAND_CONTACT_SENSOR = ContactSensorCfg(
    name="hand_contact",
    primary=ContactMatch(
        mode="body",
        pattern=r"^(left_hand_sphere_link|right_hand_sphere_link)$",
        entity="robot",
    ),
    secondary=None,  # Any contact
    fields=("found", "force", "pos"),
    reduce="maxforce",
    num_slots=1,
)
