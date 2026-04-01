from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)

from .constants import BASE_BODY_NAME, FOOT_GEOM_NAMES

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
    mode="subtree",
    pattern=r"^(left_foot_link|right_foot_link)$",
    entity="robot",
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
  track_air_time=True,
)

NONFOOT_GROUND_CONTACT_SENSOR = ContactSensorCfg(
  name="nonfoot_ground_touch",
  primary=ContactMatch(
    mode="geom",
    entity="robot",
    # Match all named geoms...
    pattern=r".+",
    # Except for the foot geoms.
    exclude=tuple(FOOT_GEOM_NAMES),
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="none",
  num_slots=1,
  history_length=4,
)

# Self-collision sensor
# Detects collisions between robot body parts
# Useful for penalty/safety rewards
SELF_COLLISION_SENSOR = ContactSensorCfg(
  name="self_collision",
  primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
  secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
  fields=("found", "force"),
  reduce="none",
  num_slots=1,
  history_length=4,
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


FOOT_FOOT_CONTACT_SENSOR = ContactSensorCfg(
  name="foot_foot_contact",
  primary=ContactMatch(
    mode="subtree",
    pattern="left_foot_link",
    entity="robot",
  ),
  secondary=ContactMatch(
    mode="subtree",
    pattern="right_foot_link",
    entity="robot",
  ),
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
)

NONFOOT_BALL_CONTACT_SENSOR = ContactSensorCfg(
  name="nonfoot_ball_contact",
  primary=ContactMatch(
    mode="geom",
    entity="robot",
    pattern=r".+",
    exclude=tuple(FOOT_GEOM_NAMES),
  ),
  secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
  fields=("found", "force"),
  reduce="none",
  num_slots=1,
  history_length=4,
)

FOOT_BALL_CONTACT_SENSOR = ContactSensorCfg(
  name="foot_ball_contact",
  primary=ContactMatch(
    mode="subtree",
    pattern=r"^(left_foot_link|right_foot_link)$",
    entity="robot",
  ),
  secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
)

FOOT_HEIGHT_SCAN = TerrainHeightSensorCfg(
  name="foot_height_scan",
  frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in ("left_foot", "right_foot")),
  pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
  ray_alignment="yaw",
  max_distance=1.0,
  exclude_parent_body=True,
  include_geom_groups=(0,),  # Terrain only.
)

TERRAIN_SCAN = RayCastSensorCfg(
  name="terrain_scan",
  frame=ObjRef(type="body", name=BASE_BODY_NAME, entity="robot"),
  ray_alignment="yaw",
  pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
  max_distance=5.0,
  exclude_parent_body=True,
  debug_vis=True,
  viz=RayCastSensorCfg.VizCfg(show_normals=True),
)
