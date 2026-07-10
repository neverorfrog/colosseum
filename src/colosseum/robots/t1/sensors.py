from mjlab.sensor import (
  CameraSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)

from .constants import (
  BASE_BODY_NAME,
  FOOT_GEOM_NAMES,
  HEAD_CAMERA_HEIGHT,
  HEAD_CAMERA_NAME,
  HEAD_CAMERA_WIDTH,
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
  reduce="netforce",
  num_slots=1,
  history_length=4,
)

FOOT_BALL_CONTACT_SENSOR = ContactSensorCfg(
  name="foot_ball_contact",
  primary=ContactMatch(
    mode="subtree",
    # Foot bodies; the subtree includes their foot[1-5]_collision capsules,
    # the only enabled foot colliders (see FEET_ONLY/FEET_SELF_COLLISION).
    pattern=r"^(left|right)_foot_link$",
    entity="robot",
  ),
  secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
  history_length=4,
)

# Inner-foot (medial) contact with the ball. Matches only the foot5_collision
# capsules — the vertical inside-face colliders meant for the inside-foot dribble
# touch — so a reward can pay specifically for inside-of-the-foot ball contact,
# which FOOT_BALL_CONTACT_SENSOR cannot isolate (it netforce-aggregates foot1-5).
FOOT5_BALL_CONTACT_SENSOR = ContactSensorCfg(
  name="foot_inner_ball_contact",
  primary=ContactMatch(
    mode="geom",
    entity="robot",
    pattern=r"^(left|right)_foot5_collision$",
  ),
  secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
  history_length=4,
)

FOOT_HEIGHT_SCAN = TerrainHeightSensorCfg(
  name="foot_height_scan",
  frame=tuple(
    ObjRef(type="site", name=s, entity="robot") for s in ("left_foot", "right_foot")
  ),
  pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
  ray_alignment="yaw",
  max_distance=1.0,
  exclude_parent_body=True,
  include_geom_groups=(0,),  # Terrain only.
)

# Base height above the terrain directly under the robot (terrain-relative, so it
# stays correct on bumpy ground unlike an env-origin-relative height). Scalar
# clearance per env; privileged critic obs only.
BASE_HEIGHT_SCAN = TerrainHeightSensorCfg(
  name="base_height_scan",
  frame=(ObjRef(type="body", name=BASE_BODY_NAME, entity="robot"),),
  pattern=RingPatternCfg.single_ring(radius=0.05, num_samples=4),
  ray_alignment="yaw",
  reduction="min",  # -> [B, 1] scalar clearance
  max_distance=1.0,
  exclude_parent_body=True,
  include_geom_groups=(0,),  # Terrain only.
)

WALL_COLLISION_SENSOR = ContactSensorCfg(
  name="wall_collision",
  primary=ContactMatch(
    mode="geom",
    entity="robot",
    pattern=r".+",
    exclude=tuple(FOOT_GEOM_NAMES),
  ),
  # No secondary: fires for any contact with non-foot robot geoms.
  # Wall geoms are the only non-terrain objects in the maze, so non-foot
  # contact is effectively wall contact (ground contact only during falls,
  # which already trigger height-based termination).
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
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

##
# Camera Sensor Configurations
##

# Head-mounted RGB-D camera wrapping the calibrated D455 defined in
# get_spec_with_head_camera(). Parameters match mjlab's booster_t1_rgbd_camera demo.
HEAD_RGBD_SENSOR = CameraSensorCfg(
  name="head_rgbd",
  camera_name=f"robot/{HEAD_CAMERA_NAME}",
  width=HEAD_CAMERA_WIDTH,
  height=HEAD_CAMERA_HEIGHT,
  data_types=("rgb", "depth"),
  use_textures=True,
  use_shadows=False,
)

# Low-resolution depth-only sensor for Phase 2 training.
# Renders at the encoder's native input resolution (128x72) so no downsampling
# is needed and GPU memory scales ~300x better than HEAD_RGBD_SENSOR.
# Use this in training configs; HEAD_RGBD_SENSOR is for visualisation / play.
HEAD_DEPTH_SENSOR_TRAIN = CameraSensorCfg(
  name="head_rgbd",
  camera_name=f"robot/{HEAD_CAMERA_NAME}",
  width=128,
  height=72,
  data_types=("depth",),
  use_textures=False,
  use_shadows=False,
)
