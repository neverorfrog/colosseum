"""Booster T1 12-DOF contact / height sensors (locomotion subset)."""

from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)

from .constants import FOOT_GEOM_NAMES

# Foot-ground contact with air-time tracking (landing/takeoff detection).
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

# Any non-foot robot geom touching the terrain (fall / stumble detection).
NONFOOT_GROUND_CONTACT_SENSOR = ContactSensorCfg(
  name="nonfoot_ground_touch",
  primary=ContactMatch(
    mode="geom",
    entity="robot",
    pattern=r".+",
    exclude=tuple(FOOT_GEOM_NAMES),
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="none",
  num_slots=1,
  history_length=4,
)

# Per-foot terrain-height ring scan (swing-height / feet-phase rewards).
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
