from pathlib import Path

import mujoco
import mujoco.viewer as viewer
from mjlab.actuator import (  # Back to position control for stability
  BuiltinPositionActuatorCfg,
)
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg

from colosseum.utils import src_dir

# ====== MJCF and Assets ======

XML: Path = src_dir() / "robots" / "ant" / "xmls" / "ant.xml"
assert XML.exists(), f"XML not found: {XML}"


def get_spec() -> mujoco.MjSpec:
  """Load the ant robot specification from XML."""
  return mujoco.MjSpec.from_file(str(XML))


# ====== Actuator Configuration ======

HIP_REFLECTED_INERTIA = 0.012
ANKLE_REFLECTED_INERTIA = 0.01
HIP_EFFORT_LIMIT = 250.0
ANKLE_EFFORT_LIMIT = 250.0
DAMPING_RATIO = 2.0
NATURAL_FREQ = 10 * 2.0 * 3.1415926535

# Position actuators
ANT_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_.*",),
  stiffness=HIP_REFLECTED_INERTIA * NATURAL_FREQ**2,
  damping=2 * DAMPING_RATIO * HIP_REFLECTED_INERTIA * NATURAL_FREQ,
  effort_limit=HIP_EFFORT_LIMIT,
  armature=HIP_REFLECTED_INERTIA,
)

ANT_ANKLE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("ankle_.*",),
  stiffness=ANKLE_REFLECTED_INERTIA * NATURAL_FREQ**2,
  damping=2 * DAMPING_RATIO * ANKLE_REFLECTED_INERTIA * NATURAL_FREQ,
  effort_limit=ANKLE_EFFORT_LIMIT,
  armature=ANKLE_REFLECTED_INERTIA,
)

# ====== Articulation ======

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ANT_HIP_ACTUATOR_CFG, ANT_ANKLE_ACTUATOR_CFG),
  soft_joint_pos_limit_factor=0.85,  # Scale down joint limits for smoother control
)

ANT_ACTION_SCALE: dict[str, float] = {}
for a in ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  if a.target_names_expr == ("hip_.*",):
    factor = 0.2
  elif a.target_names_expr == ("ankle_.*",):
    factor = 0.2
  else:
    raise ValueError(f"Unexpected actuator target pattern: {a.target_names_expr}")
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    ANT_ACTION_SCALE[n] = factor * e / s

# ====== Home Keyframe ======

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.75),  # Match torso height from XML (0.75)
  rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion (no rotation)
  joint_pos={
    "hip_1": 0.0,
    "ankle_1": 1.0,
    "hip_2": 0.0,
    "ankle_2": -1.0,
    "hip_3": 0.0,
    "ankle_3": -1.0,
    "hip_4": 0.0,
    "ankle_4": 1.0,
  },
  joint_vel={".*": 0.0},
)

# ====== Collision Config ======

# Foot geoms (the last segment of each leg - ankle geoms)
_foot_regex = "^(left_ankle_geom|right_ankle_geom|third_ankle_geom|fourth_ankle_geom)$"

# Enable collisions for all geoms with softer foot contact properties
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_geom",),
  condim={_foot_regex: 3, ".*ankle_geom": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (1.0, 0.5, 0.5)},
  solimp={_foot_regex: (0.9, 0.97, 0.01)},
  contype=1,
  conaffinity=0,  # Prevent self-collision (robot parts don't collide with each other)
)

# ====== Sensors ======

FOOT_GEOM_NAMES = (
  "left_ankle_geom",
  "right_ankle_geom",
  "third_ankle_geom",
  "fourth_ankle_geom",
)

FEET_GROUND_CONTACT_CFG = ContactSensorCfg(
  name="feet_ground_contact",
  primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
  track_air_time=True,
)

# ====== Robot Config ======


def get_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=ARTICULATION,
    collisions=(FULL_COLLISION,),
  )


ANT_ROBOT_CFG = get_robot_cfg()

# ====== Test Visualization ======

if __name__ == "__main__":
  """Visualize the ant robot in isolation."""
  robot = Entity(ANT_ROBOT_CFG)
  viewer.launch(robot.spec.compile())
