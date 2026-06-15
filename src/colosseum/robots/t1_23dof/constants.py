"""Booster T1 robot configuration.

Deploy only needs file paths and static constants. Training-only mjlab imports are
guarded so deploy can import this module without mjlab installed.
"""

import mujoco
import numpy as np

try:  # pragma: no cover - train-only dependency
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

  _MJLAB_AVAILABLE = True
except ImportError:
  _MJLAB_AVAILABLE = False

if _MJLAB_AVAILABLE:
  from colosseum.robots.t1_23dof.actuators import (
    LOCOMOTION_ACTUATORS,
    WHOLEBODY_ACTUATORS,
  )
  from colosseum.robots.t1_23dof.collisions import (
    FEET_FOREARM_WAIST_COLLISION,
    FEET_ONLY_COLLISION,
    FEET_SELF_COLLISION,
    FULL_COLLISION_WITHOUT_SELF,
  )
from colosseum.utils import src_dir

##
# MJCF and assets.
##

# Path to unified base XML (23-DOF full body)
# Locomotion (12-DOF) vs Full-body (23-DOF) is controlled by which actuators are added
XML = src_dir() / "robots" / "t1_23dof" / "xmls" / "T1_23dof.xml"

assert XML.exists(), f"XML not found: {XML}"


if _MJLAB_AVAILABLE:

  def get_spec() -> mujoco.MjSpec:
    """Load T1 base model (23 DOF structure, actuators added via Python)."""
    spec = mujoco.MjSpec.from_file(str(XML))
    spec.actuators.clear()
    return spec


# Head camera constants (pose relative to H2 body, matching real D455 mount)
HEAD_CAMERA_NAME = "d455_color"
HEAD_CAMERA_WIDTH = 1280
HEAD_CAMERA_HEIGHT = 720

# RealSense D455 calibrated intrinsics.
HEAD_CAMERA_K = np.array(
  [
    [646.0612, 0.0, 644.3064],
    [0.0, 645.1986, 357.1254],
    [0.0, 0.0, 1.0],
  ],
  dtype=np.float64,
)

HEAD_CAMERA_FOVY = float(
  np.degrees(2.0 * np.arctan(HEAD_CAMERA_HEIGHT / (2.0 * HEAD_CAMERA_K[1, 1])))
)


def _set_mujoco_camera_intrinsics(
  cam: mujoco.MjsCamera,
  width: int,
  height: int,
  fx: float,
  fy: float,
  cx: float,
  cy: float,
) -> None:
  """Set calibrated intrinsics on a MuJoCo spec camera."""
  cam.resolution[:] = (width, height)
  if hasattr(cam, "focalpixel") and hasattr(cam, "principalpixel"):
    setattr(cam, "focalpixel", np.array([fx, fy], dtype=np.float64))
    setattr(cam, "principalpixel", np.array([cx, cy], dtype=np.float64))
  else:
    cam.fovy = float(np.degrees(2.0 * np.arctan(height / (2.0 * fy))))


if _MJLAB_AVAILABLE:

  def get_spec_with_head_camera() -> mujoco.MjSpec:
    """T1 spec with a calibrated D455 RGB-D camera on the H2 head body.

    Camera pose, resolution, and intrinsics match the real D455 mount
    as defined in mjlab's booster_t1_rgbd_camera.py demo.
    """
    spec = get_spec()
    h2 = spec.body("H2")
    cam = h2.add_camera(
      name=HEAD_CAMERA_NAME,
      pos=(0.074, 0.0, 0.11),
      quat=(0.5, 0.5, -0.5, -0.5),
      fovy=HEAD_CAMERA_FOVY,
      resolution=[HEAD_CAMERA_WIDTH, HEAD_CAMERA_HEIGHT],
      proj=mujoco.mjtProjection.mjPROJ_PERSPECTIVE,
    )
    _set_mujoco_camera_intrinsics(
      cam=cam,
      width=HEAD_CAMERA_WIDTH,
      height=HEAD_CAMERA_HEIGHT,
      fx=float(HEAD_CAMERA_K[0, 0]),
      fy=float(HEAD_CAMERA_K[1, 1]),
      cx=float(HEAD_CAMERA_K[0, 2]),
      cy=float(HEAD_CAMERA_K[1, 2]),
    )
    return spec


##
# Actuator config.
##

# Booster T1 23-DOF joint names (MuJoCo XML depth-first order, matches arena sim_joint_names).
JOINT_NAMES = [
  # Head - 2 DOF
  "AAHead_yaw",
  "Head_pitch",
  # Upper body (arms) - 9 DOF
  "Left_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "Right_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
  "Waist",
  # Lower body - 12 DOF
  "Left_Hip_Pitch",
  "Left_Hip_Roll",
  "Left_Hip_Yaw",
  "Left_Knee_Pitch",
  "Left_Ankle_Pitch",
  "Left_Ankle_Roll",
  "Right_Hip_Pitch",
  "Right_Hip_Roll",
  "Right_Hip_Yaw",
  "Right_Knee_Pitch",
  "Right_Ankle_Pitch",
  "Right_Ankle_Roll",
]


# 23-DOF Full Body. Uses booster_train-derived datasheet motor models
# (kp = I*(2*pi*f)², kd = 2*zeta*I*(2*pi*f); f=10 Hz, zeta=2).
ARTICULATION = EntityArticulationInfoCfg(
  actuators=WHOLEBODY_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)

# Legs-only locomotion. Uses hand-tuned gains matching booster_gym's
# velocity-tracking training (hip 200/5, knee 200/5, ankle 50/2).
LOCOMOTION_ARTICULATION = EntityArticulationInfoCfg(
  actuators=LOCOMOTION_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)

##
# Keyframe config
##

# Home keyframe joint positions (from XML keyframe)
HOME_QPOS: dict[str, float] = {
  # Head
  "AAHead_yaw": 0.0,
  "Head_pitch": 0.0,
  # Left arm (manufacturer's deployment values)
  "Left_Shoulder_Pitch": 0.2,
  "Left_Shoulder_Roll": -1.35,
  "Left_Elbow_Pitch": 0.0,
  "Left_Elbow_Yaw": -0.5,
  # Right arm (manufacturer's deployment values)
  "Right_Shoulder_Pitch": 0.2,
  "Right_Shoulder_Roll": 1.35,
  "Right_Elbow_Pitch": 0.0,
  "Right_Elbow_Yaw": 0.5,
  # Waist
  "Waist": 0.0,
  # Left leg
  "Left_Hip_Pitch": -0.24,
  "Left_Hip_Roll": 0.0,
  "Left_Hip_Yaw": 0.0,
  "Left_Knee_Pitch": 0.5,
  "Left_Ankle_Pitch": -0.3,
  "Left_Ankle_Roll": 0.0,
  # Right leg
  "Right_Hip_Pitch": -0.24,
  "Right_Hip_Roll": 0.0,
  "Right_Hip_Yaw": 0.0,
  "Right_Knee_Pitch": 0.5,
  "Right_Ankle_Pitch": -0.3,
  "Right_Ankle_Roll": 0.0,
}

# Locomotion-specific default pose (booster_gym convention).
# Hip_Pitch=-0.2, Knee=0.4, Ankle_Pitch=-0.25, base_height=0.72.
HOME_QPOS_LOCOMOTION: dict[str, float] = {
  **HOME_QPOS,
  "Left_Hip_Pitch": -0.2,
  "Left_Knee_Pitch": 0.4,
  "Left_Ankle_Pitch": -0.25,
  "Right_Hip_Pitch": -0.2,
  "Right_Knee_Pitch": 0.4,
  "Right_Ankle_Pitch": -0.25,
}

# Locomotion-specific base height (booster_gym convention: 0.72).
LOCOMOTION_BASE_HEIGHT = (0.0, 0.0, 0.72)

##
# Robot Configuration Functions
##

if _MJLAB_AVAILABLE:

  def get_robot_cfg(
    foot_self_collision: bool = False,
    self_collision: bool = False,
    full_collision: bool = False,
    with_head_camera: bool = False,
  ) -> EntityCfg:
    if self_collision:
      collision = FEET_FOREARM_WAIST_COLLISION
    elif foot_self_collision:
      collision = FEET_SELF_COLLISION
    elif full_collision:
      collision = FULL_COLLISION_WITHOUT_SELF
    else:
      collision = FEET_ONLY_COLLISION
    spec_fn = get_spec_with_head_camera if with_head_camera else get_spec
    return EntityCfg(
      init_state=EntityCfg.InitialStateCfg(
        pos=(0, 0, 0.66),
        joint_pos=HOME_QPOS,
        joint_vel={".*": 0.0},
      ),
      collisions=(collision,),
      spec_fn=spec_fn,
      articulation=ARTICULATION,
    )

  # Convenience shorthand
  ROBOT_CFG = get_robot_cfg()

  def get_locomotion_robot_cfg(
    foot_self_collision: bool = False,
    self_collision: bool = False,
    full_collision: bool = False,
    with_head_camera: bool = False,
  ) -> EntityCfg:
    """Locomotion-oriented robot config: hand-tuned PD gains, booster_gym
    default pose, 0.72 m base height.  Identical collision and spec options
    as ``get_robot_cfg``.
    """
    if self_collision:
      collision = FEET_FOREARM_WAIST_COLLISION
    elif foot_self_collision:
      collision = FEET_SELF_COLLISION
    elif full_collision:
      collision = FULL_COLLISION_WITHOUT_SELF
    else:
      collision = FEET_ONLY_COLLISION
    spec_fn = get_spec_with_head_camera if with_head_camera else get_spec
    return EntityCfg(
      init_state=EntityCfg.InitialStateCfg(
        pos=LOCOMOTION_BASE_HEIGHT,
        joint_pos=HOME_QPOS_LOCOMOTION,
        joint_vel={".*": 0.0},
      ),
      collisions=(collision,),
      spec_fn=spec_fn,
      articulation=LOCOMOTION_ARTICULATION,
    )

##
# Symmetry configuration (left-right mirror about sagittal plane)
##

# Joint name pairs for left-right mirroring. Central joints map to themselves.
SYMMETRY_JOINT_NAMES: dict[str, str] = {
  "AAHead_yaw": "AAHead_yaw",
  "Head_pitch": "Head_pitch",
  "Left_Shoulder_Pitch": "Right_Shoulder_Pitch",
  "Left_Shoulder_Roll": "Right_Shoulder_Roll",
  "Left_Elbow_Pitch": "Right_Elbow_Pitch",
  "Left_Elbow_Yaw": "Right_Elbow_Yaw",
  "Right_Shoulder_Pitch": "Left_Shoulder_Pitch",
  "Right_Shoulder_Roll": "Left_Shoulder_Roll",
  "Right_Elbow_Pitch": "Left_Elbow_Pitch",
  "Right_Elbow_Yaw": "Left_Elbow_Yaw",
  "Waist": "Waist",
  "Left_Hip_Pitch": "Right_Hip_Pitch",
  "Left_Hip_Roll": "Right_Hip_Roll",
  "Left_Hip_Yaw": "Right_Hip_Yaw",
  "Left_Knee_Pitch": "Right_Knee_Pitch",
  "Left_Ankle_Pitch": "Right_Ankle_Pitch",
  "Left_Ankle_Roll": "Right_Ankle_Roll",
  "Right_Hip_Pitch": "Left_Hip_Pitch",
  "Right_Hip_Roll": "Left_Hip_Roll",
  "Right_Hip_Yaw": "Left_Hip_Yaw",
  "Right_Knee_Pitch": "Left_Knee_Pitch",
  "Right_Ankle_Pitch": "Left_Ankle_Pitch",
  "Right_Ankle_Roll": "Left_Ankle_Roll",
}

# Joints whose sign flips under left-right mirroring (roll/yaw axes are pseudovectors).
# Matches holosoma's flip_sign_joint_names convention.
FLIP_SIGN_JOINT_NAMES: list[str] = [
  "AAHead_yaw",
  "Left_Shoulder_Roll",
  "Left_Elbow_Yaw",
  "Right_Shoulder_Roll",
  "Right_Elbow_Yaw",
  "Waist",
  "Left_Hip_Roll",
  "Left_Hip_Yaw",
  "Right_Hip_Roll",
  "Right_Hip_Yaw",
  "Left_Ankle_Roll",
  "Right_Ankle_Roll",
]

##
# Action scales: target = scale * action + default.
##

_UPPER_BODY_JOINTS = frozenset(
  [
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Waist",
  ]
)

# 0.25 for leg joints, 0.0 for upper body (held at default pose).
LOCOMOTION_ACTION_SCALE: dict[str, float] = {
  name: (0.0 if name in _UPPER_BODY_JOINTS else 0.25) for name in JOINT_NAMES
}

if _MJLAB_AVAILABLE:
  # booster_train recipe: kp * scale = 0.25 * effort, decoupled from kp.
  WHOLEBODY_ACTION_SCALE: dict[str, float] = {
    cfg.target_names_expr[0]: 0.25 * cfg.effort_limit / cfg.stiffness
    for cfg in WHOLEBODY_ACTUATORS
  }

##
# Foot geom names (for events like friction randomization)
##

# All foot geometry names. Each foot has the disabled mesh geom, 5 collision
# capsules (4 longitudinal sole + 1 inner-face vertical, for dribbling) and the
# booster_gym flat box (used by the locomotion configs).
FOOT_GEOM_NAMES = (
  "left_foot_link",
  "left_foot1_collision",
  "left_foot2_collision",
  "left_foot3_collision",
  "left_foot4_collision",
  "left_foot5_collision",
  "left_foot_box_collision",
  "right_foot_link",
  "right_foot1_collision",
  "right_foot2_collision",
  "right_foot3_collision",
  "right_foot4_collision",
  "right_foot5_collision",
  "right_foot_box_collision",
)

FOOT_SITE_NAMES = ("left_foot", "right_foot")

FOOT_BODY_NAMES = ("left_foot_link", "right_foot_link")

BASE_BODY_NAME = "Trunk"
