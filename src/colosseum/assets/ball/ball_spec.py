"""Ball entity specification for dribbling tasks.

Physical properties match a FIFA size-5 soccer ball, taken from the
circus FieldGenerator implementation.
"""

from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg

BALL_RADIUS = 0.11  # m (22 cm diameter, size-5)
BALL_MASS = 0.425  # kg
BALL_FRICTION = 0.8  # dimensionless

_ASSETS_DIR = Path(__file__).parent


def get_ball_spec() -> mujoco.MjSpec:
  """Return MjSpec describing a free-floating soccer ball."""
  tex_path = str(_ASSETS_DIR / "ball.png")
  mesh_path = str(_ASSETS_DIR / "ball.obj")
  inertia = (2.0 / 3.0) * BALL_MASS * BALL_RADIUS**2

  xml = f"""<mujoco>
  <asset>
    <texture name="ball_diffuse" type="2d" file="{tex_path}"/>
    <material name="ball_mat" texture="ball_diffuse" texrepeat="1 1"
              specular="0.3" shininess="0.5" reflectance="0.05" rgba="0.5 0.5 0.5 1"/>
    <mesh name="ball_mesh" file="{mesh_path}"/>
  </asset>
  <worldbody>
    <body name="ball" pos="0 0 {BALL_RADIUS}">
      <freejoint/>
      <inertial mass="{BALL_MASS}"
                diaginertia="{inertia} {inertia} {inertia}"
                pos="0 0 0"/>
      <geom name="ball_collision" type="sphere" size="{BALL_RADIUS}"
            condim="6" friction="{BALL_FRICTION} 0.005 0.02" rgba="0 0 0 0"/>
      <geom name="ball_visual" type="mesh" mesh="ball_mesh" material="ball_mat"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
  return mujoco.MjSpec.from_string(xml)


def get_ball_cfg() -> EntityCfg:
  """Return a fresh EntityCfg for the soccer ball."""
  return EntityCfg(
    spec_fn=get_ball_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, BALL_RADIUS),
      joint_pos={},  # freejoint only — no articulated joints
    ),
  )
