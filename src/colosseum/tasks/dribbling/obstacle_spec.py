"""Cylinder obstacle entity for adversarial dribbling training.

The obstacle mimics an opposing player: a fixed-base cylinder that the ball
and robot can collide with.  mjlab auto-wraps fixed-base entities in mocap
bodies, so the obstacle position is controlled each episode via
entity.write_mocap_pose_to_sim().
"""

import mujoco
from mjlab.entity import EntityCfg

# Physical dimensions -- roughly matching a human player.
OBSTACLE_HEIGHT: float = 1.2  # m
OBSTACLE_RADIUS: float = 0.15  # m

# Default number of obstacle instances used in scene and encoder configs.
NUM_OBSTACLES: int = 1

# Default local position used before the first episode reset.
# Placed far away so inactive obstacles do not interfere with the robot spawn.
_PARK_LOCAL_Y: float = 200.0


def get_obstacle_spec(
  height: float = OBSTACLE_HEIGHT,
  radius: float = OBSTACLE_RADIUS,
) -> mujoco.MjSpec:
  """Return a fixed-base cylinder MjSpec for one obstacle.

  The body is centered at (0, 0, height/2) so its base sits on the
  ground plane (z = 0).  No freejoint so mjlab treats this as fixed-base
  and automatically wraps the root body as a mocap body.
  """
  xml = f"""<mujoco>
  <worldbody>
    <body name="obstacle" pos="0 0 {height / 2:.4f}">
      <geom name="obstacle_collision"
            type="cylinder"
            size="{radius:.4f} {height / 2:.4f}"
            condim="3"
            friction="0.8 0.005 0.0001"
            rgba="0.85 0.25 0.1 0.9"/>
    </body>
  </worldbody>
</mujoco>"""
  return mujoco.MjSpec.from_string(xml)


def get_obstacle_cfg(index: int = 0) -> EntityCfg:
  """Return an EntityCfg for obstacle number index.

  Parked far from the env origin in the local y-direction so that inactive
  obstacles (before curriculum enables them) are physically out of reach.
  """
  return EntityCfg(
    spec_fn=get_obstacle_spec,
    init_state=EntityCfg.InitialStateCfg(
      # The obstacle body in the spec XML is already offset by height/2 in z,
      # so the mocap_base wrapper sits at z=0 (ground level).
      pos=(0.0, _PARK_LOCAL_Y * (index + 1), 0.0),
    ),
  )
