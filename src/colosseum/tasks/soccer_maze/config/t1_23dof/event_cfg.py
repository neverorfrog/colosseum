"""Event configuration for T1 soccer-maze task."""

from mjlab.envs.mdp.dr import body_com_offset, encoder_bias, geom_friction
from mjlab.envs.mdp.events import push_by_setting_velocity, reset_joints_by_offset
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, FOOT_GEOM_NAMES
from colosseum.tasks.maze.mdp.events import reset_wall_positions
from colosseum.tasks.soccer_maze.mdp.events import reset_robot_and_ball

events = {
  # Mocap walls must be positioned at startup and after every reset.
  "reset_wall_positions_startup": EventTermCfg(
    func=reset_wall_positions,
    mode="startup",
  ),
  "reset_wall_positions": EventTermCfg(
    func=reset_wall_positions,
    mode="reset",
  ),
  # Robot and ball are reset together so the ball position is computed from
  # the same sampled maze cell — no data-staleness between separate events.
  "reset_robot_and_ball": EventTermCfg(
    func=reset_robot_and_ball,
    mode="reset",
    params={
      "robot_z_offset": 0.665,
      "ball_x_offset": 0.4,
      "ball_z": 0.2,
      "yaw_range": (0.0, 0.0),
    },
  ),
  "reset_robot_joints": EventTermCfg(
    func=reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  ),
  # "push_robot": EventTermCfg(
  #   func=push_by_setting_velocity,
  #   mode="interval",
  #   interval_range_s=(1.0, 3.0),
  #   params={
  #     "velocity_range": {
  #       "x": (-0.5, 0.5),
  #       "y": (-0.5, 0.5),
  #       "z": (-0.4, 0.4),
  #       "roll": (-0.52, 0.52),
  #       "pitch": (-0.52, 0.52),
  #       "yaw": (-0.78, 0.78),
  #     },
  #   },
  # ),
  # "foot_friction": EventTermCfg(
  #   mode="startup",
  #   func=geom_friction,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
  #     "operation": "abs",
  #     "ranges": (0.3, 1.2),
  #     "shared_random": True,
  #   },
  # ),
  # "encoder_bias": EventTermCfg(
  #   mode="startup",
  #   func=encoder_bias,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot"),
  #     "bias_range": (-0.015, 0.015),
  #   },
  # ),
  # "base_com": EventTermCfg(
  #   mode="startup",
  #   func=body_com_offset,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME)),
  #     "operation": "add",
  #     "ranges": {
  #       0: (-0.025, 0.025),
  #       1: (-0.025, 0.025),
  #       2: (-0.03, 0.03),
  #     },
  #   },
  # ),
}
