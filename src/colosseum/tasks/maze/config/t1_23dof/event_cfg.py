"""Event configuration for T1 maze task."""

from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.tasks.maze.mdp.events import reset_to_valid_maze_position

events = {
  "reset_agent_position": EventTermCfg(
    func=reset_to_valid_maze_position,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "z_offset": 0.665,  # T1 root (freejoint) height when feet are on ground
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
}
