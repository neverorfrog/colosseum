"""Booster T1 maze navigation environment configuration."""

from dataclasses import dataclass, field

from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from colosseum.config.types.task import TaskConfig, register_task
from colosseum.envs.abstraction_based_env import AbstractionBasedEnvCfg
from colosseum.managers.abstraction_manager import AbstractionTermCfg
from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, get_robot_cfg
from colosseum.robots.t1_23dof.sensors import (
  FEET_GROUND_CONTACT_SENSOR,
  SELF_COLLISION_SENSOR,
  WALL_COLLISION_SENSOR,
)
from colosseum.tasks.maze.maps import MAPS
from colosseum.tasks.maze.maze import Maze, MazeCfg
from colosseum.tasks.maze.mdp.grid_abstraction import GridAbstractionTermCfg
from colosseum.tasks.maze.terrain import MazeTerrainEntityCfg

from .algo_cfg import t1_maze_ppo_cfg
from .cact_cfg import actions, commands, curriculum, terminations
from .event_cfg import events
from .observation_cfg import observations
from .reward_cfg import rewards


def sim_cfg() -> SimulationCfg:
  return SimulationCfg(
    nconmax=500,
    njmax=1500,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
    ),
  )


def viewer_cfg() -> ViewerConfig:
  return ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name=BASE_BODY_NAME,
    distance=20.0,
    elevation=-10.0,
    azimuth=90.0,
  )


def scene_cfg(maze: Maze, num_envs: int) -> SceneCfg:
  return SceneCfg(
    num_envs=num_envs,
    env_spacing=maze.cell_size * len(maze.maze_map) * 5.0,
    entities={"robot": get_robot_cfg()},
    terrain=MazeTerrainEntityCfg(
      maze_cfg=maze.cfg,
    ),
    sensors=(FEET_GROUND_CONTACT_SENSOR, WALL_COLLISION_SENSOR, SELF_COLLISION_SENSOR),
    extent=2.0,
  )


def abstractions_cfg(
  maze: Maze, resolution_factor: int = 2
) -> dict[str, AbstractionTermCfg]:
  grid_frame = maze.build_upsampled_grid_frame(resolution_factor)
  obstacle_mask = maze.build_obstacle_mask(resolution_factor)
  return {
    "grid": GridAbstractionTermCfg(
      grid_frame=grid_frame,
      obstacle_mask=obstacle_mask,
      direction_method="gradient",
    ),
  }


def t1_maze_env_cfg(
  scenario: str = "umaze",
  num_envs: int = 64,
  resolution_factor: int = 2,
  play: bool = False,
) -> AbstractionBasedEnvCfg:
  """Create Booster T1 maze navigation task configuration.

  Args:
      scenario: Maze layout key from MAPS (e.g. "umaze", "small", "medium").
      num_envs: Number of parallel environments.
      resolution_factor: Upsampling factor for the grid abstraction.
      play: If True, configures for single-env visualization.
  """
  maze = Maze(MazeCfg(maze_map=MAPS[scenario], cell_size=5.0, wall_height=2.0, wall_size_factor=1.0))

  cfg = AbstractionBasedEnvCfg(
    scene=scene_cfg(maze, num_envs=1 if play else num_envs),
    sim=sim_cfg(),
    viewer=viewer_cfg(),
    actions=actions,
    observations=observations,
    rewards=rewards,
    commands=commands,
    terminations=terminations,
    events=events,
    curriculum={} if play else curriculum,
    abstractions=abstractions_cfg(maze, resolution_factor),
    metrics={},
    decimation=4,
    episode_length_s=90.0,
  )

  if play:
    cfg.observations["actor"].enable_corruption = False

  return cfg


@register_task("t1-maze")
@dataclass(frozen=True)
class T1MazeTask(TaskConfig):
  name: str = "t1-maze"
  env: AbstractionBasedEnvCfg = field(default_factory=t1_maze_env_cfg)

  @property
  def train_env_cfg(self):
    return self.env

  @property
  def play_env_cfg(self):
    return t1_maze_env_cfg(play=True)

  @property
  def algo_cfg(self):
    return t1_maze_ppo_cfg()
