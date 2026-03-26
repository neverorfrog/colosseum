"""Booster T1 dribbling environment configurations.

The dribbling task builds on the flat velocity task but replaces random
velocity commands with a BallApproachCommand that points the robot toward
a soccer ball.  The ball is a free-floating entity added to the scene and
is respawned at a random position (1–3 m from the robot) on every episode.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from colosseum.assets.ball.ball_spec import get_ball_cfg
from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, get_robot_cfg
from colosseum.robots.t1_23dof.sensors import (
  FEET_GROUND_CONTACT_SENSOR,
  FOOT_BALL_CONTACT_SENSOR,
  NONFOOT_GROUND_CONTACT_SENSOR,
  SELF_COLLISION_SENSOR,
)

from .cact_cfg import actions, commands, curriculum, terminations
from .event_cfg import events
from .observation_cfg import observations
from .reward_cfg import rewards


def scene_cfg(play: bool = False) -> SceneCfg:
  return SceneCfg(
    terrain=TerrainEntityCfg(),
    sensors=(
      FEET_GROUND_CONTACT_SENSOR,
      FOOT_BALL_CONTACT_SENSOR,
      NONFOOT_GROUND_CONTACT_SENSOR,
      SELF_COLLISION_SENSOR,
    ),
    entities={"ball": get_ball_cfg(), "robot": get_robot_cfg()},
    num_envs=1,
    extent=10.0,
  )


def viewer_cfg() -> ViewerConfig:
  return ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name=BASE_BODY_NAME,
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
  )


def sim_cfg() -> SimulationCfg:
  return SimulationCfg(
    nconmax=80,
    njmax=1500,
    contact_sensor_maxmatch=500,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
      ccd_iterations=50,
    ),
  )


def booster_t1_dribbling_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 dribbling task configuration.

  Starts from the flat velocity config and:
  - Adds a soccer ball as a free-floating scene entity.
  - Replaces the random velocity command with BallApproachCommand.
  - Adds a reset event that respawns the ball at a random position each episode.
  """
  cfg = ManagerBasedRlEnvCfg(
    scene=scene_cfg(play),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics={},
    viewer=viewer_cfg(),
    sim=sim_cfg(),
    decimation=4,
    episode_length_s=20.0,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg
