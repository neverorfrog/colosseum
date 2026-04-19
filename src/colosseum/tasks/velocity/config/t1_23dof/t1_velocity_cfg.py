from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from colosseum.config.types.task import TaskConfig, register_task
from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, get_robot_cfg
from colosseum.robots.t1_23dof.sensors import (
  FEET_GROUND_CONTACT_SENSOR,
  FOOT_HEIGHT_SCAN,
  NONFOOT_GROUND_CONTACT_SENSOR,
  SELF_COLLISION_SENSOR,
)

from .algo_cfg import booster_t1_ppo_cfg
from .cact_cfg import actions, commands, curriculum, terminations
from .event_cfg import events
from .observation_cfg import observations
from .reward_cfg import rewards


def scene_cfg(play: bool = False) -> SceneCfg:
  return SceneCfg(
    terrain=TerrainEntityCfg(),
    entities={"robot": get_robot_cfg()},
    sensors=(
      FEET_GROUND_CONTACT_SENSOR,
      FOOT_HEIGHT_SCAN,
      NONFOOT_GROUND_CONTACT_SENSOR,
      SELF_COLLISION_SENSOR,
    ),
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
    nconmax=45,
    njmax=1500,
    contact_sensor_maxmatch=500,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
      ccd_iterations=500,
    ),
  )


def booster_t1_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 dribbling task configuration.

  Starts from the flat velocity config and:
  - Adds a soccer ball as a free-floating scene entity.
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


@register_task("t1-velocity")
@dataclass(frozen=True)
class T1VelocityTask(TaskConfig):
  name: str = "t1-velocity"
  env: ManagerBasedRlEnvCfg = field(default_factory=booster_t1_velocity_env_cfg)

  @property
  def train_env_cfg(self):
    return self.env

  @property
  def play_env_cfg(self):
    return booster_t1_velocity_env_cfg(play=True)

  @property
  def algo_cfg(self):
    return booster_t1_ppo_cfg()
