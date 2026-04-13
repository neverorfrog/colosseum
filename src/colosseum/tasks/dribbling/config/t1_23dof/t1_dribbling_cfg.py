"""Booster T1 dribbling environment configurations."""

from dataclasses import dataclass, field, replace
from functools import partial

from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from colosseum.assets.ball.ball_spec import get_ball_cfg
from colosseum.config.types.task import TaskConfig, register_task
from colosseum.envs.rma_based_env import RmaBasedEnvCfg
from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, get_robot_cfg
from colosseum.robots.t1_23dof.sensors import (
  FEET_GROUND_CONTACT_SENSOR,
  FOOT_BALL_CONTACT_SENSOR,
  FOOT_FOOT_CONTACT_SENSOR,
  FOOT_HEIGHT_SCAN,
  HEAD_DEPTH_SENSOR_TRAIN,
  HEAD_RGBD_SENSOR,
  NONFOOT_BALL_CONTACT_SENSOR,
  NONFOOT_GROUND_CONTACT_SENSOR,
  SELF_COLLISION_SENSOR,
)
from colosseum.tasks.dribbling.mdp.rma_terms import BallRmaTermCfg
from colosseum.tasks.dribbling.viz import DribblingViz

from .algo_cfg import booster_t1_dribbling_ppo_cfg
from .cact_cfg import actions, commands, curriculum, terminations
from .event_cfg import events
from .observation_cfg import observations
from .reward_cfg import rewards


def scene_cfg(play: bool = False, use_depth_camera: bool = False) -> SceneCfg:
  sensors = [
    FEET_GROUND_CONTACT_SENSOR,
    FOOT_HEIGHT_SCAN,
    FOOT_BALL_CONTACT_SENSOR,
    FOOT_FOOT_CONTACT_SENSOR,
    NONFOOT_BALL_CONTACT_SENSOR,
    NONFOOT_GROUND_CONTACT_SENSOR,
    SELF_COLLISION_SENSOR,
  ]
  if use_depth_camera:
    # Play mode uses full-res RGBD for visualisation; training uses the
    # low-res depth-only sensor to keep GPU memory manageable at scale.
    sensors.append(HEAD_RGBD_SENSOR if play else HEAD_DEPTH_SENSOR_TRAIN)
  return SceneCfg(
    terrain=TerrainEntityCfg(),
    sensors=tuple(sensors),
    entities={
      "ball": get_ball_cfg(),
      "robot": get_robot_cfg(foot_self_collision=True, with_head_camera=True),
    },
    num_envs=1,
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


def booster_t1_dribbling_env_cfg(
  play: bool = False, use_depth_camera: bool = False, show_depth: bool = False
) -> RmaBasedEnvCfg:
  cfg = RmaBasedEnvCfg(
    scene=scene_cfg(play, use_depth_camera=use_depth_camera),
    use_depth_camera=use_depth_camera,
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
    encoders={
      "ball": BallRmaTermCfg(
        privileged_obs_group="privileged_ball",
        adaptation_obs_group="depth_frames" if use_depth_camera else None,
        latent_dim=64,
      ),
    },
    viz_callbacks=[("camera_ball", partial(DribblingViz, show_depth=show_depth))],
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


@register_task("t1-dribbling")
@dataclass(frozen=True)
class T1DribblingTask(TaskConfig):
  name: str = "t1-dribbling"
  env: RmaBasedEnvCfg = field(default_factory=booster_t1_dribbling_env_cfg)
  use_depth_camera: bool = False
  show_depth: bool = False
  """Show a cv2 filmstrip of the encoder's depth buffer during play (--show-depth)."""

  @property
  def train_env_cfg(self):
    cfg = booster_t1_dribbling_env_cfg(use_depth_camera=self.use_depth_camera)
    return replace(cfg, scene=replace(cfg.scene, num_envs=self.env.scene.num_envs))

  @property
  def play_env_cfg(self):
    return booster_t1_dribbling_env_cfg(
      play=True, use_depth_camera=True, show_depth=self.show_depth
    )

  @property
  def algo_cfg(self):
    return booster_t1_dribbling_ppo_cfg()
