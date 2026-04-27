"""Booster T1 dribbling environment configurations."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import partial

from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from colosseum.assets.ball.ball_spec import get_ball_cfg
from colosseum.config.types.task import TaskConfig, register_task
from colosseum.envs.colosseum_env import ColosseumEnvCfg
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
from colosseum.tasks.dribbling.mdp.obstacle_commands import ObstacleCommandCfg
from colosseum.research.dribbling.rma_terms import DribblingRmaTermCfg
from colosseum.tasks.dribbling.obstacle_spec import NUM_OBSTACLES, get_obstacle_cfg
from colosseum.tasks.dribbling.viz import DribblingViz

from .algo_cfg import (
  booster_t1_dribbling_dagger_ppo_cfg,
  booster_t1_dribbling_ppo_cfg,
)
from .cact_cfg import actions, commands, curriculum, terminations
from .constraint_cfg import dribbling_constraints
from .event_cfg import events
from .observation_cfg import observations
from .reward_cfg import rewards


def _get_obstacle_stage_cfg(stage_index: int) -> dict:
  stages = curriculum["obstacle"].params["stages"]
  if stage_index < 0 or stage_index >= len(stages):
    raise ValueError(
      f"Invalid obstacle_stage_index={stage_index}. "
      f"Valid values are -1 or 0..{len(stages) - 1}."
    )
  return deepcopy(stages[stage_index])


def _build_obstacle_command_from_stage(
  stage_index: int,
  num_obstacles: int,
) -> ObstacleCommandCfg:
  stage = _get_obstacle_stage_cfg(stage_index)
  return ObstacleCommandCfg(
    num_obstacles=num_obstacles,
    num_active=stage.get("num_active", 0),
    behavior=stage.get("behavior", "none"),
    distance_range=stage.get("distance_range", (1.5, 3.0)),
    lateral_offset_range=stage.get("lateral_offset_range", (-0.8, 0.8)),
    min_speed=stage.get("min_speed", 0.0),
    max_speed=stage.get("max_speed", 0.0),
    velocity_resample_time_range=stage.get("velocity_resample_time_range", (0.5, 1.0)),
  )


def scene_cfg(
  play: bool = False,
  use_depth_camera: bool = False,
  num_obstacles: int = NUM_OBSTACLES,
) -> SceneCfg:
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

  obstacle_entities = {
    f"obstacle_{k}": get_obstacle_cfg(k) for k in range(num_obstacles)
  }

  return SceneCfg(
    terrain=TerrainEntityCfg(),
    sensors=tuple(sensors),
    env_spacing=10.0,
    entities={
      "ball": get_ball_cfg(),
      "robot": get_robot_cfg(foot_self_collision=True, with_head_camera=True),
      **obstacle_entities,
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
  play: bool = False,
  use_depth_camera: bool = False,
  show_depth: bool = False,
  num_obstacles: int = NUM_OBSTACLES,
  obstacle_stage_index: int = -1,
  ball_spawn_x_range: tuple[float, float] | None = None,
) -> ColosseumEnvCfg:
  cfg = ColosseumEnvCfg(
    scene=scene_cfg(
      play, use_depth_camera=use_depth_camera, num_obstacles=num_obstacles
    ),
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
    constraints=dribbling_constraints,
    encoders={
      "dribbling": DribblingRmaTermCfg(
        privileged_obs_group="privileged_ball",
        obstacle_privileged_obs_group="privileged_obstacles",
        adaptation_obs_group="depth_frames" if use_depth_camera else None,
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

  if ball_spawn_x_range is not None:
    cfg.events = dict(cfg.events)
    reset_ball = deepcopy(cfg.events["reset_ball"])
    reset_ball.params["pose_range"] = dict(reset_ball.params["pose_range"])
    reset_ball.params["pose_range"]["x"] = ball_spawn_x_range
    cfg.events["reset_ball"] = reset_ball

  if obstacle_stage_index >= 0:
    cfg.commands = dict(cfg.commands)
    cfg.commands["adversary"] = _build_obstacle_command_from_stage(
      obstacle_stage_index, num_obstacles
    )
    cfg.curriculum = dict(cfg.curriculum)
    cfg.curriculum.pop("obstacle", None)
  elif play:
    # In play mode, -1 means keep obstacle progression enabled rather than
    # forcing a fixed final setup. Keep only the obstacle curriculum term.
    cfg.curriculum = {"obstacle": deepcopy(curriculum["obstacle"])}

  return cfg


@register_task("t1-dribbling")
@dataclass(frozen=True)
class T1DribblingTask(TaskConfig):
  name: str = "t1-dribbling"
  env: ColosseumEnvCfg = field(default_factory=booster_t1_dribbling_env_cfg)
  use_depth_camera: bool = False
  show_depth: bool = False
  obstacle_stage_index: int = -1
  ball_spawn_x_range: tuple[float, float] | None = None
  use_dagger: bool = False
  teacher_checkpoint: str = ""
  """Show a cv2 filmstrip of the encoder's depth buffer during play (--show-depth)."""

  @property
  def train_env_cfg(self):
    cfg = booster_t1_dribbling_env_cfg(
      use_depth_camera=self.use_depth_camera,
      obstacle_stage_index=self.obstacle_stage_index,
      ball_spawn_x_range=self.ball_spawn_x_range,
    )
    return replace(cfg, scene=replace(cfg.scene, num_envs=self.env.scene.num_envs))

  @property
  def play_env_cfg(self):
    return booster_t1_dribbling_env_cfg(
      play=True,
      use_depth_camera=True,
      show_depth=self.show_depth,
      obstacle_stage_index=self.obstacle_stage_index,
    )

  @property
  def algo_cfg(self):
    if self.use_dagger:
      if not self.teacher_checkpoint:
        raise ValueError("teacher_checkpoint must be set when use_dagger=True.")
      return booster_t1_dribbling_dagger_ppo_cfg(
        teacher_checkpoint=self.teacher_checkpoint
      )
    return booster_t1_dribbling_ppo_cfg()
