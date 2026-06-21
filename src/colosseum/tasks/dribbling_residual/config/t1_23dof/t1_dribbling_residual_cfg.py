"""Residual dribbling task: frozen walk branch + trainable residual branch.

The scene carries the T1 plus a ball. Two command channels drive the two skills:
``twist`` (BallTwistCommand) steers the frozen walk toward the ball, and
``ball_vel`` (BallVelocityCommand) gives the residual a dribble setpoint. The
observation manager packs four groups (``loco_actor``, ``dribble_actor``,
``orchestrator``, ``critic``) consumed by ResidualPPO; see observation_cfg.py.

The step-0 validation (frozen v4 walk steered by the ball-as-twist trick) can
still be run with the velocity checkpoint:

    pixi run play task:t1-dribbling-residual \
      --checkpoint models/t1-velocity/v4/t1-velocity_ppo_v4.pt --debug-vel
"""

from dataclasses import dataclass, field

from mjlab.envs.mdp.events import (
  reset_root_state_uniform,
)
from mjlab.managers import (
  EventTermCfg,
  SceneEntityCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.nan_guard import NanGuardCfg
from mjlab.viewer import ViewerConfig

from colosseum.assets.ball.ball_spec import get_ball_cfg
from colosseum.config.types.task import TaskConfig, register_task
from colosseum.envs.colosseum_env import ColosseumEnvCfg
from colosseum.robots.t1_23dof.constants import (
  BASE_BODY_NAME,
  get_robot_cfg,
)
from colosseum.robots.t1_23dof.sensors import (
  FEET_GROUND_CONTACT_SENSOR,
  FOOT_BALL_CONTACT_SENSOR,
  FOOT_HEIGHT_SCAN,
  NONFOOT_GROUND_CONTACT_SENSOR,
  SELF_COLLISION_SENSOR,
)
from colosseum.tasks.dribbling.mdp.gait_phase_command import GaitPhaseCommandCfg
from colosseum.tasks.dribbling_residual.mdp.ball_velocity_command import (
  BallVelocityCommandCfg,
)

from .algo_cfg import booster_t1_residual_ppo_cfg
from .cat_cfg import actions, commands, terminations
from .curriculum_cfg import curriculum
from .event_cfg import events
from .observation_cfg import observations
from .reward_cfg import rewards


def scene_cfg(play: bool = False) -> SceneCfg:
  return SceneCfg(
    terrain=TerrainEntityCfg(),
    entities={
      "robot": get_robot_cfg(with_head_camera=True, dribbling=True),
      "ball": get_ball_cfg(),
    },
    sensors=(
      FEET_GROUND_CONTACT_SENSOR,
      FOOT_BALL_CONTACT_SENSOR,
      FOOT_HEIGHT_SCAN,
      NONFOOT_GROUND_CONTACT_SENSOR,
      SELF_COLLISION_SENSOR,
    ),
    num_envs=4096,
    extent=10.0,
    env_spacing=10.0,
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
      ccd_iterations=500,
    ),
    nan_guard=NanGuardCfg(enabled=True),
  )


def booster_t1_dribbling_residual_env_cfg(play: bool = False) -> ColosseumEnvCfg:
  cfg = ColosseumEnvCfg(
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
    episode_length_s=30.0,
    encoders={},
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["loco_actor"].enable_corruption = False
    cfg.observations["dribble_actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events = {}
    cfg.events["reset_base"] = EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.0, 0.0),
          "y": (-0.0, 0.0),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    )

    cfg.commands["ball_vel"] = BallVelocityCommandCfg(
      speed_range=(0.1, 0.5),
    )

    cfg.events["reset_ball"] = EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("ball"),
        "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "z": (0.11, 0.11)},
        "velocity_range": {},
      },
    )

    gait = cfg.commands["gait_phase"]
    assert isinstance(gait, GaitPhaseCommandCfg)
    gait.randomize_phase = False

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


@register_task("t1-dribbling-residual")
@dataclass(frozen=True)
class T1DribblingResidualTask(TaskConfig):
  name: str = "t1-dribbling-residual"
  env: ColosseumEnvCfg = field(default_factory=booster_t1_dribbling_residual_env_cfg)

  @property
  def train_env_cfg(self):
    return booster_t1_dribbling_residual_env_cfg(play=False)

  @property
  def play_env_cfg(self):
    return booster_t1_dribbling_residual_env_cfg(play=True)

  @property
  def algo_cfg(self):
    return booster_t1_residual_ppo_cfg()
