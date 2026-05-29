import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  SceneEntityCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.nan_guard import NanGuardCfg
from mjlab.viewer import ViewerConfig

from colosseum.config.types.task import TaskConfig, register_task
from colosseum.envs.colosseum_env import ColosseumEnvCfg
from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, get_robot_cfg
from colosseum.robots.t1_23dof.sensors import (
  FEET_GROUND_CONTACT_SENSOR,
  FOOT_HEIGHT_SCAN,
  NONFOOT_GROUND_CONTACT_SENSOR,
  SELF_COLLISION_SENSOR,
)
from colosseum.tasks.dribbling.mdp.gait_phase_command import GaitPhaseCommandCfg
from colosseum.tasks.velocity.mdp.rma_term import VelocityRmaTermCfg
from colosseum.utils import project_root

from .algo_cfg import booster_t1_ppo_cfg, booster_t1_rsl_rl_runner_cfg
from .cat_cfg import actions, commands, terminations
from .curriculum_cfg import curriculum
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
    nconmax=45,
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


def booster_t1_velocity_env_cfg(play: bool = False) -> ColosseumEnvCfg:
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
    encoders={
      "env_params": VelocityRmaTermCfg(),
    },
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events = {}
    cfg.events["reset_robot_joints"] = EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    )

    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.curriculum = False  # play/eval: uniform-box sampling from ranges
    twist.heading_command = True
    twist.rel_forward_envs = 0.7
    twist.rel_world_envs = 0.0
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 1.0
    twist.ranges.heading = (-math.pi / 3, math.pi / 3)
    twist.heading_control_stiffness = 1.5

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


def get_t1_velocity_checkpoint(version: int) -> Path:
  """Return the .pt checkpoint path for the given version number."""
  version_dir = project_root() / "models" / "t1-velocity" / f"v{version}"
  pts = list(version_dir.glob("*.pt"))
  if not pts:
    raise FileNotFoundError(f"No .pt checkpoint found in {version_dir}")
  return pts[0]


def booster_t1_velocity_rough_env_cfg(play: bool = False) -> ColosseumEnvCfg:
  cfg = booster_t1_velocity_env_cfg(play=play)

  cfg.sim.nconmax = 120

  # FEET_ONLY_COLLISION leaves all non-foot geoms physically transparent. At tile
  # seam boundaries the 5mm foot spheres can momentarily lose contact with both
  # adjacent tile surfaces and tunnel through. FULL_COLLISION_WITHOUT_SELF gives
  # every named body geom a terrain-collidable surface, so the shin/knee catches
  # the robot even when a foot sphere slips through a seam.
  cfg.scene.entities["robot"] = get_robot_cfg(full_collision=True)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = copy.deepcopy(ROUGH_TERRAINS_CFG)

  if play:
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.border_width = 10.0
  else:
    cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.curriculum = {
      **cfg.curriculum,
      "terrain_levels": CurriculumTermCfg(
        func=terrain_levels_vel,
        params={"command_name": "twist"},
      ),
    }

  return cfg


@register_task("t1-velocity")
@dataclass(frozen=True)
class T1VelocityTask(TaskConfig):
  name: str = "t1-velocity"
  env: ColosseumEnvCfg = field(default_factory=booster_t1_velocity_env_cfg)

  @property
  def train_env_cfg(self):
    return self.env

  @property
  def play_env_cfg(self):
    return booster_t1_velocity_env_cfg(play=True)

  @property
  def algo_cfg(self):
    return booster_t1_ppo_cfg()

  @property
  def rl_cfg(self):
    return booster_t1_rsl_rl_runner_cfg()


@register_task("t1-velocity-rough")
@dataclass(frozen=True)
class T1VelocityRoughTask(TaskConfig):
  name: str = "t1-velocity-rough"
  env: ColosseumEnvCfg = field(default_factory=booster_t1_velocity_rough_env_cfg)

  @property
  def train_env_cfg(self):
    return self.env

  @property
  def play_env_cfg(self):
    return booster_t1_velocity_rough_env_cfg(play=True)

  @property
  def algo_cfg(self):
    return booster_t1_ppo_cfg()

  @property
  def rl_cfg(self):
    return booster_t1_rsl_rl_runner_cfg()
