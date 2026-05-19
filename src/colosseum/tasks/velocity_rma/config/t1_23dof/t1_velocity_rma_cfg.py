"""t1-velocity-rma task configuration.

Three-phase RMA training on top of the standard velocity locomotion task:

  Phase 1 (this config): PPO with PrivilegedEncoder reading DR physics params.
           Train until the base locomotion policy converges (~same as t1-velocity).

  Phase 2: load Phase 1 checkpoint, call build_adaptation_optimizer(), run
           ProprioWindowEncoder regression.

  Phase 3: load Phase 2 checkpoint, call build_phase3_optimizer(), fine-tune
           policy with frozen ProprioWindowEncoder predictions.
"""

import math
from dataclasses import dataclass, field

from colosseum.config.types.task import TaskConfig, register_task
from colosseum.envs.colosseum_env import ColosseumEnvCfg
from colosseum.tasks.velocity.config.t1_23dof.cat_cfg import actions, commands, terminations
from colosseum.tasks.velocity.config.t1_23dof.curriculum_cfg import curriculum
from colosseum.tasks.velocity.config.t1_23dof.event_cfg import events
from colosseum.tasks.velocity.config.t1_23dof.reward_cfg import rewards
from colosseum.tasks.velocity.config.t1_23dof.t1_velocity_cfg import (
  scene_cfg,
  sim_cfg,
  viewer_cfg,
)
from colosseum.tasks.velocity_rma.config.t1_23dof.algo_cfg import booster_t1_rma_ppo_cfg
from colosseum.tasks.velocity_rma.config.t1_23dof.observation_cfg import observations
from colosseum.tasks.velocity_rma.mdp.rma_term import VelocityRmaTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg


def booster_t1_velocity_rma_env_cfg(play: bool = False) -> ColosseumEnvCfg:
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
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.heading_command = True
    twist.rel_forward_envs = 1.0
    twist.rel_world_envs = 0.0
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 1.0
    twist.ranges.heading = (-math.pi / 3, math.pi / 3)
    twist.heading_control_stiffness = 0.5

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


@register_task("t1-velocity-rma")
@dataclass(frozen=True)
class T1VelocityRmaTask(TaskConfig):
  name: str = "t1-velocity-rma"
  env: ColosseumEnvCfg = field(default_factory=booster_t1_velocity_rma_env_cfg)

  @property
  def train_env_cfg(self):
    return self.env

  @property
  def play_env_cfg(self):
    return booster_t1_velocity_rma_env_cfg(play=True)

  @property
  def algo_cfg(self):
    return booster_t1_rma_ppo_cfg()
