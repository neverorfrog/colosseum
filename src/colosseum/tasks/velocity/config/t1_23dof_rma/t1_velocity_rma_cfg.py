from dataclasses import dataclass, field

from colosseum.config.types.task import TaskConfig, register_task
from colosseum.envs.colosseum_env import ColosseumEnvCfg
from colosseum.tasks.velocity.config.t1_23dof.t1_velocity_cfg import (
  booster_t1_velocity_env_cfg,
)
from colosseum.tasks.velocity.mdp.rma_term import VelocityRmaTermCfg

from .algo_cfg import booster_t1_ppo_cfg, booster_t1_rsl_rl_runner_cfg
from .observation_cfg import observations


def booster_t1_velocity_rma_env_cfg(play: bool = False) -> ColosseumEnvCfg:
  cfg = booster_t1_velocity_env_cfg(play=play)
  cfg.encoders["env_params"] = VelocityRmaTermCfg()
  cfg.observations = observations
  return cfg


@register_task("t1-velocity-rma")
@dataclass(frozen=True)
class T1VelocityTask(TaskConfig):
  name: str = "t1-velocity-rma"
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
