"""Dribbling task registration."""

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg

from colosseum.config.types.task import TaskConfig, register_task


def _make_train_cfg():
  from colosseum.tasks.dribbling.config.t1_23dof.t1_dribbling_cfg import (
    booster_t1_dribbling_env_cfg,
  )

  return booster_t1_dribbling_env_cfg()


def _make_play_cfg():
  from colosseum.tasks.dribbling.config.t1_23dof.t1_dribbling_cfg import (
    booster_t1_dribbling_env_cfg,
  )

  return booster_t1_dribbling_env_cfg(play=True)


def _make_algo_cfg():
  from colosseum.tasks.dribbling.config.t1_23dof.algo_cfg import booster_t1_dribbling_ppo_cfg
  return booster_t1_dribbling_ppo_cfg()


@register_task("t1-dribbling")
@dataclass(frozen=True)
class T1DribblingTask(TaskConfig):
  name: str = "t1-dribbling"
  env: ManagerBasedRlEnvCfg = field(default_factory=_make_train_cfg)

  @property
  def train_env_cfg(self):
    return self.env

  @property
  def play_env_cfg(self):
    return _make_play_cfg()

  @property
  def algo_cfg(self):
    return _make_algo_cfg()
