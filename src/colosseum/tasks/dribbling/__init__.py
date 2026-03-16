"""Dribbling task registration."""

from dataclasses import dataclass

from colosseum.config.types.task import TaskConfig, register_task


def _make_train_cfg():
  from colosseum.tasks.dribbling.config.t1_23dof.env_cfgs import (
    booster_t1_dribbling_env_cfg,
  )
  return booster_t1_dribbling_env_cfg()


def _make_play_cfg():
  from colosseum.tasks.dribbling.config.t1_23dof.env_cfgs import (
    booster_t1_dribbling_env_cfg,
  )
  return booster_t1_dribbling_env_cfg(play=True)


def _make_rl_cfg():
  from colosseum.tasks.velocity.config.t1_23dof.rl_cfg import (
    booster_t1_ppo_runner_cfg,
  )
  cfg = booster_t1_ppo_runner_cfg()
  cfg.experiment_name = "t1_dribbling"
  return cfg


@register_task("t1-dribbling")
@dataclass(frozen=True)
class T1DribblingTask(TaskConfig):
  name: str = "t1-dribbling"

  @property
  def train_env_cfg(self):
    return _make_train_cfg()

  @property
  def play_env_cfg(self):
    return _make_play_cfg()

  @property
  def rl_cfg(self):
    return _make_rl_cfg()
