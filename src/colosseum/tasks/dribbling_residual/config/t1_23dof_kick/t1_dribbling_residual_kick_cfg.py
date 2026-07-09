"""Single-strong-kick variant of t1-dribbling-residual.

Same scene / observations / events / curriculum as the base task; only the
algorithm (per-joint orchestrator), rewards (kick impulse + relaxed pose) and
terminations (ball-kicked-away success) differ. The robot walks up to the ball,
strikes it once, and the episode ends when the struck ball leaves the radius.
"""

from dataclasses import dataclass, field

from colosseum.config.types.task import TaskConfig, register_task
from colosseum.envs.colosseum_env import ColosseumEnvCfg
from colosseum.robots.t1.sensors import FOOT5_BALL_CONTACT_SENSOR

from ..t1.t1_dribbling_residual_cfg import (
  booster_t1_dribbling_residual_env_cfg as _base_env_cfg,
)
from .algo_cfg import booster_t1_residual_ppo_cfg
from .cat_cfg import commands, terminations
from .event_cfg import events
from .reward_cfg import rewards


def booster_t1_dribbling_residual_kick_env_cfg(play: bool = False) -> ColosseumEnvCfg:
  cfg = _base_env_cfg(play)
  # Inner-face (foot5) ball contact sensor for the inside-foot kick rewards.
  cfg.scene.sensors = cfg.scene.sensors + (FOOT5_BALL_CONTACT_SENSOR,)
  cfg.rewards = rewards
  cfg.terminations = terminations
  cfg.events = events
  cfg.commands = commands
  return cfg


@register_task("t1-dribbling-residual-kick")
@dataclass(frozen=True)
class T1DribblingResidualKickTask(TaskConfig):
  name: str = "t1-dribbling-residual-kick"
  env: ColosseumEnvCfg = field(
    default_factory=booster_t1_dribbling_residual_kick_env_cfg
  )

  @property
  def train_env_cfg(self):
    return booster_t1_dribbling_residual_kick_env_cfg(play=False)

  @property
  def play_env_cfg(self):
    return booster_t1_dribbling_residual_kick_env_cfg(play=True)

  @property
  def algo_cfg(self):
    return booster_t1_residual_ppo_cfg()
