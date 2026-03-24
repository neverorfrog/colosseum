"""Velocity task registration for both mjlab and colosseum registries.

Training task classes require mjlab and are only registered when mjlab is available.
Deployment code lives in deploy/ subpackage and has no mjlab dependency.
"""

try:
  from dataclasses import dataclass, field

  from mjlab.envs import ManagerBasedRlEnvCfg

  from colosseum.config.types.task import TaskConfig, register_task

  def _make_train_cfg():
    from colosseum.tasks.velocity.config.t1_23dof.t1_velocity_cfg import (
      booster_t1_velocity_env_cfg,
    )

    return booster_t1_velocity_env_cfg()

  def _make_play_cfg():
    from colosseum.tasks.velocity.config.t1_23dof.t1_velocity_cfg import (
      booster_t1_velocity_env_cfg,
    )

    return booster_t1_velocity_env_cfg(True)

  def _make_algo_cfg():
    from colosseum.tasks.velocity.config.t1_23dof.algo_cfg import booster_t1_ppo_cfg
    return booster_t1_ppo_cfg()

  def _make_rl_cfg():
    from colosseum.tasks.velocity.config.t1_23dof.algo_cfg import booster_t1_rsl_rl_runner_cfg
    return booster_t1_rsl_rl_runner_cfg()

  @register_task("t1-velocity")
  @dataclass(frozen=True)
  class T1VelocityFlatTask(TaskConfig):
    name: str = "t1-velocity"
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

    @property
    def rl_cfg(self):
      return _make_rl_cfg()

except ImportError:
  pass
