"""Velocity task registration for both mjlab and colosseum registries."""

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg

from colosseum.config.types.task import TaskConfig, register_task


def _make_rough_train_cfg():
    from colosseum.tasks.velocity.config.t1_23dof.env_cfgs import booster_t1_rough_env_cfg
    return booster_t1_rough_env_cfg()


def _make_rough_play_cfg():
    from colosseum.tasks.velocity.config.t1_23dof.env_cfgs import booster_t1_rough_env_cfg
    return booster_t1_rough_env_cfg(play=True)


def _make_flat_train_cfg():
    from colosseum.tasks.velocity.config.t1_23dof.env_cfgs import booster_t1_flat_env_cfg
    return booster_t1_flat_env_cfg()


def _make_flat_play_cfg():
    from colosseum.tasks.velocity.config.t1_23dof.env_cfgs import booster_t1_flat_env_cfg
    return booster_t1_flat_env_cfg(play=True)


def _make_rl_cfg():
    from colosseum.tasks.velocity.config.t1_23dof.rl_cfg import booster_t1_ppo_runner_cfg
    return booster_t1_ppo_runner_cfg()


@register_task("t1-velocity-rough")
@dataclass(frozen=True)
class T1VelocityRoughTask(TaskConfig):
    name: str = "t1-velocity-rough"
    env: ManagerBasedRlEnvCfg = field(default_factory=_make_rough_train_cfg)

    @property
    def train_env_cfg(self):
        return self.env

    @property
    def play_env_cfg(self):
        return _make_rough_play_cfg()

    @property
    def rl_cfg(self):
        return _make_rl_cfg()


@register_task("t1-velocity-flat")
@dataclass(frozen=True)
class T1VelocityFlatTask(TaskConfig):
    name: str = "t1-velocity-flat"
    env: ManagerBasedRlEnvCfg = field(default_factory=_make_flat_train_cfg)

    @property
    def train_env_cfg(self):
        return self.env

    @property
    def play_env_cfg(self):
        return _make_flat_play_cfg()

    @property
    def rl_cfg(self):
        return _make_rl_cfg()
