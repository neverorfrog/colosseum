"""Velocity task driven by the manufacturer actuator set.

Identical to ``t1-velocity`` except the robot uses ``get_robot_cfg`` — the
manufacturer datasheet motors on mjlab's DcMotorActuator (peak/rated torque +
peak-speed torque-speed curve) — instead of the hand-tuned
``get_locomotion_robot_cfg``. The joint-position action scale switches to the
manufacturer recipe accordingly (0.25 * peak torque / kp per joint). Keeping the
two side by side avoids swapping actuators in and out of the base task whenever
the other set is tested.
"""

import copy
from dataclasses import dataclass, field

from colosseum.config.types.task import TaskConfig, register_task
from colosseum.envs.colosseum_env import ColosseumEnvCfg
from colosseum.robots.t1_23dof.constants import (
  MANUFACTURER_ACTION_SCALE,
  get_robot_cfg,
)
from colosseum.tasks.velocity.config.t1_23dof.cat_cfg import (
  JOINT_NAMES,
)
from colosseum.tasks.velocity.config.t1_23dof.t1_velocity_cfg import (
  booster_t1_velocity_env_cfg,
)

from .algo_cfg import (
  booster_t1_ppo_cfg,
  booster_t1_rsl_rl_runner_cfg,
)
from .cat_cfg import actions
from .reward_cfg import rewards

UNACTUATED_JOINTS = {
  "AAHead_yaw",
  "Head_pitch",
}


def booster_t1_velocity_manu_env_cfg(play: bool = False) -> ColosseumEnvCfg:
  cfg = booster_t1_velocity_env_cfg(play=play)
  cfg.scene.entities["robot"] = get_robot_cfg()
  cfg.actions = actions
  cfg.rewards = rewards
  return cfg


@register_task("t1-velocity-manu")
@dataclass(frozen=True)
class T1VelocityManuTask(TaskConfig):
  name: str = "t1-velocity-manu"
  env: ColosseumEnvCfg = field(default_factory=booster_t1_velocity_manu_env_cfg)

  @property
  def train_env_cfg(self):
    return self.env

  @property
  def play_env_cfg(self):
    return booster_t1_velocity_manu_env_cfg(play=True)

  @property
  def algo_cfg(self):
    return booster_t1_ppo_cfg()

  @property
  def rl_cfg(self):
    return booster_t1_rsl_rl_runner_cfg()
