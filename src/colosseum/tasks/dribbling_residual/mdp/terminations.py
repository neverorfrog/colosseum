"""Ball-lost termination for the residual dribbling task.

The dribble target is a fixed direction proxy for the whole episode; the policy
is rewarded for dribbling cleanly toward it, never for arriving. The genuine
failure mode is losing the ball, so the episode ends when the ball stays outside
the robot's field of view for longer than ``loss_timeout`` seconds.

"In view" uses the same head-camera reach as the perception model and head IK
(``HEAD_FOV_HALF`` / ``HEAD_MAX_RANGE``): the head swivels to keep the ball
centred, so the ball is visible whenever its body-frame bearing is within the
head's yaw reach and it is within range. The counter is continuous -- any
sighting resets it to zero -- so brief occlusions don't end the episode.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.manager_base import ManagerTermBase

from colosseum.tasks.dribbling_residual.mdp.head_ik_action import (
  HEAD_FOV_HALF,
  HEAD_MAX_RANGE,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg


class BallLostTermination(ManagerTermBase):
  """Terminate when the ball is out of view for more than ``loss_timeout`` s."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    p = cfg.params
    self._loss_timeout = float(p.get("loss_timeout", 3.0))
    self._fov_half = float(p.get("fov_half_angle", HEAD_FOV_HALF))
    self._max_range = float(p.get("max_range", HEAD_MAX_RANGE))
    self._dt = env.step_dt
    self._unseen_time = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict:
    if env_ids is None:
      self._unseen_time.zero_()
    else:
      self._unseen_time[env_ids] = 0.0
    return {}

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del kwargs
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    rel = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
    rng = rel.norm(dim=-1)

    q = robot.data.root_link_quat_w
    yaw = torch.atan2(
      2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
      1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
    )
    bearing = torch.atan2(rel[:, 1], rel[:, 0]) - yaw
    bearing = (bearing + math.pi) % (2 * math.pi) - math.pi

    visible = (bearing.abs() < self._fov_half) & (rng < self._max_range)
    self._unseen_time = torch.where(
      visible, torch.zeros_like(self._unseen_time), self._unseen_time + self._dt
    )
    return self._unseen_time > self._loss_timeout


def ball_lost_penalty(
  env: ManagerBasedRlEnv, term_name: str = "ball_lost"
) -> torch.Tensor:
  """Discrete penalty fired on the step the ball-lost termination triggers.

  Reads the termination done buffer (terminations are computed before rewards),
  so it spikes exactly once per lost episode. Use with a negative weight.
  """
  return env.termination_manager.get_term(term_name).float()


class BallKickedAway(ManagerTermBase):
  """Terminate (success) a settle window after the ball is first struck.

  For the single-kick variant: the robot walks up and hits the ball once. The
  kick is *confirmed* ``credit_steps`` steps after the first foot-ball contact
  (long enough for the struck ball to leave the foot so the velocity reward and
  the success bonus register it, but far shorter than a swing-foot return, so
  the policy cannot take a second touch). The episode then keeps running for
  ``settle_steps`` more steps before terminating: the ball rewards gate off
  once the ball is away, so during the settle window the locomotion rewards
  dominate and the policy must recover from the kick follow-through into a
  stable gait -- an unstable landing costs it the whole window's reward.

  ``kick_confirmed`` is a one-shot flag raised on the confirmation step; the
  ``ball_kicked_away_bonus`` reward reads it off this instance.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    p = cfg.params
    self._credit_steps = int(p.get("credit_steps", 8))
    self._settle_steps = int(p.get("settle_steps", 0))
    self._sensor_name = str(p.get("sensor_name", "foot_ball_contact"))
    self._armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._steps_since = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.kick_confirmed = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict:
    if env_ids is None:
      self._armed.zero_()
      self._steps_since.zero_()
      self.kick_confirmed.zero_()
    else:
      self._armed[env_ids] = False
      self._steps_since[env_ids] = 0
      self.kick_confirmed[env_ids] = False
    return {}

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del kwargs
    found = env.scene[self._sensor_name].data.found
    if found is not None:
      contact = (found.flatten(start_dim=1) > 0).any(dim=-1)
      self._armed |= contact

    self._steps_since += self._armed.long()
    self.kick_confirmed = self._armed & (self._steps_since == self._credit_steps)
    return self._armed & (
      self._steps_since >= self._credit_steps + self._settle_steps
    )


def ball_kicked_away_bonus(
  env: ManagerBasedRlEnv, term_name: str = "ball_kicked_away"
) -> torch.Tensor:
  """Discrete success reward fired the step the kick is confirmed.

  Reads the one-shot ``kick_confirmed`` flag off the ``BallKickedAway`` term
  instance (terminations are computed before rewards), NOT the done buffer:
  the termination itself fires ``settle_steps`` later, after the recovery
  window. Use with a positive weight.
  """
  return env.termination_manager.get_term_cfg(term_name).func.kick_confirmed.float()
