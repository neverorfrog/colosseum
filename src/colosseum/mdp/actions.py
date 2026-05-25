"""Custom action terms for colosseum."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm
from mjlab.utils.buffers.delay_buffer import DelayBuffer


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
  """JointPositionAction with stochastic per-env action delay.

  Each environment independently samples a delay from
  ``[min_delay_steps, max_delay_steps]`` at every policy step, simulating
  motor communication and processing latency.  The default range of 0–2
  policy steps corresponds to 0–40 ms at 50 Hz control.
  """

  min_delay_steps: int = 0
  max_delay_steps: int = 2

  def build(self, env: ManagerBasedRlEnv) -> ActionTerm:
    return DelayedJointPositionAction(self, env)


class DelayedJointPositionAction(JointPositionAction):
  """Joint position action backed by mjlab's DelayBuffer."""

  cfg: DelayedJointPositionActionCfg

  def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._delay_buffer = DelayBuffer(
      min_lag=cfg.min_delay_steps,
      max_lag=cfg.max_delay_steps,
      batch_size=self.num_envs,
      device=self.device,
      per_env=True,
    )
    # Holds the delayed target computed in process_actions, reused by each substep.
    self._delayed_actions = torch.zeros(
      self.num_envs, self._num_targets, device=self.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    super().reset(env_ids=env_ids)
    self._delay_buffer.reset(batch_ids=env_ids)

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    self._delay_buffer.append(self._processed_actions)
    self._delayed_actions = self._delay_buffer.compute()

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      self._delayed_actions - encoder_bias, joint_ids=self._target_ids
    )
