"""kicking-residual termination functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def terminate_after_kick(
  env: ManagerBasedRlEnv,
  sensor_name: str = "foot_ball_contact",
  min_contact_force: float = 2.0,
  delay_steps: int = 100,
) -> torch.Tensor:
  """End the episode ``delay_steps`` control steps after the first foot-ball strike.

  One strike per episode: the first time a foot contacts the ball
  (force >= ``min_contact_force``) a countdown starts, and the episode terminates
  ``delay_steps`` control steps later. ``delay_steps`` must exceed the reward's
  ``kick_credit_steps`` so the full kick-credit window pays out (and the ball
  flight is observed) before the episode ends — at 50 Hz, 100 steps = 2 s.

  This removes the incentive behind the crouch-and-hug pathology: with no second
  tap to set up, re-contact stability and ball proximity stop paying, so the
  policy can only maximize the single strike.

  True termination (``time_out`` unset on the cfg): once the kick is delivered the
  episode is genuinely done, so the critic does not bootstrap a phantom future
  return past it — the kick is the terminal goal. Per-env; the strike record is
  advanced once per control step via the ``common_step_counter`` latch and reset
  each new episode (``episode_length_buf == 1`` is the first control step, since
  the buffer is never 0 at termination-compute time).
  """
  step_at = getattr(env, "_kick_term_step_at", None)
  if step_at is None or step_at.shape[0] != env.num_envs:
    # episode-step index of the first strike; -1 == no strike yet this episode.
    step_at = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    env._kick_term_step_at = step_at  # type: ignore[attr-defined]
    env._kick_term_latch = -1  # type: ignore[attr-defined]

  current_step = int(env.common_step_counter)
  if current_step != env._kick_term_latch:  # type: ignore[attr-defined]
    env._kick_term_latch = current_step  # type: ignore[attr-defined]

    # Clear the strike record at the start of each new episode.
    step_at[env.episode_length_buf == 1] = -1

    # Peak force over the step from per-substep force_history ([B, P, H, 3]); the
    # instantaneous `data.force` misses brief kick impulses (see _get_kick_gate).
    fh = env.scene[sensor_name].data.force_history
    force = fh.norm(dim=-1).amax(dim=(1, 2))  # [B] over feet & substeps
    strike_now = force >= min_contact_force
    first_strike = strike_now & (step_at < 0)
    step_at[first_strike] = env.episode_length_buf[first_strike]

  return (step_at >= 0) & (env.episode_length_buf - step_at >= delay_steps)
