"""Reset event: give the ball a random initial roll with a certain probability.

By default the ball spawns dead still, which makes the velocity channel of the
ball observation near-constant (``vx, vy ~= 0``) and uninformative. Rolling the
ball on a fraction of episodes forces the policy to actually read ball velocity
and to kick a moving ball, which is the common case in a real game.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def reset_ball_random_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  prob: float,
  speed_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
  """With probability ``prob`` per env, roll the ball at a random heading.

  Must run *after* ``reset_ball`` (which zeros the ball velocity): the un-rolled
  envs keep zero velocity and the rolled ones get a horizontal ``[vx, vy]`` at a
  uniformly random heading and a speed drawn from ``speed_range``.
  """
  env_ids = resolve_env_ids(env, env_ids)
  n = len(env_ids)
  device = env.device

  roll = torch.rand(n, device=device) < prob
  heading = torch.rand(n, device=device) * (2 * math.pi) - math.pi
  lo, hi = speed_range
  speed = torch.rand(n, device=device) * (hi - lo) + lo
  speed = torch.where(roll, speed, torch.zeros_like(speed))

  velocities = torch.zeros(n, 6, device=device)
  velocities[:, 0] = torch.cos(heading) * speed
  velocities[:, 1] = torch.sin(heading) * speed

  asset = env.scene[asset_cfg.name]
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)
