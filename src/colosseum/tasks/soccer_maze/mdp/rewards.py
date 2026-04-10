"""Soccer-maze specific reward functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def heading_alignment_world_cmd(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward for robot heading alignment with a world-frame command direction.

  The ball velocity command is in world frame [vx_w, vy_w, 0].  We rotate it
  into the robot body frame, then compute cos(heading_error) = vx_body / |v_body|.

  Returns +1 when the robot faces the commanded direction, 0 when 90° off,
  and -1 when facing backwards.  This breaks the sideways-walking optimum
  that can emerge in maze corridors.
  """
  command = env.command_manager.get_command(command_name)
  vel_world_2d = command[:, :2]  # [N, 2]

  robot = env.scene[asset_cfg.name]
  root_quat_w = robot.data.root_link_quat_w  # [N, 4] (w, x, y, z)
  quat_conj = torch.cat([root_quat_w[:, :1], -root_quat_w[:, 1:]], dim=-1)

  vel_world_3d = torch.cat(
    [vel_world_2d, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1
  )  # [N, 3]
  vel_body_2d = quat_apply(quat_conj, vel_world_3d)[:, :2]  # [N, 2]

  vel_norm = vel_body_2d.norm(dim=-1).clamp(min=1e-6)
  return vel_body_2d[:, 0] / vel_norm  # cos(heading_error)
