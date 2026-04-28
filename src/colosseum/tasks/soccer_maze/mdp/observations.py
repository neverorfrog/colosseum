"""Soccer-maze specific observation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply

from colosseum.mdp.abstraction.maze.grid_abstraction import GridAbstraction
from colosseum.tasks.soccer_maze.mdp.sokoban_command import SokobanCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ball_vel_command_body(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ball velocity command rotated into the robot body frame. Shape (N, 2).

  Reads ``SokobanCommand.ball_vel`` (world frame, non-zero only during PUSH) so
  the observation is zero during MOVE — matching the actor's expectation that
  the ball target is meaningful only while pushing.
  """
  sokoban: SokobanCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  cmd_w = sokoban.ball_vel  # [N, 2] world
  robot = env.scene["robot"]
  quat_w = robot.data.root_link_quat_w  # [N, 4]
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  cmd_3d = torch.cat(
    [cmd_w, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1
  )  # [N, 3]
  return quat_apply(quat_conj, cmd_3d)[:, :2]  # [N, 2]


def obstacle_map(env: ManagerBasedRlEnv, abstraction_name: str) -> torch.Tensor:
  """Flattened obstacle mask as bird's-eye view of the maze. Shape [N, rows*cols].

  Each element is 1.0 (wall) or 0.0 (free).  The mask is env-independent so the
  same tensor is broadcast across all environments.  Row-major order: element
  [i * cols + j] corresponds to grid cell (i, j).

  This is a privileged observation — add it to critic_terms only.
  """
  assert hasattr(env, "abstraction_manager")
  abstraction = env.abstraction_manager.get_term(abstraction_name)
  assert isinstance(abstraction, GridAbstraction)
  flat = abstraction.map.float().flatten()  # [rows*cols]
  return flat.unsqueeze(0).expand(env.num_envs, -1)  # [N, rows*cols]


def robot_vel_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Robot velocity command in body frame. Shape [N, 3]: [vx_b, vy_b, omega_z].

  Same format as AbstractionVelocityCommand: body-frame direction scaled by
  speed plus the yaw-rate from the heading P-controller.  Populated only during
  MOVE; zero during PUSH so the actor sees no locomotion target while kicking.
  """
  sokoban: SokobanCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  return torch.cat(
    [sokoban.robot_lin_vel, sokoban.robot_omega_z.unsqueeze(-1)], dim=-1
  )  # [N, 3]


def ball_vel_xy_body(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball XY velocity in robot body frame. Shape (N, 2).

  Consistent frame with ball_pos (also body frame) so the actor can reason
  about ball state without needing heading knowledge.
  """
  ball_vel_w = env.scene["ball"].data.root_link_lin_vel_w[:, :3]  # [N, 3]
  robot = env.scene["robot"]
  quat_w = robot.data.root_link_quat_w  # [N, 4]
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  return quat_apply(quat_conj, ball_vel_w)[:, :2]  # [N, 2]
