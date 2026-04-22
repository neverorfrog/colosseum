"""Soccer-maze specific reward functions.

Body-frame variants of ball velocity rewards.  The Sokoban plan commands the
ball in world frame (``SokobanCommand.ball_vel``, non-zero only during PUSH).
The actor, however, observes ball position and ball velocity in body frame.
These rewards do the comparison in body frame so the signal the policy
optimises matches what it sees.

Mathematically the L2 and angular rewards are rotation-invariant — the values
are identical to their world-frame counterparts — but keeping everything in
body frame avoids mixing frames when the command is rotated for the actor
observation.

Ball-reward accessors read ``ball_vel`` directly via ``get_term(name)`` (not
``get_command(name)``, which returns the robot locomotion command and would
produce junk signal during MOVE).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply

from colosseum.mdp.abstraction.maze.sokoban_grid_abstraction import SokobanGridAbstraction
from colosseum.tasks.soccer_maze.mdp.sokoban_command import SokobanCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _sokoban(env: ManagerBasedRlEnv, command_name: str) -> SokobanCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, SokobanCommand)
  return term


def _ball_vel_body(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball XY velocity in robot body frame. Shape (N, 2)."""
  ball_vel_w = env.scene["ball"].data.root_link_lin_vel_w[:, :3]
  quat_w = env.scene["robot"].data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  return quat_apply(quat_conj, ball_vel_w)[:, :2]


def _cmd_body(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ball-velocity command rotated into robot body frame. Shape (N, 2).

  Reads ``SokobanCommand.ball_vel`` (world frame, non-zero only during PUSH).
  """
  cmd_w = _sokoban(env, command_name).ball_vel  # [N, 2] world frame
  quat_w = env.scene["robot"].data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  cmd_3d = torch.cat([cmd_w, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1)
  return quat_apply(quat_conj, cmd_3d)[:, :2]


def ball_vel_tracking_body(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 1.0,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """exp(-sharpness * |v_ball_b - v_cmd_b|²). Body-frame, PUSH only.

  Gated on ball speed > min_speed so the reward is silent when the ball is
  stationary (undefined velocity direction); the approach reward covers that phase.
  """
  ball_vel_b = _ball_vel_body(env)
  cmd_b = _cmd_body(env, command_name)
  error_sq = ((ball_vel_b - cmd_b) ** 2).sum(dim=-1)
  moving = (ball_vel_b.norm(dim=-1) > min_speed).float()
  return torch.exp(-sharpness * error_sq) * _sokoban(env, command_name).is_push.float() * moving


def ball_vel_norm_body(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 1.0,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """exp(-sharpness * (|v_cmd| - |v_ball|)²). Magnitude match, PUSH only.

  Gated on ball speed > min_speed. Uses sokoban.ball_vel (world frame) so it
  reads the pure push signal, not the locomotion command carried by .command.
  """
  ball_vel_w = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  sokoban = _sokoban(env, command_name)
  speed_err = (sokoban.ball_vel.norm(dim=-1) - ball_vel_w.norm(dim=-1)) ** 2
  moving = (ball_vel_w.norm(dim=-1) > min_speed).float()
  return torch.exp(-sharpness * speed_err) * sokoban.is_push.float() * moving


def ball_vel_angle_body(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """Direction match 1 - (ψ_ball - ψ_cmd)²/π². Body-frame, PUSH only.

  Gated on both ball speed and command speed > min_speed: ball must be moving
  and a push command must be active.
  """
  ball_vel_b = _ball_vel_body(env)
  cmd_b = _cmd_body(env, command_name)
  psi_ball = torch.atan2(ball_vel_b[:, 1], ball_vel_b[:, 0])
  psi_cmd = torch.atan2(cmd_b[:, 1], cmd_b[:, 0])
  angle_err = (psi_ball - psi_cmd + math.pi) % (2 * math.pi) - math.pi
  active = (cmd_b.norm(dim=-1) > min_speed).float() * (ball_vel_b.norm(dim=-1) > min_speed).float()
  return (1.0 - (angle_err**2) / (math.pi**2)) * active


def robot_ball_yaw_body(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """Ball ahead of robot along command direction (body frame).

  e1 = 1 - dot(d_robot->ball_b, cmd_dir_b)  (ball in command direction)
  e2 = 1 - d_robot->ball_b[0] / |d|         (ball in front, X-forward)
  reward = exp(-2 * (e1 + e2))
  """
  robot = env.scene["robot"]
  ball_pos_w = env.scene["ball"].data.root_link_pos_w[:, :3]
  robot_pos_w = robot.data.root_link_pos_w[:, :3]

  # Robot-to-ball direction in body frame
  relative_w = ball_pos_w - robot_pos_w
  quat_w = robot.data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  d_ball_b = quat_apply(quat_conj, relative_w)[:, :2]
  d_norm = d_ball_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  d_ball_b_unit = d_ball_b / d_norm

  # Command direction in body frame
  cmd_b = _cmd_body(env, command_name)
  cmd_norm = cmd_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  unit_cmd = cmd_b / cmd_norm

  e1 = 1.0 - (d_ball_b_unit * unit_cmd).sum(dim=-1)
  e2 = 1.0 - d_ball_b_unit[:, 0]  # dot with [1, 0] (X-forward)

  reward = torch.exp(-2.0 * (e1 + e2))
  return reward * (cmd_b.norm(dim=-1) > min_speed).float()


def robot_heading_alignment(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """cos(heading_error) during MOVE; zero during PUSH.

  Uses SokobanCommand.heading_cos = X-component of target direction in body frame,
  which SokobanCommand already zeros during PUSH.  Range [-1, 1]: +1 when robot
  faces the MOVE target, 0 when 90° off, -1 when backward.
  """
  sokoban: SokobanCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  return sokoban.heading_cos


def robot_lin_vel_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Track robot body-frame linear velocity. MOVE only."""
  sokoban = _sokoban(env, command_name)
  actual = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
  error = ((sokoban.robot_lin_vel - actual) ** 2).sum(dim=-1)
  return torch.exp(-error / std**2) * (~sokoban.is_push).float()


def robot_ang_vel_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Track robot body-frame yaw rate. MOVE only."""
  sokoban = _sokoban(env, command_name)
  actual = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
  error = (sokoban.robot_omega_z - actual) ** 2
  return torch.exp(-error / std**2) * (~sokoban.is_push).float()


def robot_ball_approach_vel_push(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Approach reward gated to PUSH only.

  Rewards the robot for moving toward the ball at the commanded push speed,
  active throughout the PUSH phase (whether ball is stationary or rolling).
  This keeps the robot always oriented toward the ball so it can re-kick
  if the ball stops short of the target cell.
  """
  sokoban: SokobanCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  robot = env.scene["robot"]
  ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
  robot_pos = robot.data.root_link_pos_w[:, :2]
  d = ball_pos - robot_pos
  d_unit = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  approach_vel = (robot.data.root_link_lin_vel_w[:, :2] * d_unit).sum(dim=-1)
  cmd_speed = sokoban.ball_vel.norm(dim=-1)  # non-zero only during PUSH
  return (
    (approach_vel / cmd_speed.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
    * sokoban.is_push.float()
  )


def lateral_velocity_penalty_push(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Penalise body-frame lateral velocity during PUSH.

  Sideways walking wastes time and prevents the robot from lining up with the
  ball.  The robot should turn (omega_z) to face the ball and walk forward.
  """
  sokoban: SokobanCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  vy = env.scene["robot"].data.root_link_lin_vel_b[:, 1]
  return -(vy**2) * sokoban.is_push.float()


def ball_displacement_push(
  env: ManagerBasedRlEnv,
  command_name: str,
  cell_size: float = 2.0,
) -> torch.Tensor:
  """Ball progress toward target cell during PUSH, normalized to [0, 1].

  Measures how far the ball has traveled in the push direction since the start
  of the current PUSH action. Reward is 1.0 when ball has crossed a full cell
  (cell_size metres). Unlike velocity rewards this stays high after the kick,
  so the robot has no incentive to chase and re-kick a rolling ball.
  """
  sokoban = _sokoban(env, command_name)
  ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
  displacement = ball_pos - sokoban.push_start_ball_pos
  push_dir = sokoban.ball_vel / sokoban.ball_vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  progress = (displacement * push_dir).sum(dim=-1).clamp(min=0.0, max=cell_size)
  return (progress / cell_size) * sokoban.is_push.float()


def ball_push_target_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  speed_ref: float = 0.5,
) -> torch.Tensor:
  """Ball velocity projected toward the PUSH target cell center. PUSH only.

  Mirrors dribbling's ball_target_progress: rewards the ball for moving in the
  direction of the target cell, normalized by speed_ref.  Active throughout
  the PUSH phase (ball stationary or rolling) so the robot always has a
  gradient to keep pushing until the cell is reached.
  """
  sokoban = _sokoban(env, command_name)
  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  ball_vel_xy = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  target_xy = sokoban.push_target_pos  # [N, 2] world frame

  target_vec = target_xy - ball_xy
  target_dist = target_vec.norm(dim=-1)
  target_dir = target_vec / target_dist.unsqueeze(-1).clamp(min=1e-6)

  progress = (ball_vel_xy * target_dir).sum(dim=-1).clamp(min=0.0)
  return (progress / max(speed_ref, 1e-6)).clamp(max=1.0) * sokoban.is_push.float()


def ball_push_target_reached(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float = 0.5,
) -> torch.Tensor:
  """One-shot bonus when the ball is within threshold of the PUSH target cell center.

  Mirrors dribbling's ball_target_reached: provides a discrete bonus that fires
  each step the ball is inside the target radius, giving a clear arrival signal.
  Gated to PUSH so it doesn't fire spuriously during MOVE.
  """
  sokoban = _sokoban(env, command_name)
  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  target_xy = sokoban.push_target_pos  # [N, 2] world frame
  dist = (ball_xy - target_xy).norm(dim=-1)
  return (dist < threshold).float() * sokoban.is_push.float()


def robot_ball_distance_push(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 0.5,
) -> torch.Tensor:
  """robot_ball_distance gated to PUSH actions only.

  During MOVE the robot must freely reposition; penalizing distance then
  prevents it from walking behind the ball to set up the correct push angle.
  """
  sokoban: SokobanCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  robot_pos = env.scene["robot"].data.root_link_pos_w[:, :2]
  ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
  dist_sq = ((ball_pos - robot_pos) ** 2).sum(dim=-1)
  return torch.exp(-sharpness * dist_sq) * sokoban.is_push.float()


def action_step_timeout_penalty(
  env: ManagerBasedRlEnv,
  abstraction_name: str = "sokoban",
  window_steps: int = 200,
) -> torch.Tensor:
  """Penalise envs stuck on the same plan step for too long.

  Counts consecutive steps without plan advancement.  Once the counter exceeds
  *window_steps* (≈ 4 s at 50 Hz) the function returns -1.0 per step.

  The counter resets on any change to ``current_step`` — both forward
  advancement and replanning (which resets ``current_step`` to 0).  This
  creates pressure to complete each plan action within the window without
  over-penalising genuinely slow but legitimate phases.
  """
  abstraction = env.abstraction_manager.get_term(abstraction_name)
  assert isinstance(abstraction, SokobanGridAbstraction)

  if not hasattr(env, "_action_step_counter"):
    env._action_step_counter = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._prev_plan_step = abstraction.current_step.clone()  # type: ignore[attr-defined]

  step_changed = abstraction.current_step != env._prev_plan_step  # type: ignore[attr-defined]
  env._action_step_counter = torch.where(  # type: ignore[attr-defined]
    step_changed,
    torch.zeros_like(env._action_step_counter),  # type: ignore[attr-defined]
    env._action_step_counter + 1,  # type: ignore[attr-defined]
  )
  env._prev_plan_step = abstraction.current_step.clone()  # type: ignore[attr-defined]

  return -(env._action_step_counter >= window_steps).float()  # type: ignore[attr-defined]


def ball_at_final_goal(
  env: ManagerBasedRlEnv,
  command_name: str = "goal",
  threshold: float = 0.5,
) -> torch.Tensor:
  """Large per-step bonus when the ball is inside the maze goal zone.

  Reads the goal position from the ``goal`` command (world-frame XY) and
  compares it to the ball's current position.  Active in both MOVE and PUSH
  phases so the agent always has a gradient toward completing the maze.

  This prevents the agent from hovering just outside the arrived_at_goal
  threshold: the bonus dominates any per-step locomotion reward and makes
  crossing the goal boundary strictly better than stopping short.
  """
  goal_pos = env.command_manager.get_command(command_name)  # [N, 2] world XY
  ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
  dist = (ball_pos - goal_pos).norm(dim=-1)
  return (dist < threshold).float()


def no_ball_contact_push(
  env: ManagerBasedRlEnv,
  command_name: str,
  window_steps: int = 250,
  ball_speed_threshold: float = 0.1,
) -> torch.Tensor:
  """Penalise PUSH phases where the robot fails to get the ball moving.

  A counter tracks consecutive PUSH steps where the ball speed is below
  *ball_speed_threshold*.  It resets whenever the ball is actually moving
  (a real kick occurred) or the PUSH phase ends.  Once the counter exceeds
  *window_steps* the function returns -1.0 per step.

  Foot-ball contact was not used because the feet can brush the ball while
  marching in place, which would reset the counter without a real kick.

  Default window: 250 steps ≈ 5 s at 50 Hz policy rate.
  """
  sokoban = _sokoban(env, command_name)

  if not hasattr(env, "_no_ball_contact_steps"):
    env._no_ball_contact_steps = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=env.device
    )

  ball_speed = env.scene["ball"].data.root_link_lin_vel_w[:, :2].norm(dim=-1)
  ball_moving = ball_speed > ball_speed_threshold

  # Increment during PUSH while ball is stationary; reset otherwise.
  env._no_ball_contact_steps = torch.where(  # type: ignore[attr-defined]
    sokoban.is_push & ~ball_moving,
    env._no_ball_contact_steps + 1,
    torch.zeros_like(env._no_ball_contact_steps),
  )

  return -(env._no_ball_contact_steps >= window_steps).float()
