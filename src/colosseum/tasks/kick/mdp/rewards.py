"""kick reward functions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply

from colosseum.mdp.ball_rewards import ball_vel_body

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from colosseum.tasks.kick.mdp.commands import BallAngleCommand


def foot_ball_contact_motion(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """Reward foot-ball contact only when it sets the ball in motion.

  Bootstrap reward without a commanded direction: pays the step a foot touches
  the ball AND the ball is moving (|v_ball| > min_speed). A trapped
  (~stationary) ball pays nothing, so the contact reward can't be farmed by
  pinning the ball between the feet.
  """
  found = env.scene[sensor_name].data.found
  if found is None:
    return torch.zeros(env.num_envs, device=env.device)
  contact = (found.flatten(start_dim=1) > 0).any(dim=-1).float()

  ball_vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  good = ball_vel.norm(dim=-1) > min_speed
  return contact * good.float()


def ball_vel_angle_to_kick_goal(
  env: ManagerBasedRlEnv,
  command_name: str = "ball_angle",
  min_speed: float = 0.05,
) -> torch.Tensor:
  """Direction match 1 - (ψ_ball - ψ_goal)²/π². Zero when ball is stationary.

  Like ``colosseum.mdp.ball_rewards.ball_vel_angle_body``, but ``command_name``
  is a body-frame angle (BallAngleCommand, shape (N, 1), radians) rather than a
  world-frame velocity vector requiring rotation via ``cmd_body``.
  """
  ball_vel_b = ball_vel_body(env)
  psi_ball = torch.atan2(ball_vel_b[:, 1], ball_vel_b[:, 0])
  psi_cmd = env.command_manager.get_command(command_name)[:, 0]
  angle_err = (psi_ball - psi_cmd + math.pi) % (2 * math.pi) - math.pi
  reward = 1.0 - (angle_err**2) / (math.pi**2)
  return reward * (ball_vel_b.norm(dim=-1) > min_speed).float()


def robot_ball_yaw_to_kick_goal(
  env: ManagerBasedRlEnv,
  command_name: str = "ball_angle",
) -> torch.Tensor:
  """Ball ahead of robot along the kick-goal direction AND robot facing that way.

  e1 = 1 - dot(d_robot→ball_b, cmd_dir_b)
  e2 = 1 - d_robot→ball_b[0] / |d|  (ball in front, X-forward)
  reward = exp(-2 * (e1 + e2))

  Like ``colosseum.mdp.ball_rewards.robot_ball_yaw_body``, but ``command_name``
  is a body-frame angle (BallAngleCommand, shape (N, 1), radians) rather than a
  world-frame velocity vector requiring rotation via ``cmd_body``. The goal
  direction is always well-defined, so there's no min_speed gate.
  """
  robot = env.scene["robot"]
  ball_pos_w = env.scene["ball"].data.root_link_pos_w[:, :3]
  robot_pos_w = robot.data.root_link_pos_w[:, :3]

  relative_w = ball_pos_w - robot_pos_w
  quat_w = robot.data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  d_ball_b = quat_apply(quat_conj, relative_w)[:, :2]
  d_norm = d_ball_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  d_ball_b_unit = d_ball_b / d_norm

  angle = env.command_manager.get_command(command_name)[:, 0]
  unit_cmd = torch.stack([angle.cos(), angle.sin()], dim=-1)

  e1 = 1.0 - (d_ball_b_unit * unit_cmd).sum(dim=-1)
  e2 = 1.0 - d_ball_b_unit[:, 0]

  return torch.exp(-2.0 * (e1 + e2))


def _get_kick_gate(
  env: ManagerBasedRlEnv,
  sensor_name: str = "foot_ball_contact",
  min_contact_force: float = 2.0,
  kick_credit_steps: int = 15,
) -> torch.Tensor:
  """Per-env kick credit gate, updated at most once per policy step."""
  credit = getattr(env, "_kick3_credit", None)
  if credit is None or credit.shape[0] != env.num_envs:
    credit = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._kick3_credit = credit  # type: ignore[attr-defined]
    env._kick3_gate_step = -1  # type: ignore[attr-defined]
    env._kick3_gate = torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]

  current_step = int(env.common_step_counter)
  if current_step != env._kick3_gate_step:  # type: ignore[attr-defined]
    env._kick3_gate_step = current_step  # type: ignore[attr-defined]

    if hasattr(env, "episode_length_buf"):
      done_mask = env.episode_length_buf == 0
      if done_mask.any():
        credit[done_mask] = 0

    sensor = env.scene[sensor_name]
    contact_force = sensor.data.force.flatten(start_dim=1).norm(dim=-1)
    strike_now = contact_force >= min_contact_force

    credit[strike_now] = kick_credit_steps
    env._kick3_gate = (credit > 0).float()  # type: ignore[attr-defined]
    credit -= 1
    credit.clamp_(min=0)

  return env._kick3_gate  # type: ignore[attr-defined]


def ball_speed_kick_to_goal(
  env: ManagerBasedRlEnv,
  command_name: str = "ball_angle",
  max_reward: float = 8.0,
  saturation_velocity: float = 4.0,
  foot_contact_sensor_name: str = "foot_ball_contact",
  min_contact_force: float = 2.0,
  kick_credit_steps: int = 15,
  direction_weight: float = 0.0,
) -> torch.Tensor:
  """Speed/direction-blended kick reward, toward the BallAngleCommand goal.

  Like ``colosseum.tasks.kicking_5.mdp.rewards.ball_speed_kick``, but the
  direction term reads the per-episode world-frame goal angle stored by
  BallAngleCommand (``command_name``'s ``world_angle``) instead of a separate
  ``goal_angle`` command term — ``ball_angle`` itself is body-frame and not
  usable directly here.
  """
  gate = _get_kick_gate(env, foot_contact_sensor_name, min_contact_force, kick_credit_steps)

  ball_vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]  # (N, 2)
  ball_speed = ball_vel.norm(dim=-1)

  speed_only = ball_speed.clamp(0.0, max_reward)

  command: BallAngleCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  world_angle = command.world_angle[:, 0]
  d_hat = torch.stack([world_angle.cos(), world_angle.sin()], dim=-1)
  projection = (ball_vel * d_hat).sum(dim=-1)
  # Scaled so |projection| == saturation_velocity maps to |reward| == max_reward.
  k = math.log(max_reward + 1.0) / saturation_velocity
  projection_reward = torch.sign(projection) * (torch.exp(k * projection.abs()) - 1.0)
  projection_reward = projection_reward.clamp(min=-max_reward, max=max_reward)

  reward = (1.0 - direction_weight) * speed_only + direction_weight * projection_reward
  return gate * reward
