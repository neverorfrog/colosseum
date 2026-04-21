import torch
from mjlab.envs import ManagerBasedRlEnv

from colosseum.tasks.dribbling.mdp.obstacle_commands import ObstacleCommand


def ball_captured(
  env: ManagerBasedRlEnv,
  command_name: str = "adversary",
  capture_radius: float = 0.5,
) -> torch.Tensor:
  """Terminate when the ball is within *capture_radius* of any active obstacle.

  Returns a bool tensor of shape (N,).  Episodes with no active obstacles
  never terminate via this condition.
  """
  term = env.command_manager.get_term(command_name)
  if term.cfg.num_active == 0:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  ball = env.scene["ball"]
  ball_xy = ball.data.root_link_pos_w[:, :2]  # (N, 2)
  obs_xy = term.obstacle_positions_w[:, : term.cfg.num_active]  # (N, K_active, 2)

  dist = (obs_xy - ball_xy.unsqueeze(1)).norm(dim=-1)  # (N, K_active)
  return dist.min(dim=-1).values < capture_radius  # (N,)


def ball_lost(
  env: ManagerBasedRlEnv,
  max_robot_ball_distance: float = 2.0,
) -> torch.Tensor:
  """Terminate when the robot loses the ball by more than max_robot_ball_distance."""
  robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  return (ball_xy - robot_xy).norm(dim=-1) > max_robot_ball_distance
