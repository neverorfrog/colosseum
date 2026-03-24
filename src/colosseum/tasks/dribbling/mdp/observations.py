import torch
from mjlab.envs import ManagerBasedRlEnvCfg

from .ball_approach_command import BallApproachCommand


def ball_position(env: ManagerBasedRlEnvCfg) -> torch.Tensor:
  ball_command = env.commands["ball_approach"]
  assert isinstance(ball_command, BallApproachCommand)
  return ball_command.ball
