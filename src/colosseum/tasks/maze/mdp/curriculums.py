import torch
from mjlab.envs import ManagerBasedRlEnv

from colosseum.mdp.abstraction.maze.abstraction_velocity_command import AbstractionVelocityCommandCfg


def base_velocity_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[dict],
) -> dict[str, torch.Tensor]:
  """Ramp up AbstractionVelocityCommand base_velocity over training stages.

  Each stage is a dict with keys ``step`` (global step threshold) and
  ``base_velocity`` (target speed in m/s). Stages are applied in order —
  the last stage whose ``step`` has been reached wins.
  """
  del env_ids
  command_term = env.command_manager.get_term(command_name)
  cfg: AbstractionVelocityCommandCfg = command_term.cfg
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      cfg.base_velocity = stage["base_velocity"]
  return {"base_velocity": torch.tensor(cfg.base_velocity)}


def wall_collision_termination_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  end_step: int = 5_000_000,
  start_prob: float = 0.0,
  end_prob: float = 1.0,
) -> torch.Tensor:
  """Schedule wall-collision termination probability as training progresses.

  At the start of training the probability is ``start_prob`` so episodes are
  never (or rarely) cut short on collision, letting the agent explore freely
  and still collect the negative collision reward. As training advances the
  probability rises linearly until it reaches ``end_prob`` at ``end_step``
  global steps.

  The result is stored on the environment as ``env.wall_termination_prob`` so
  that the ``collided_with_wall`` termination function can read it each step.

  Args:
      env: The RL environment.
      env_ids: Environments being reset (unused – probability is global).
      end_step: Global step count at which the probability reaches end_prob.
      start_prob: Termination probability at step 0.
      end_prob: Termination probability at end_step and beyond.

  Returns:
      Scalar tensor with the current probability (logged as a curriculum metric).
  """
  del env_ids  # Probability is global, not per-environment.

  total_transitions = env.common_step_counter * env.num_envs
  progress = min(total_transitions / end_step, 1.0)
  prob = start_prob + (end_prob - start_prob) * progress

  env.wall_termination_prob = prob

  return torch.tensor(prob, dtype=torch.float32)
