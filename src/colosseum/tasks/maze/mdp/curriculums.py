import torch
from mjlab.envs import ManagerBasedRlEnv


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
