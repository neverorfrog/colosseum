"""Teacher policy for DAgger-style imitation learning.

Wraps a pre-trained checkpoint (e.g. velocity task policy) and exposes a
single method get_actions() that maps the student's observation tensor to
deterministic teacher actions.

The key assumption is that the teacher's observation is a contiguous suffix
of the student's observation:

    student_obs[:, obs_start_idx:] == teacher_obs

This holds for the maze → velocity transfer where the maze actor obs is:
    [agent_pos(2), goal_pos(2), <velocity actor obs (78)>]

so obs_start_idx = 4 for the maze task.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from colosseum.algorithm.networks.ppo_networks import PpoActor
from colosseum.algorithm.utils.normalization import EmpiricalNormalization
from colosseum.config.types.networks import PpoActorConfig


class TeacherPolicy(nn.Module):
  """Pre-trained teacher policy used during DAgger rollouts.

  Loads actor weights and observation normalizer from a colosseum checkpoint,
  freezes all parameters, and provides deterministic action inference.
  """

  def __init__(
    self,
    checkpoint_path: str | Path,
    obs_dim: int,
    action_dim: int,
    actor_cfg: PpoActorConfig,
    obs_start_idx: int,
    device: torch.device | str,
  ) -> None:
    super().__init__()
    self.obs_start_idx = obs_start_idx
    checkpoint = torch.load(
      Path(checkpoint_path), map_location="cpu", weights_only=False
    )

    if obs_dim <= 0:
      first_linear_weight = checkpoint["actor_state_dict"]["backbone.0.weight"]
      obs_dim = int(first_linear_weight.shape[1])

    self.actor = PpoActor(obs_dim, action_dim, actor_cfg)
    self.actor.load_state_dict(checkpoint["actor_state_dict"])

    self.normalizer = EmpiricalNormalization(obs_dim)
    self.normalizer.load_state_dict(checkpoint["actor_obs_normalizer_state_dict"])

    self.to(device)
    self.eval()
    for param in self.parameters():
      param.requires_grad_(False)

  @torch.no_grad()
  def get_actions(self, student_obs: torch.Tensor) -> torch.Tensor:
    """Return deterministic teacher actions from student observations.

    Args:
        student_obs: Full student actor obs [num_envs, student_obs_dim]

    Returns:
        Deterministic action means [num_envs, action_dim]
    """
    teacher_obs = student_obs[:, self.obs_start_idx :]
    norm_obs = self.normalizer(teacher_obs)
    return self.actor.forward(norm_obs)
