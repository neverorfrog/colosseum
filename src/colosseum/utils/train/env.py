"""Environment utilities for training and evaluation."""

import torch
from mjlab.envs import ManagerBasedRlEnv


class ViewerCompatibleEnv(ManagerBasedRlEnv):
  """Adds get_observations() for mjlab viewer compatibility.

  The mjlab viewer expects environments to have a get_observations() method,
  but ManagerBasedRlEnv doesn't provide it.
  """

  def get_observations(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Get current observations without stepping the environment."""
    return self.observation_manager.compute()
