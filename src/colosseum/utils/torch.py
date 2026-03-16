"""PyTorch utilities for RL training, including observation space handling."""

import random

import numpy as np
import torch
from gymnasium import spaces as gym_spaces
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils import spaces as mjlab_spaces


def set_seed(seed: int, torch_deterministic: bool = True):
  """Set random seeds for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  if torch_deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  else:
    torch.backends.cudnn.benchmark = True


def get_device(cuda: bool = True, device_id: int = 0) -> torch.device:
  """Get PyTorch device."""
  if cuda and torch.cuda.is_available():
    return torch.device(f"cuda:{device_id}")
  return torch.device("cpu")


# ====== Observation Space Utilities ======


def get_obs_dims(env: ManagerBasedRlEnv) -> dict[str, int]:
  """Extract observation dimensions from environment's observation space.

  Handles both flat Box spaces and Dict spaces (mjlab-style with "actor"/"critic" groups).
  Supports both gymnasium.spaces and mjlab.utils.spaces types.

  Args:
      env: Environment with single_observation_space attribute

  Returns:
      Dictionary mapping observation group names to dimensions.
      For Box spaces: {"actor": dim, "critic": dim} (same dim for both)
      For Dict spaces: {"actor": actor_dim, "critic": critic_dim}

  Raises:
      TypeError: If observation space is neither Box nor Dict
      ValueError: If Dict space is missing required groups

  Example:
      >>> env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
      >>> dims = get_obs_dims(env)
      >>> print(dims)
      {"actor": 4, "critic": 4}
  """
  obs_space = env.single_observation_space

  # Check for Dict space (both gymnasium and mjlab)
  is_dict = isinstance(obs_space, (gym_spaces.Dict, mjlab_spaces.Dict))
  # Check for Box space (both gymnasium and mjlab)
  is_box = isinstance(obs_space, (gym_spaces.Box, mjlab_spaces.Box))

  if is_dict:
    if "actor" not in obs_space.spaces:
      raise ValueError(
        f"Dict observation space must have 'actor' group. "
        f"Found: {list(obs_space.spaces.keys())}"
      )
    if "critic" not in obs_space.spaces:
      raise ValueError(
        f"Dict observation space must have 'critic' group. "
        f"Found: {list(obs_space.spaces.keys())}"
      )

    dims = {}
    for group_name in ["actor", "critic"]:
      group_space = obs_space.spaces[group_name]
      if isinstance(group_space, (gym_spaces.Box, mjlab_spaces.Box)):
        dims[group_name] = int(np.prod(group_space.shape))
      elif isinstance(group_space, (gym_spaces.Dict, mjlab_spaces.Dict)):
        total = 0
        for subspace in group_space.spaces.values():
          if isinstance(subspace, (gym_spaces.Box, mjlab_spaces.Box)):
            total += int(np.prod(subspace.shape))
        dims[group_name] = total
      else:
        raise TypeError(
          f"Unsupported space type for group '{group_name}': {type(group_space)}"
        )

    return dims

  elif is_box:
    obs_dim = int(np.prod(obs_space.shape))
    return {"actor": obs_dim, "critic": obs_dim}

  else:
    raise TypeError(
      f"Unsupported observation space type: {type(obs_space)}. "
      f"Must be Box or Dict."
    )
