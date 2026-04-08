"""RmaPPO — PPO with RMA-style privileged encoders.

Minimal subclass of PPO. Differences from plain PPO:

1. Expects ``env.unwrapped.rma_manager`` (RmaManager) to be present.
2. Widens the actor input: proprio_dim + rma_manager.total_latent_dim.
3. Adds encoder parameters to the optimizer.
4. Overrides ``_compose_actor_input`` to concatenate encoder latents.
5. Overrides ``get_privileged_obs`` to extract the groups the encoders need.
6. Overrides ``_build_rollout_buffer`` to pre-allocate privileged obs storage.

Phase 1: all encoders are PrivilegedEncoders, trained jointly with the policy
         via the PPO loss (gradients flow: loss → actor → z → encoder.params).
Phase 2: encoders are swapped (via rma_manager.swap_encoder) and fine-tuned
         with a separate script; this class is not responsible for Phase 2.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any, Callable

import torch
import torch.optim as optim
from mjlab.envs import ManagerBasedRlEnv

from colosseum.algorithm.ppo import PPO
from colosseum.algorithm.rollout_buffer import RolloutBuffer
from colosseum.config.types.algorithm import PpoConfig, register_algorithm
from colosseum.managers.rma_manager import RmaManager


@register_algorithm("rma_ppo", config_class=PpoConfig)
class RmaPPO(PPO):
  """PPO with RMA-style privileged encoders.

  Expects the environment to expose ``rma_manager`` on its unwrapped instance.
  All encoder parameters train jointly with the policy via the same optimizer.
  """

  def __init__(
    self,
    config: PpoConfig,
    env: ManagerBasedRlEnv,
    device: str | torch.device,
    log_fn: Callable[[dict[str, float], int], None],
    log_interval: int,
  ) -> None:
    # Resolve rma_manager before super().__init__ so _build_networks can use it
    unwrapped = getattr(env, "unwrapped", env)
    self.rma_manager: RmaManager = unwrapped.rma_manager
    super().__init__(
      config=config, env=env, device=device, log_fn=log_fn, log_interval=log_interval
    )

  # ------------------------------------------------------------------
  # Network / optimizer / buffer construction
  # ------------------------------------------------------------------

  def _build_networks(self) -> None:
    """Build actor and critic, widening actor input by total_latent_dim."""
    super()._build_networks()

    # Widen actor: rebuild with proprio_dim + latent_dim as input
    from colosseum.algorithm.ppo_networks import PpoActor

    assert isinstance(self.config, PpoConfig)
    actor_input_dim = self.actor_obs_dim + self.rma_manager.total_latent_dim
    self.actor = PpoActor(
      actor_input_dim,
      self.action_dim,
      self.config.actor,
    ).to(self.device)

  def _build_optimizers(self) -> None:
    """Single optimizer for actor + critic + all encoders."""
    assert isinstance(self.config, PpoConfig)
    self.optimizer = optim.Adam(
      chain(
        self.actor.parameters(),
        self.value_net.parameters(),
        self.rma_manager.parameters(),
      ),
      lr=self.config.learning_rate,
    )

  def _build_rollout_buffer(self) -> None:
    """Allocate buffer with privileged obs storage per encoder group."""
    assert isinstance(self.config, PpoConfig)

    # Gather per-group dims from the observation manager
    obs_mgr = self.env.observation_manager
    privileged_obs_dims = {
      group: obs_mgr.group_obs_dim[group][0]
      for group in self.rma_manager.privileged_group_names
    }

    self.rollout_buffer = RolloutBuffer(
      num_envs=self.env.num_envs,
      num_steps=self.config.num_steps_per_env,
      actor_obs_dim=self.actor_obs_dim,
      critic_obs_dim=self.critic_obs_dim,
      action_dim=self.action_dim,
      device=self.device,
      privileged_obs_dims=privileged_obs_dims,
    )

  # ------------------------------------------------------------------
  # RMA hooks
  # ------------------------------------------------------------------

  def get_privileged_obs(self, obs: Any) -> dict[str, torch.Tensor]:
    """Extract the groups needed by the rma_manager from the obs dict."""
    if not isinstance(obs, dict):
      return {}
    return {
      group: obs[group]
      for group in self.rma_manager.privileged_group_names
      if group in obs
    }

  def _compose_actor_input(
    self,
    actor_obs: torch.Tensor,
    privileged_obs: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    """Concatenate normalised proprio with encoder latents.

    Args:
      actor_obs:      (N, actor_obs_dim) normalised proprioceptive obs.
      privileged_obs: Dict of GT privileged groups for the encoders.

    Returns:
      (N, actor_obs_dim + total_latent_dim) actor input tensor.
    """
    z = self.rma_manager.encode(privileged_obs)
    return torch.cat([actor_obs, z], dim=-1)

  # ------------------------------------------------------------------
  # Checkpoint
  # ------------------------------------------------------------------

  def save(self, path: str | Path, **extra_state: Any) -> None:
    """Save PPO checkpoint including encoder state dicts."""
    if "global_step" not in extra_state:
      raise ValueError("global_step must be provided in extra_state")

    state_dict = {
      "actor_state_dict": self.actor.state_dict(),
      "value_net_state_dict": self.value_net.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "actor_obs_normalizer_state_dict": self.actor_obs_normalizer.state_dict(),
      "critic_obs_normalizer_state_dict": self.critic_obs_normalizer.state_dict(),
      "rma_manager_state_dict": self.rma_manager.state_dict(),
      "global_step": extra_state["global_step"],
      "learning_rate": self.learning_rate,
      "config": self.config,
    }
    for key, value in extra_state.items():
      if key not in state_dict:
        state_dict[key] = value
    self._save_checkpoint(path, state_dict)

  def load(self, path: str | Path) -> dict[str, Any]:
    """Load PPO checkpoint including encoder state dicts."""
    checkpoint = self._load_checkpoint(path)

    self.actor.load_state_dict(checkpoint["actor_state_dict"])
    self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.actor_obs_normalizer.load_state_dict(
      checkpoint["actor_obs_normalizer_state_dict"]
    )
    self.critic_obs_normalizer.load_state_dict(
      checkpoint["critic_obs_normalizer_state_dict"]
    )
    if "rma_manager_state_dict" in checkpoint:
      self.rma_manager.load_state_dict(checkpoint["rma_manager_state_dict"])

    self.global_step = checkpoint["global_step"]
    self.learning_rate = checkpoint.get("learning_rate", self.learning_rate)
    self._restore_env_step_counter()

    return {
      "global_step": checkpoint["global_step"],
      "metadata": checkpoint.get("metadata", {}),
      "config": checkpoint.get("config"),
    }
