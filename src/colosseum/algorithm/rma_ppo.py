"""RmaPPO — PPO with RMA-style privileged encoders.

Minimal subclass of PPO. Differences from plain PPO:

1. Expects ``env.unwrapped.rma_manager`` (RmaManager) to be present.
2. Widens the actor input: proprio_dim + rma_manager.total_latent_dim.
3. Adds privileged encoder parameters to the optimizer (Phase 1).
4. Overrides ``_compose_actor_input`` to concatenate encoder latents.
5. Overrides ``get_privileged_obs`` to extract the groups the encoders need.
6. Overrides ``_build_rollout_buffer`` to pre-allocate privileged obs storage.

Phase 1: PrivilegedEncoders trained jointly with the policy via the PPO loss
         (gradients flow: loss → actor → z → encoder.params).
Phase 2: call ``build_adaptation_optimizer()`` then use ``adaptation_learning_step()``
         instead of the standard PPO learning step. The privileged encoders and
         policy are frozen; only adaptation encoders are updated (MSE regression).
"""

from __future__ import annotations

import time
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Callable

import torch
import torch.optim as optim
from loguru import logger
from mjlab.envs import ManagerBasedRlEnv

from colosseum.algorithm.base_algorithm import ObsType
from colosseum.algorithm.ppo import PPO
from colosseum.algorithm.rollout_buffer import RolloutBuffer
from colosseum.config.types.algorithm import PpoConfig, register_algorithm
from colosseum.managers.rma_manager import RmaManager
from colosseum.utils.logger import extract_episode_metrics


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
    self._phase: int = 1  # 1 = PPO (default), 2 = adaptation encoder regression
    # Inference phase: controls which encoder _compose_actor_input uses at eval time.
    # Reads from config.inference_phase if present (RmaPPOConfig), else defaults to 1.
    self._inference_phase: int = getattr(config, "inference_phase", 1)
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
  # Rollout collection — Phase 1 (PPO with privileged encoder latents)
  # ------------------------------------------------------------------

  def _collect_rollout(
    self,
    current_actor_obs: torch.Tensor,
    current_critic_obs: torch.Tensor,
    current_dones: torch.Tensor,
    obs_dict: ObsType,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ObsType]:
    """Phase 1 rollout: same as PPO but widens actor input with encoder latents."""
    assert isinstance(self.config, PpoConfig)

    self.rollout_buffer.clear()

    with torch.no_grad():
      for _step in range(self.config.num_steps_per_env):
        norm_actor_obs_base = self.actor_obs_normalizer(current_actor_obs)
        norm_critic_obs = self.critic_obs_normalizer(current_critic_obs)

        # Compose actor input: cat([norm_proprio, encoder_latents])
        current_privileged_obs = self.get_privileged_obs(obs_dict)
        norm_actor_obs = self._compose_actor_input(norm_actor_obs_base, current_privileged_obs)

        actions, log_probs, action_means, action_stds = self.actor.act_with_log_prob(
          norm_actor_obs
        )
        values = self.value_net(norm_critic_obs)

        obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = (terminated | truncated).float()

        next_actor_obs = self.get_actor_obs(obs_dict)
        next_critic_obs = self.get_critic_obs(obs_dict)

        if self.config.obs_normalization:
          self.actor_obs_normalizer.update(next_actor_obs)
          self.critic_obs_normalizer.update(next_critic_obs)

        self.cur_reward_sum += rewards

        if not getattr(self.env.cfg, "is_finite_horizon", True):
          truncated_mask = truncated.float()
          if truncated_mask.any():
            norm_next_critic = self.critic_obs_normalizer(next_critic_obs)
            truncated_values = self.value_net(norm_next_critic).squeeze(-1)
            rewards = rewards + self.config.gamma * truncated_values * truncated_mask

        self.episode_length_buf += 1
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
          self.rewbuffer.extend(self.cur_reward_sum[done_ids].cpu().numpy().tolist())
          self.cur_reward_sum[done_ids] = 0.0
          self.episode_lengths.extend(
            self.episode_length_buf[done_ids].cpu().numpy().tolist()
          )
          self.episode_length_buf[done_ids] = 0

        self.update_episode_counts(terminated, truncated)

        if "log" in infos and dones.any():
          self.latest_episode_metrics = extract_episode_metrics(infos["log"])

        self.rollout_buffer.add(
          actor_obs=current_actor_obs,
          critic_obs=current_critic_obs,
          actions=actions,
          rewards=rewards,
          dones=dones,
          values=values,
          log_probs=log_probs,
          action_means=action_means,
          action_stds=action_stds,
          privileged_obs=current_privileged_obs,
        )

        current_actor_obs = next_actor_obs
        current_critic_obs = next_critic_obs
        current_dones = dones

      norm_last_critic = self.critic_obs_normalizer(current_critic_obs)
      last_values = self.value_net(norm_last_critic)

    normalize_globally = not self.config.normalize_advantage_per_mini_batch
    self.rollout_buffer.compute_returns_and_advantages(
      last_values=last_values,
      gamma=self.config.gamma,
      lam=self.config.lam,
      normalize_advantage=normalize_globally,
    )

    return current_actor_obs, current_critic_obs, current_dones, obs_dict

  # ------------------------------------------------------------------
  # Training dispatch + Phase 2 loop
  # ------------------------------------------------------------------

  def train(self) -> None:
    if self._phase == 1:
      super().train()
    else:
      self._train_phase2()

  def _train_phase2(self) -> None:
    """Phase 2 outer loop: adaptation encoder regression.

    build_adaptation_optimizer() and _phase = 2 must be set before calling
    (train_phase2.py handles this).
    """
    assert isinstance(self.config, PpoConfig)

    self.start_time = time.time()

    losses_buffer: dict[str, list[float]] = defaultdict(list)
    collection_time_sum = 0.0

    total_timesteps = self.config.learning_steps
    steps_per_iter = self.config.num_steps_per_env * self.env.num_envs
    num_iterations = total_timesteps // steps_per_iter
    log_interval_iters = max(1, self.log_interval // steps_per_iter)

    logger.info("=" * 80)
    logger.info("Starting RMA Phase 2 training (adaptation encoder regression)")
    logger.info(f"Total steps:    {total_timesteps}")
    logger.info(f"Steps per iter: {steps_per_iter}")
    logger.info(f"Num iterations: {num_iterations}")
    logger.info("=" * 80)

    obs_dict, _ = self.env.reset(seed=self.seed)

    for iteration in range(1, num_iterations + 1):
      start = time.perf_counter()
      obs_dict, loss_dict = self._phase2_learning_step(obs_dict)
      collection_time_sum += time.perf_counter() - start

      for k, v in loss_dict.items():
        losses_buffer[k].append(v)

      self.global_step += steps_per_iter
      self._maybe_save_checkpoint(self.global_step)

      if iteration % log_interval_iters == 0:
        self._log_training_metrics(
          step=self.global_step,
          losses_buffer=losses_buffer,
          collection_time=collection_time_sum,
          learning_time=0.0,
          log_interval=log_interval_iters,
          total_timesteps=total_timesteps,
          title="RMA Phase 2",
          use_rich=self.config.use_rich_logging,
          steps_per_log_step=self.config.num_steps_per_env,
        )
        losses_buffer.clear()
        collection_time_sum = 0.0

    self.env.close()

  def _phase2_learning_step(
    self,
    obs_dict: ObsType,
  ) -> tuple[ObsType, dict[str, float]]:
    """Collect aligned (priv_obs, adapt_obs) pairs then run MSE regression.

    Collection and learning are coupled here so that each adapt_obs snapshot
    is taken from the same physics state as its paired priv_obs — fixing the
    temporal misalignment that broke the previous rollout-buffer approach.

    Args:
      obs_dict: Current observation dict (maintained across iterations).

    Returns:
      (updated obs_dict, metrics with "adapt/loss")
    """
    assert isinstance(self.config, PpoConfig)

    priv_obs_list: list[dict[str, torch.Tensor]] = []
    adapt_obs_list: list[dict[str, torch.Tensor]] = []
    adapt_mask_list: list[torch.Tensor | None] = []

    # --- Collection phase (all networks frozen) ---
    with torch.no_grad():
      for _ in range(self.config.num_steps_per_env):
        actor_obs = self.get_actor_obs(obs_dict)
        norm_actor_obs = self.actor_obs_normalizer(actor_obs)
        priv_obs = self.get_privileged_obs(obs_dict)
        actor_input = self._compose_actor_input(norm_actor_obs, priv_obs)
        actions = self.actor.act_inference(actor_input)

        obs_dict, _, _, _, _ = self.env.step(actions)
        # env.step() calls rma_manager.update() → depth buffers and FOV state are current

        # Snapshot aligned pairs from this physics state
        priv_obs_list.append(self.get_privileged_obs(obs_dict))
        adapt_obs_list.append(self.rma_manager.get_adaptation_obs())
        adapt_mask_list.append(self.rma_manager.get_adaptation_mask())

    # --- Learning phase (only adaptation encoder params have grad) ---
    T = len(priv_obs_list)
    total_size = T * self.env.num_envs

    # Stack list-of-dicts → dict of (T*N, ...) flat tensors
    stacked_priv: dict[str, torch.Tensor] = {
      key: torch.stack([p[key] for p in priv_obs_list], dim=0).flatten(0, 1)
      for key in priv_obs_list[0]
    }
    stacked_adapt: dict[str, torch.Tensor] = {
      key: torch.stack([a[key] for a in adapt_obs_list], dim=0).flatten(0, 1)
      for key in adapt_obs_list[0]
    }

    # Stack per-step masks → (T*N,) bool, or None if no term defines a mask
    stacked_mask: torch.Tensor | None = None
    if any(m is not None for m in adapt_mask_list):
      stacked_mask = torch.stack([
        m if m is not None
        else torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        for m in adapt_mask_list
      ]).flatten(0, 1)  # (T*N,)

    valid_frac = stacked_mask.float().mean().item() if stacked_mask is not None else 1.0

    mini_batch_size = total_size // self.config.num_mini_batches
    total_loss = 0.0
    num_updates = 0

    for _ in range(self.config.num_learning_epochs):
      indices = torch.randperm(total_size, device=self.device)
      for i in range(self.config.num_mini_batches):
        start = i * mini_batch_size
        batch_idx = indices[start : start + mini_batch_size]
        batch_mask = stacked_mask[batch_idx] if stacked_mask is not None else None

        metrics = self.adaptation_learning_step(
          privileged_obs={k: v[batch_idx] for k, v in stacked_priv.items()},
          adaptation_obs={k: v[batch_idx] for k, v in stacked_adapt.items()},
          mask=batch_mask,
        )
        total_loss += metrics["adapt/loss"]
        num_updates += 1

    return obs_dict, {
      "adapt/loss": total_loss / max(num_updates, 1),
      "adapt/valid_frac": valid_frac,
    }

  # ------------------------------------------------------------------
  # Phase 2 — adaptation training setup
  # ------------------------------------------------------------------

  def build_adaptation_optimizer(self, lr: float = 1e-3) -> None:
    """Build a separate optimizer for adaptation encoders (call before Phase 2).

    Freezes actor, critic, and privileged encoders. Only adaptation encoder
    parameters are left trainable.
    """
    # Freeze everything trained in Phase 1
    for p in self.actor.parameters():
      p.requires_grad_(False)
    for p in self.value_net.parameters():
      p.requires_grad_(False)
    for p in self.rma_manager.privileged_parameters():
      p.requires_grad_(False)

    adapt_params = list(self.rma_manager.adaptation_parameters())
    if not adapt_params:
      raise RuntimeError(
        "No adaptation parameters found. Make sure adaptation_encoder is set "
        "on at least one RmaTerm."
      )
    self._adaptation_optimizer = optim.Adam(adapt_params, lr=lr)

  def adaptation_learning_step(
    self,
    privileged_obs: dict[str, torch.Tensor],
    adaptation_obs: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
  ) -> dict[str, float]:
    """Phase 2 learning step: MSE regression of adaptation encoder.

    Mirrors train2.py from TUM ADLR:
      z_target = frozen privileged_encoder(privileged_obs).detach()
      z_pred   = adaptation_encoder(adaptation_obs)
      loss     = MSE(z_pred[mask], z_target[mask])   # mask=None → all samples

    Args:
      privileged_obs: Dict of GT privileged groups (for frozen target).
      adaptation_obs: Dict of sensor groups (for trainable prediction).
      mask:           Optional (B,) bool — temporally-aligned validity mask
                      snapshotted at collection time. None = all samples valid.

    Returns:
      Metrics dict with "adapt/loss".
    """
    loss = self.rma_manager.compute_adaptation_loss(privileged_obs, adaptation_obs, mask=mask)

    self._adaptation_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
      self.rma_manager.adaptation_parameters(), max_norm=1.0
    )
    self._adaptation_optimizer.step()

    return {"adapt/loss": loss.item()}

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

    During training (_phase drives the training loop) this is always called
    with privileged_obs and runs phase=1. At inference time _inference_phase
    controls which encoder is used: 1=privileged (default), 2=adaptation.

    Args:
      actor_obs:      (N, actor_obs_dim) normalised proprioceptive obs.
      privileged_obs: Dict of GT privileged groups for the encoders.

    Returns:
      (N, actor_obs_dim + total_latent_dim) actor input tensor.
    """
    if self._inference_phase == 2 and not self.actor.training:
      adapt_obs = self.rma_manager.get_adaptation_obs()
      z = self.rma_manager.encode_phase2_with_fallback(privileged_obs, adapt_obs)
    else:
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

    metadata = checkpoint.get("metadata", {})
    if metadata.get("phase") == 2:
      self._inference_phase = 2

    return {
      "global_step": checkpoint["global_step"],
      "metadata": metadata,
      "config": checkpoint.get("config"),
    }
