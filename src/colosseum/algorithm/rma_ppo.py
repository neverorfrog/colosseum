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
from colosseum.algorithm.utils.rollout_buffer import RolloutBuffer
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
    from colosseum.algorithm.networks.ppo_networks import PpoActor

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
        norm_actor_obs = self._compose_actor_input(
          norm_actor_obs_base, current_privileged_obs
        )

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

    self.rollout_buffer.compute_returns_and_advantages(
      last_values=last_values,
      gamma=self.config.gamma,
      lam=self.config.lam,
      normalize_advantage=False,
    )

    if not self.config.normalize_advantage_per_mini_batch:
      self.rollout_buffer.advantages = self._normalize_advantages_multi_gpu(
        self.rollout_buffer.advantages
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
    """Collect aligned temporal sequences then run adaptation regression.

    Collection and learning are coupled here so each adaptation snapshot is
    temporally aligned with its privileged target.

    Args:
      obs_dict: Current observation dict (maintained across iterations).

    Returns:
      (updated obs_dict, adaptation metrics)
    """
    assert isinstance(self.config, PpoConfig)

    priv_obs_list: list[dict[str, torch.Tensor]] = []
    frame_list: list[torch.Tensor] = []
    reset_mask_list: list[torch.Tensor] = []
    fov_mask_list: list[torch.Tensor] = []

    if not self.rma_manager.adaptation_group_names:
      raise RuntimeError("Phase 2 requires at least one adaptation observation group.")
    frame_group = self.rma_manager.adaptation_group_names[0]

    warmup_steps = max(
      (
        getattr(getattr(term, "cfg", None), "warmup_steps", 0)
        for term in self.rma_manager._terms.values()
      ),
      default=0,
    )

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

        # Snapshot aligned pairs from this physics state.
        # Adaptation obs (depth frames) are offloaded to CPU immediately to
        # avoid accumulating several GiB on the GPU across num_steps_per_env.
        priv_obs_list.append(self.get_privileged_obs(obs_dict))
        adapt_obs = self.rma_manager.get_adaptation_obs()
        if frame_group not in adapt_obs:
          raise RuntimeError(
            f"Missing adaptation obs group '{frame_group}' during Phase 2 collection."
          )
        frame_list.append(adapt_obs[frame_group])

        reset_event = self.rma_manager.get_reset_event()
        if reset_event is None:
          reset_event = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.device
          )
        reset_mask_list.append(reset_event)

        fov_mask = self.rma_manager.get_adaptation_mask()
        if fov_mask is None:
          fov_mask = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        fov_mask_list.append(fov_mask)

    # --- Learning phase (only adaptation encoder params have grad) ---
    # Stack temporal tensors as (N, T, ...)
    stacked_priv: dict[str, torch.Tensor] = {
      key: torch.stack([p[key] for p in priv_obs_list], dim=1)
      for key in priv_obs_list[0]
    }
    frames = torch.stack(frame_list, dim=1)  # (N, T, 1, H, W)
    reset_mask = torch.stack(reset_mask_list, dim=1)  # (N, T)
    fov_mask = torch.stack(fov_mask_list, dim=1)  # (N, T)

    # Build warm-up mask: false for first K steps after reset/FOV re-entry.
    if warmup_steps > 0:
      since_reset = torch.zeros(
        self.env.num_envs,
        dtype=torch.long,
        device=self.device,
      )
      warm_mask_steps: list[torch.Tensor] = []
      for t in range(reset_mask.shape[1]):
        since_reset = torch.where(
          reset_mask[:, t],
          torch.zeros_like(since_reset),
          since_reset + 1,
        )
        warm_mask_steps.append(since_reset >= warmup_steps)
      warm_mask = torch.stack(warm_mask_steps, dim=1)
    else:
      warm_mask = torch.ones_like(reset_mask)

    loss_mask = fov_mask & warm_mask
    valid_frac = loss_mask.float().mean().item()

    # Keep sequence order intact; no shuffling across time.
    metrics_sum: dict[str, float] = defaultdict(float)
    num_updates = 0
    for _ in range(self.config.num_learning_epochs):
      metrics = self.adaptation_learning_step(
        privileged_obs=stacked_priv,
        adaptation_obs={
          frame_group: frames,
          "reset_mask": reset_mask,
          "loss_mask": loss_mask,
        },
        mask=loss_mask,
      )
      for key, value in metrics.items():
        metrics_sum[key] += value
      num_updates += 1

    result = {key: value / max(num_updates, 1) for key, value in metrics_sum.items()}
    result["adapt/valid_frac"] = valid_frac
    return obs_dict, result

  # ------------------------------------------------------------------
  # Phase 2 — adaptation training setup
  # ------------------------------------------------------------------

  def build_adaptation_optimizer(self, lr: float = 1e-3) -> None:
    """Build a separate optimizer for adaptation encoders (call before Phase 2).

    Freezes actor, critic, and privileged encoders. Only adaptation encoder
    parameters are left trainable. Puts adaptation encoders in train mode
    (critical for BatchNorm running stats to update).
    """
    # Freeze everything trained in Phase 1
    for p in self.actor.parameters():
      p.requires_grad_(False)
    for p in self.value_net.parameters():
      p.requires_grad_(False)
    for p in self.rma_manager.privileged_parameters():
      p.requires_grad_(False)

    # Ensure adaptation encoders are in train mode so BatchNorm stats update.
    # The privileged encoders stay frozen (eval mode via requires_grad=False,
    # but we also explicitly set them to eval to freeze BN running stats).
    for term in self.rma_manager._terms.values():
      term.privileged_encoder.eval()
      if term.adaptation_encoder is not None:
        term.adaptation_encoder.train()

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
    """Phase 2 learning step for adaptation encoders.

    Manager returns a dictionary of named losses. This method sums them for
    backward and returns both total and component metrics.

    Args:
      privileged_obs: Dict of GT privileged groups (for frozen target).
      adaptation_obs: Dict of sensor groups (for trainable prediction).
      mask:           Optional (B,) bool — temporally-aligned validity mask
                      snapshotted at collection time. None = all samples valid.

    Returns:
      Metrics dict with total and component losses.
    """
    loss_terms = self.rma_manager.compute_adaptation_loss(
      privileged_obs,
      adaptation_obs,
      mask=mask,
    )
    total_loss = torch.stack(list(loss_terms.values())).sum()

    self._adaptation_optimizer.zero_grad()
    total_loss.backward()
    self._distributed_average_optimizer_grads(self._adaptation_optimizer)
    torch.nn.utils.clip_grad_norm_(
      self.rma_manager.adaptation_parameters(), max_norm=1.0
    )
    self._adaptation_optimizer.step()

    metrics = {"adapt/loss": float(total_loss.item())}
    for key, value in loss_terms.items():
      metrics[f"adapt/{key}"] = float(value.item())

    if self.is_distributed:
      ordered_keys = sorted(metrics.keys())
      reduced_values = self._distributed_sum_vector([metrics[k] for k in ordered_keys])
      metrics = {
        key: reduced_values[i] / self.world_size for i, key in enumerate(ordered_keys)
      }

    return metrics

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
