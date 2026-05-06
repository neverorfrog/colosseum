"""Base algorithm class with checkpoint utilities.

Inspired by holosoma's BaseAlgo pattern, providing minimal save/load functionality
for RL algorithms.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils import spaces as mjlab_spaces

from colosseum.algorithm.utils.normalization import ObsNormalizer
from colosseum.utils.logger import log_training_step

if TYPE_CHECKING:
  from colosseum.config.types.algorithm import AlgorithmConfig

# Type alias for observation structure from mjlab environments
ObsType = Union[dict[str, Union[torch.Tensor, dict[str, torch.Tensor]]], torch.Tensor]


def cpu_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  """Move all tensors in state dict to CPU.

  Args:
      state_dict: State dictionary with tensors

  Returns:
      State dictionary with all tensors on CPU
  """
  return {
    k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
  }


class BaseAlgorithm(ABC):
  """Base class for RL algorithms with checkpoint functionality.

  Provides:
  - Abstract train() method that subclasses must implement
  - save() and load() interface for checkpointing
  - Metadata management for reproducibility
  - Clean checkpoint structure
  - Optional logging callback

  Subclasses must implement:
  - train() method (main training loop)
  - save() method (checkpoint saving)
  - load() method (checkpoint loading)

  Args:
      config: Algorithm configuration
      env: Environment instance
      device: Torch device for training (e.g., "cuda", "cpu")
      log_fn: Optional logging callback (metrics_dict, step) -> None
      log_interval: Logging cadence in environment steps
  """

  def __init__(
    self,
    config: "AlgorithmConfig",
    env: ManagerBasedRlEnv,
    device: str | torch.device,
    log_fn: Callable[[dict[str, float], int], None],
    log_interval: int,
  ) -> None:
    self.config = config
    self.device = torch.device(device) if isinstance(device, str) else device
    self.log_fn = log_fn
    self.log_interval = int(log_interval)
    self._metadata: dict[str, Any] = {}
    self.seed = config.seed
    self.actor: Any  # Set by subclasses (e.g., PPO._build_networks)
    self.actor_obs_normalizer: ObsNormalizer  # Set by subclasses
    self.critic_obs_normalizer: ObsNormalizer  # Set by subclasses

    # Instantiate environment using provided class
    self.env = env
    # Check if using Dict observations (mjlab-style)
    self.use_dict_obs = isinstance(self.env.single_observation_space, mjlab_spaces.Dict)

    # Episode tracking (universal for all episodic RL algorithms)
    self.latest_episode_metrics: dict[str, float] | None = None
    self.episode_lengths: deque[float] = deque(maxlen=100)  # Rolling average buffer
    self.start_time: float | None = None

    # Success rate tracking: rolling window over last 100 finished episodes
    # Each entry is True (success) or False (failure/timeout)
    self.episode_outcomes: deque[bool] = deque(maxlen=100)

    # Optional checkpointing configuration (set by runner)
    self._checkpoint_dir: Path | None = None
    self._save_interval: int | None = None
    self._last_save_step: int = -1
    self._last_saved_obstacle_stage_index: int | None = None

    # Optional evaluation configuration (set by training script)
    self._eval_env: ManagerBasedRlEnv | None = None
    self._eval_interval: int = 0
    self._eval_episodes: int = 10
    self._last_eval_step: int = 0

    # Global step counter (total env transitions across the full training run)
    self.global_step: int = 0

    # Distributed training state (set by torchrun process env).
    self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    self.rank = int(os.environ.get("RANK", "0"))
    self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    self.is_distributed = (
      self.world_size > 1
      and torch.distributed.is_available()
      and torch.distributed.is_initialized()
    )
    self.is_main_process = self.rank == 0

    # Setting the seed
    self._maybe_seed()

  def get_actor_obs(self, obs: ObsType) -> torch.Tensor:
    """Extract actor observations from obs dict (rsl_rl style).

    Args:
        obs: Either a dict with "actor" key or a flat tensor

    Returns:
        Tensor of shape (num_envs, actor_obs_dim)
    """
    if self.use_dict_obs:
      assert isinstance(obs, dict), "Expected dict observations but got tensor"
      policy_obs = obs["actor"]
      assert isinstance(policy_obs, torch.Tensor), "Expected actor obs to be a tensor"
      return policy_obs
    assert isinstance(obs, torch.Tensor), "Expected tensor observations but got dict"
    return obs

  def get_critic_obs(self, obs: ObsType) -> torch.Tensor:
    """Extract critic observations from obs dict (rsl_rl style).

    Args:
        obs: Either a dict with "critic" key or a flat tensor

    Returns:
        Tensor of shape (num_envs, critic_obs_dim)
    """
    if self.use_dict_obs:
      assert isinstance(obs, dict), "Expected dict observations but got tensor"
      critic_obs = obs["critic"]
      assert isinstance(critic_obs, torch.Tensor), "Expected critic obs to be a tensor"
      return critic_obs
    assert isinstance(obs, torch.Tensor), "Expected tensor observations but got dict"
    return obs

  def _unwrap_env(self):
    """Get underlying environment (handles wrappers)."""
    return getattr(self, "unwrapped_env", self.env)

  def update_episode_counts(
    self,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    success_term_name: str = "arrived_at_goal",
  ) -> None:
    """Update rolling success rate for episodes that ended this step.

    For each finished episode, appends True (success) or False (failure/timeout)
    to a rolling deque of the last 100 episodes.

    Success is determined by checking whether the specific termination term
    (e.g., "arrived_at_goal") fired, rather than treating all non-timeout
    terminations as success.

    Args:
        terminated: Boolean tensor (num_envs,) - True if episode terminated
        truncated: Boolean tensor (num_envs,) - True if episode timed out
        success_term_name: Name of the termination term that indicates success

    Example:
        >>> obs, rewards, terminated, truncated, infos = self.env.step(actions)
        >>> self.update_episode_counts(terminated, truncated)
    """
    dones = terminated | truncated
    num_done = dones.sum().item()
    if num_done == 0:
      return

    # Check if the success termination term exists and fired
    unwrapped = getattr(self.env, "unwrapped", self.env)
    term_mgr = getattr(unwrapped, "termination_manager", None)
    if term_mgr is not None and success_term_name in term_mgr.active_terms:
      success_flags = term_mgr.get_term(success_term_name)  # [num_envs] bool
    else:
      # Fallback: any non-timeout termination counts as success
      success_flags = terminated & ~truncated

    # Append one outcome per finished episode
    done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
    for idx in done_indices:
      self.episode_outcomes.append(bool(success_flags[idx].item()))

  def _log_training_metrics(
    self,
    step: int,
    losses_buffer: dict[str, list[float]],
    collection_time: float,
    learning_time: float,
    log_interval: int,
    total_timesteps: int,
    title: str = "Training",
    use_rich: bool = True,
    steps_per_log_step: int = 1,
  ) -> None:
    """Log comprehensive training metrics to W&B and Rich panel.

    Generic implementation that works for all algorithms.
    Algorithms are responsible for:
    - Accumulating losses in losses_buffer
    - Updating self.latest_episode_metrics from env infos
    - Updating self.episode_lengths when episodes complete

    Args:
        step: Current training step
        losses_buffer: Accumulated losses over log_interval
        collection_time: Total collection phase time (seconds)
        learning_time: Total learning phase time (seconds)
        log_interval: Number of steps between logs
        total_timesteps: Total training steps
        title: Title for console output (e.g., "SAC Training")
        use_rich: Whether to use Rich console output
        steps_per_log_step: Env steps per log step (1 for SAC, num_steps_per_env for PPO)
    """
    # Only rank 0 emits logs/metrics in distributed mode.
    if not self.is_main_process:
      return

    # Average accumulated losses
    avg_losses: dict[str, float] = {
      k: float(np.mean(v)) for k, v in losses_buffer.items() if len(v) > 0
    }

    # Time metrics
    assert self.start_time is not None, "start_time must be set before logging"

    # Total elapsed time since training started
    elapsed_time = time.time() - self.start_time

    # Steps per second (overall average)
    sps = int(step / elapsed_time) if elapsed_time > 0 else 0

    # Interval metrics (for this logging period)
    interval_time = collection_time + learning_time
    fps = (
      (self.env.num_envs * log_interval * steps_per_log_step) / interval_time
      if interval_time > 0
      else 0.0
    )

    # Prepare episode metrics for display
    episode_metrics_for_display = None
    curriculum_step, obstacle_stage_progress_pct = self._get_curriculum_progress_info()
    if self.latest_episode_metrics is not None:
      episode_metrics_for_display = self.latest_episode_metrics.copy()

      # Mean/Reward: prefer rolling average from algorithm (e.g. PPO's rewbuffer)
      # if the algorithm already added it to losses_buffer; otherwise fall back
      # to the snapshot sum of per-step reward components from infos["log"].
      if "mean_reward" in avg_losses:
        episode_metrics_for_display["Mean/Reward"] = avg_losses["mean_reward"]
      else:
        episode_metrics_for_display["Mean/Reward"] = sum(
          v
          for k, v in self.latest_episode_metrics.items()
          if k.startswith("Episode_Reward/")
        )

      # Add mean episode length from tracked completed episodes
      if len(self.episode_lengths) > 0:
        episode_metrics_for_display["Mean/Episode_Length"] = float(
          np.mean(self.episode_lengths)
        )

    # Logging architecture:
    # - log_fn: Logs to W&B (train/ and episode metrics with their original prefixes)
    # - log_training_step: Rich console output only (no W&B to avoid duplication)
    assert self.log_fn is not None, "log_fn must be provided for logging"

    # Log training metrics to W&B
    train_metrics = {f"train/{k}": v for k, v in avg_losses.items()}
    train_metrics["time/interval_collection"] = (
      collection_time  # Time for this interval
    )
    train_metrics["time/interval_learning"] = learning_time  # Time for this interval
    train_metrics["time/interval_total"] = interval_time  # Time for this interval
    train_metrics["time/elapsed"] = elapsed_time  # Total time since start
    train_metrics["time/sps"] = sps  # Overall steps/sec
    train_metrics["time/fps"] = fps  # Interval frames/sec
    self.log_fn(train_metrics, step)

    # Log episode metrics to W&B
    if self.latest_episode_metrics is not None:
      self.log_fn(self.latest_episode_metrics, step)
    if episode_metrics_for_display is not None:
      self.log_fn(episode_metrics_for_display, step)

    # Log rolling success rate (last 100 episodes)
    if len(self.episode_outcomes) > 0:
      success_rate = sum(self.episode_outcomes) / len(self.episode_outcomes)
      self.log_fn(
        {
          "Episode_Success/rate": success_rate,
        },
        step,
      )

    # Console output (Rich or loguru) - no W&B logging
    log_training_step(
      step=step,
      total_steps=total_timesteps,
      loss_dict=avg_losses,
      episode_metrics=episode_metrics_for_display,
      phase=self._metadata.get("phase"),
      curriculum_step=curriculum_step,
      obstacle_stage_progress_pct=obstacle_stage_progress_pct,
      collection_time=collection_time,
      learning_time=learning_time,
      elapsed_time=elapsed_time,
      num_envs=self.env.num_envs,
      log_interval=log_interval,
      title=title,
      use_rich=self.config.use_rich_logging,
      steps_per_log_step=steps_per_log_step,
    )

  @abstractmethod
  def train(self) -> None:
    """Main training loop."""
    pass

  def attach_metadata(self, **kwargs: Any) -> None:
    """Attach metadata that will be saved with checkpoints."""
    self._metadata.update(kwargs)

  def _maybe_seed(self) -> None:
    if self.seed is None:
      return
    np.random.seed(self.seed)
    torch.manual_seed(self.seed)
    torch.backends.cudnn.deterministic = True

  # --- Checkpointing helpers configured by the training driver ---
  def configure_checkpointing(
    self, checkpoint_dir: str | Path, save_interval: int | None
  ) -> None:
    """Enable periodic checkpoint saving during training."""
    self._checkpoint_dir = Path(checkpoint_dir)
    if save_interval is None or save_interval <= 0:
      self._save_interval = None
    else:
      self._save_interval = int(save_interval)
    self._last_save_step = 0
    self._last_saved_obstacle_stage_index = None

  def _maybe_save_checkpoint(self, step: int) -> None:
    """Save periodic checkpoints and stage-transition checkpoints when due."""
    if self._checkpoint_dir is None:
      return

    stage_index, stage_name = self._get_current_obstacle_stage_info()

    if self._last_saved_obstacle_stage_index is None:
      self._last_saved_obstacle_stage_index = stage_index

    if stage_index is not None and stage_index != self._last_saved_obstacle_stage_index:
      self._save_stage_checkpoint(step, stage_index, stage_name)
      self._last_saved_obstacle_stage_index = stage_index

    if self._save_interval is None or self._save_interval <= 0:
      return

    due = (step - self._last_save_step) >= self._save_interval
    if not due:
      return

    ckpt_dir = self._checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    phase_index = self._get_current_phase_index()
    ckpt_path = ckpt_dir / self._format_checkpoint_name(
      step=step,
      phase_index=phase_index,
      stage_index=stage_index,
    )
    try:
      if stage_index is not None:
        self._metadata["current_obstacle_stage_index"] = stage_index
      if stage_name is not None:
        self._metadata["current_obstacle_stage_name"] = stage_name
      self.save(
        ckpt_path,
        global_step=step,
        obstacle_stage_index=stage_index,
        obstacle_stage_name=stage_name,
      )
      self._last_save_step = step

      # Maintain a 'latest.pt' symlink for easy discovery
      latest_link = ckpt_dir / "latest.pt"
      try:
        if latest_link.exists() or latest_link.is_symlink():
          latest_link.unlink()
        latest_link.symlink_to(ckpt_path.name)
      except Exception:
        # Fallback: copy if symlink not permitted
        try:
          import shutil

          shutil.copy2(ckpt_path, latest_link)
        except Exception:
          pass

    except Exception as e:
      logger.error(f"Failed to save checkpoint at step {step}: {e}")

  def _get_current_obstacle_stage_info(self) -> tuple[int | None, str | None]:
    """Resolve the currently active obstacle stage from pinning or curriculum."""
    stage_index: int | None = None
    stage_name: str | None = None

    forced_stage = self._metadata.get("obstacle_stage_index")
    if isinstance(forced_stage, (int, float)) and int(forced_stage) >= 0:
      stage_index = int(forced_stage)

    if stage_index is None:
      curriculum_cfg = getattr(getattr(self.env, "cfg", None), "curriculum", None)
      if isinstance(curriculum_cfg, dict):
        obstacle_term = curriculum_cfg.get("obstacle")
        obstacle_params = getattr(obstacle_term, "params", None)
        stages = (
          obstacle_params.get("stages") if isinstance(obstacle_params, dict) else None
        )
        if isinstance(stages, list) and stages:
          common_step = int(getattr(self.env.unwrapped, "common_step_counter", 0))
          stage_index = 0
          for idx, stage in enumerate(stages):
            if common_step >= stage["step"]:
              stage_index = idx
              stage_name = stage.get("behavior")

    if stage_index is None and self.latest_episode_metrics is not None:
      metric_stage = self.latest_episode_metrics.get("Curriculum/obstacle_stage_index")
      if metric_stage is not None:
        stage_index = int(round(metric_stage))

    if stage_index is None:
      return None, None

    if stage_name is None:
      stage_name = {
        0: "none",
        1: "static_blocker",
        2: "lateral_blocker",
        3: "ball_attacker",
        4: "mixed_attackers",
      }.get(stage_index, "unknown")

    return stage_index, stage_name

  def _get_current_phase_index(self) -> int | None:
    """Return the current training phase when available."""
    phase = self._metadata.get("phase")
    if isinstance(phase, (int, float)):
      return int(phase)
    return None

  def _format_checkpoint_name(
    self,
    step: int,
    phase_index: int | None,
    stage_index: int | None,
  ) -> str:
    """Build a checkpoint filename with step and obstacle stage."""
    parts = [f"model_{step:07d}"]
    if stage_index is not None:
      parts.append(f"stage{stage_index}")
    return "_".join(parts) + ".pt"

  def _get_curriculum_progress_info(self) -> tuple[int | None, float | None]:
    """Return current curriculum step and current obstacle-stage completion."""
    common_step = getattr(self.env.unwrapped, "common_step_counter", None)
    if common_step is None:
      return None, None

    curriculum_step = int(common_step)
    curriculum_cfg = getattr(getattr(self.env, "cfg", None), "curriculum", None)
    if not isinstance(curriculum_cfg, dict):
      return curriculum_step, None

    obstacle_term = curriculum_cfg.get("obstacle")
    obstacle_params = getattr(obstacle_term, "params", None)
    stages = (
      obstacle_params.get("stages") if isinstance(obstacle_params, dict) else None
    )
    if not isinstance(stages, list) or not stages:
      return curriculum_step, None

    stage_index, _ = self._get_current_obstacle_stage_info()
    if stage_index is None or stage_index < 0 or stage_index >= len(stages):
      return curriculum_step, None

    stage_start = int(stages[stage_index]["step"])
    if stage_index + 1 >= len(stages):
      return curriculum_step, 100.0

    next_stage_start = int(stages[stage_index + 1]["step"])
    if next_stage_start <= stage_start:
      return curriculum_step, 100.0

    progress_pct = (
      100.0 * (curriculum_step - stage_start) / (next_stage_start - stage_start)
    )
    progress_pct = float(np.clip(progress_pct, 0.0, 100.0))
    return curriculum_step, progress_pct

  def _save_stage_checkpoint(
    self, step: int, stage_index: int, stage_name: str | None
  ) -> None:
    """Save/update the per-stage checkpoint when entering a new obstacle stage."""
    if self._checkpoint_dir is None:
      return

    ckpt_dir = self._checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    phase_index = self._get_current_phase_index()
    stage_parts = [f"latest_stage_{stage_index}"]
    if phase_index is not None:
      stage_parts.append(f"phase{phase_index}")
    ckpt_path = ckpt_dir / ("_".join(stage_parts) + ".pt")

    try:
      self._metadata["current_obstacle_stage_index"] = stage_index
      if stage_name is not None:
        self._metadata["current_obstacle_stage_name"] = stage_name
      self.save(
        ckpt_path,
        global_step=step,
        obstacle_stage_index=stage_index,
        obstacle_stage_name=stage_name,
      )

      logger.info(
        f"Saved stage-transition checkpoint: {ckpt_path.name} "
        f"(stage {stage_index}: {stage_name}, step {step})"
      )
    except Exception as e:
      logger.error(
        f"Failed to save stage checkpoint for obstacle stage {stage_index} at step {step}: {e}"
      )

  # --- Evaluation helpers configured by the training driver ---
  def configure_evaluation(
    self, eval_env: ManagerBasedRlEnv, eval_interval: int, eval_episodes: int
  ) -> None:
    """Enable periodic evaluation during training."""
    self._eval_env = eval_env
    self._eval_interval = eval_interval
    self._eval_episodes = eval_episodes
    self._last_eval_step = 0

  def _maybe_evaluate(self, step: int) -> None:
    """Run evaluation if configured and due at this step."""
    if not self.is_main_process:
      return

    if self._eval_env is None or self._eval_interval <= 0:
      return

    due = (step - self._last_eval_step) >= self._eval_interval
    if not due:
      return

    self._last_eval_step = step
    eval_metrics = self.evaluate(self._eval_env, self._eval_episodes)
    self.log_fn(eval_metrics, step)
    logger.info(
      f"[Eval @ step {step}] "
      + ", ".join(f"{k}: {v:.3f}" for k, v in eval_metrics.items())
    )

  def _eval_get_action(self, normalized_obs: torch.Tensor) -> torch.Tensor:
    """Get deterministic action for evaluation. Override in subclasses if needed."""
    actions, _, _ = self.actor.get_action(normalized_obs)
    return actions

  def _prepare_eval_actor_obs(
    self, actor_obs: torch.Tensor, eval_env: ManagerBasedRlEnv
  ) -> torch.Tensor:
    """Transform raw actor obs before normalization during evaluation."""
    return actor_obs

  def evaluate(
    self,
    eval_env: ManagerBasedRlEnv,
    num_episodes: int,
    success_term_name: str = "arrived_at_goal",
  ) -> dict[str, float]:
    """Run deterministic evaluation episodes and return metrics."""
    # Switch to eval mode
    was_training = self.actor.training
    self.actor.eval()
    self.actor_obs_normalizer.eval()

    successes: list[bool] = []
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []

    obs_dict, _ = eval_env.reset()
    actor_obs = self.get_actor_obs(obs_dict)

    current_ep_length = torch.zeros(eval_env.num_envs, device=self.device)
    current_ep_reward = torch.zeros(eval_env.num_envs, device=self.device)

    while len(successes) < num_episodes:
      with torch.no_grad():
        actor_obs = self._prepare_eval_actor_obs(actor_obs, eval_env)
        normalized_obs = self.actor_obs_normalizer(actor_obs)
        actions = self._eval_get_action(normalized_obs)

      obs_dict, rewards, terminated, truncated, infos = eval_env.step(actions)
      actor_obs = self.get_actor_obs(obs_dict)

      current_ep_length += 1
      current_ep_reward += rewards.squeeze(-1) if rewards.dim() > 1 else rewards

      dones = terminated | truncated
      if dones.any():
        unwrapped = getattr(eval_env, "unwrapped", eval_env)
        term_mgr = getattr(unwrapped, "termination_manager", None)
        if term_mgr is not None and success_term_name in term_mgr.active_terms:
          success_flags = term_mgr.get_term(success_term_name)
        else:
          success_flags = terminated & ~truncated

        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_indices:
          if len(successes) >= num_episodes:
            break
          successes.append(bool(success_flags[idx].item()))
          episode_lengths.append(int(current_ep_length[idx].item()))
          episode_rewards.append(float(current_ep_reward[idx].item()))

        current_ep_length[dones] = 0
        current_ep_reward[dones] = 0

    # Restore training mode
    if was_training:
      self.actor.train()
      self.actor_obs_normalizer.train()

    success_rate = sum(successes) / len(successes) if successes else 0.0
    mean_ep_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    mean_ep_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0

    return {
      "eval/success_rate": success_rate,
      "eval/mean_episode_length": mean_ep_length,
      "eval/mean_reward": mean_ep_reward,
    }

  def export_onnx(self, path: str | Path) -> Path:
    """Export the actor (with observation normalizer) to ONNX.

    The exported model takes a single input "obs" and produces "actions".

    Args:
        path: Output path for the .onnx file.

    Returns:
        Resolved Path to the written ONNX file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    was_training = self.actor.training
    self.actor.eval()
    self.actor_obs_normalizer.eval()

    wrapper = nn.Sequential(self.actor_obs_normalizer, self.actor).cpu()
    wrapper.eval()
    obs_dim = self.env.observation_manager.group_obs_dim["actor"][0]
    dummy = torch.zeros(1, obs_dim)

    torch.onnx.export(
      wrapper,
      (dummy,),
      str(path),
      export_params=True,
      opset_version=18,
      input_names=["obs"],
      output_names=["actions"],
    )

    if was_training:
      self.actor.train()
      self.actor_obs_normalizer.train()
    # Move back to original device
    self.actor.to(self.device)
    self.actor_obs_normalizer.to(self.device)

    logger.success(f"ONNX exported: {path}")
    return path

  def _restore_env_step_counter(self) -> None:
    """Restore env.common_step_counter from global_step after loading a checkpoint.

    common_step_counter increments once per env.step() call (i.e. once per
    policy step across all parallel envs), so it equals global_step / num_envs.
    Without this, curriculum stages restart from 0 on every resume.
    """
    common_step = self.global_step // self.env.num_envs
    self.env.unwrapped.common_step_counter = common_step
    stage_index, stage_name = self._get_current_obstacle_stage_info()
    self._last_saved_obstacle_stage_index = stage_index
    if stage_index is not None:
      self._metadata["current_obstacle_stage_index"] = stage_index
    if stage_name is not None:
      self._metadata["current_obstacle_stage_name"] = stage_name
    logger.info(
      f"Restored common_step_counter={common_step} from global_step={self.global_step}"
    )

  @abstractmethod
  def save(self, path: str | Path, **extra_state: Any) -> None:
    """Save checkpoint to disk."""
    pass

  @abstractmethod
  def load(self, path: str | Path) -> dict[str, Any]:
    """Load checkpoint from disk."""
    pass

  def _save_checkpoint(
    self,
    path: str | Path,
    state_dict: dict[str, Any],
  ) -> None:
    """Internal helper to save checkpoint with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if self._metadata:
      state_dict["metadata"] = self._metadata

    algo_name = getattr(self, "_algo_name", None)
    if algo_name is not None:
      state_dict.setdefault("algo_name", algo_name)

    cpu_dict = {}
    for key, value in state_dict.items():
      if isinstance(value, dict):
        cpu_dict[key] = (
          cpu_state(value)
          if any(isinstance(v, torch.Tensor) for v in value.values())
          else value
        )
      elif isinstance(value, torch.Tensor):
        cpu_dict[key] = value.cpu()
      else:
        cpu_dict[key] = value

    torch.save(cpu_dict, path)
    logger.success(f"Checkpoint saved: {path}")

  def _distributed_average_optimizer_grads(
    self, optimizer: torch.optim.Optimizer
  ) -> None:
    """Average gradients across all distributed workers.

    This implements synchronous data-parallel optimization without wrapping the
    model in DDP, which keeps custom actor methods and checkpoint format intact.
    """
    if not self.is_distributed:
      return

    seen: set[int] = set()
    for group in optimizer.param_groups:
      for param in group["params"]:
        if param is None or param.grad is None:
          continue
        param_id = id(param)
        if param_id in seen:
          continue
        seen.add(param_id)

        torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
        param.grad.data.div_(self.world_size)

  def _distributed_mean_scalar(self, value: float) -> float:
    """Compute cross-rank mean for a scalar float."""
    if not self.is_distributed:
      return value

    tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= self.world_size
    return float(tensor.item())

  def _distributed_sum_vector(self, values: list[float]) -> list[float]:
    """Compute cross-rank sum for a small float vector."""
    if not self.is_distributed:
      return values

    tensor = torch.tensor(values, device=self.device, dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return [float(v) for v in tensor.tolist()]

  def _load_checkpoint(
    self,
    path: str | Path,
    map_location: str | torch.device | None = None,
  ) -> dict[str, Any]:
    """Internal helper to load checkpoint."""
    path = Path(path)

    if not path.exists():
      raise FileNotFoundError(f"Checkpoint not found: {path}")

    if map_location is None:
      map_location = self.device

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    logger.success(f"Checkpoint loaded: {path}")

    return checkpoint


def get_latest_checkpoint(
  checkpoint_dir: str | Path, pattern: str = "*.pt"
) -> Path | None:
  """Find the most recent checkpoint in a directory."""
  checkpoint_dir = Path(checkpoint_dir)

  if not checkpoint_dir.exists():
    return None

  checkpoints = list(checkpoint_dir.glob(pattern))

  if not checkpoints:
    return None

  latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
  return latest


def cleanup_old_checkpoints(
  checkpoint_dir: str | Path,
  keep_last_n: int = 5,
  pattern: str = "*.pt",
) -> None:
  """Remove old checkpoints, keeping only the N most recent."""
  checkpoint_dir = Path(checkpoint_dir)

  if not checkpoint_dir.exists():
    return

  checkpoints = sorted(
    checkpoint_dir.glob(pattern),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
  )

  if len(checkpoints) <= keep_last_n:
    return

  for ckpt_path in checkpoints[keep_last_n:]:
    ckpt_path.unlink()

  logger.info(
    f"Checkpoint cleanup: kept {keep_last_n}, removed {len(checkpoints) - keep_last_n}"
  )
