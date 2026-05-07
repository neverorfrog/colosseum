"""Colosseum-local curriculum functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import CurriculumTermCfg


# ---------------------------------------------------------------------------
# Helpers for transition-based stage curricula
# ---------------------------------------------------------------------------

_TRANSITION_RESERVED = {"transitions", "params"}


def _validate_transition_stages(
  term_cfg: Any,
  term_name: str,
  stages: list[dict[str, Any]],
) -> None:
  """Validate ordering and field/param keys for transition-keyed stage lists."""
  for i in range(1, len(stages)):
    if stages[i]["transitions"] < stages[i - 1]["transitions"]:
      raise ValueError(
        f"Curriculum stages for '{term_name}' must be in nondecreasing "
        f"transition order, but stage {i} has transitions "
        f"{stages[i]['transitions']} < {stages[i - 1]['transitions']}."
      )
  for stage in stages:
    for key in stage:
      if key not in _TRANSITION_RESERVED and not hasattr(term_cfg, key):
        raise AttributeError(
          f"Field '{key}' does not exist on the term config for '{term_name}'."
        )
  term_params = getattr(term_cfg, "params", {})
  for stage in stages:
    unknown = stage.get("params", {}).keys() - term_params.keys()
    if unknown:
      raise KeyError(
        f"Stage at transitions={stage['transitions']} sets unknown param(s) "
        f"{unknown} on term '{term_name}'. Check for typos."
      )


def _apply_transition_stages(
  term_cfg: Any,
  total_transitions: int,
  stages: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
  """Apply all stages whose threshold has been reached; return a log snapshot."""
  for stage in stages:
    if total_transitions >= stage["transitions"]:
      for key, value in stage.items():
        if key not in _TRANSITION_RESERVED:
          setattr(term_cfg, key, value)
      if "params" in stage and hasattr(term_cfg, "params"):
        term_cfg.params.update(stage["params"])

  logged_fields: set[str] = set()
  logged_params: set[str] = set()
  for stage in stages:
    for key in stage:
      if key not in _TRANSITION_RESERVED:
        logged_fields.add(key)
    for key in stage.get("params", {}):
      logged_params.add(key)

  term_params = getattr(term_cfg, "params", {})
  result: dict[str, torch.Tensor] = {}
  for key in logged_fields:
    v = getattr(term_cfg, key)
    if isinstance(v, (int, float, bool)):
      result[key] = torch.tensor(v)
    elif isinstance(v, torch.Tensor):
      result[key] = v
  for key in logged_params:
    v = term_params[key]
    if isinstance(v, (int, float, bool)):
      result[key] = torch.tensor(v)
    elif isinstance(v, torch.Tensor):
      result[key] = v
  return result


# ---------------------------------------------------------------------------
# standing_curriculum_by_transitions
# ---------------------------------------------------------------------------


class standing_curriculum_by_transitions:
  """Schedule command-term attributes over total agent transitions.

  Identical structure to reward_curriculum_by_transitions but targets a
  command term instead of a reward term.  Typical use: ramp down
  rel_standing_envs as training progresses.

  Example::

    CurriculumTermCfg(
      func=standing_curriculum_by_transitions,
      params={
        "command_name": "twist",
        "stages": [
          {"transitions": 0,            "rel_standing_envs": 0.9},
          {"transitions": 50_000_000,   "rel_standing_envs": 0.5},
          {"transitions": 100_000_000,  "rel_standing_envs": 0.1},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    command_name: str = cfg.params["command_name"]
    self._stages: list[dict[str, Any]] = cfg.params["stages"]
    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found."
    self._term_cfg = command_term.cfg
    self._num_envs = env.num_envs
    _validate_transition_stages(self._term_cfg, command_name, self._stages)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    stages: list[dict[str, Any]],
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, stages
    total = env.common_step_counter * self._num_envs
    return _apply_transition_stages(self._term_cfg, total, self._stages)


# ---------------------------------------------------------------------------
# reward_curriculum_by_transitions
# ---------------------------------------------------------------------------


class reward_curriculum_by_transitions:
  """Like mjlab's reward_curriculum but stage thresholds are total agent transitions.

  Using agent transitions (common_step_counter * num_envs) instead of raw env
  steps makes the schedule invariant to num_envs: the same config file produces
  the same curriculum whether you train with 4096 or 20 000 parallel envs.

  Example::

    CurriculumTermCfg(
      func=reward_curriculum_by_transitions,
      params={
        "reward_name": "body_ang_vel",
        "stages": [
          {"transitions": 0,           "weight": -0.05},
          {"transitions": 50_000_000,  "weight": -0.3},
          {"transitions": 100_000_000, "weight": -1.0},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    reward_name: str = cfg.params["reward_name"]
    self._stages: list[dict[str, Any]] = cfg.params["stages"]
    self._term_cfg = env.reward_manager.get_term_cfg(reward_name)
    self._num_envs = env.num_envs
    _validate_transition_stages(self._term_cfg, reward_name, self._stages)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    stages: list[dict[str, Any]],
  ) -> dict[str, torch.Tensor]:
    del env_ids, reward_name, stages
    total = env.common_step_counter * self._num_envs
    return _apply_transition_stages(self._term_cfg, total, self._stages)


# ---------------------------------------------------------------------------
# penalty_curriculum  (episode-length adaptive)
# ---------------------------------------------------------------------------


class penalty_curriculum:
  """Adaptive penalty curriculum: scale reward weights based on average episode length.

  When episodes are short (robot falls quickly) all named penalties are scaled
  down so the robot can first learn to survive. As episodes grow longer the
  scale is raised, gradually tightening the behavioural requirements.

  Scale update rule (called each time any env resets):
    - avg < level_down_threshold  →  scale *= (1 - degree)
    - avg > level_up_threshold    →  scale *= (1 + degree)
    - otherwise                   →  scale unchanged
    Scale is clamped to [min_scale, max_scale] after each update.

  The running average uses an EMA whose update weight is proportional to the
  fraction of envs that reset, making it invariant to num_envs.

  Example::

    CurriculumTermCfg(
      func=penalty_curriculum,
      params={
        "reward_names": ["body_ang_vel", "action_rate_l2", "dof_pos_limits"],
        "initial_scale":         0.1,
        "min_scale":             0.0,
        "max_scale":             1.0,
        "level_down_threshold":  150.0,   # policy steps
        "level_up_threshold":    750.0,   # policy steps
        "degree":                0.05,
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._env = env

    reward_names: list[str] = params["reward_names"]
    self._initial_scale = float(params.get("initial_scale", 0.1))
    self._min_scale = float(params.get("min_scale", 0.0))
    self._max_scale = float(params.get("max_scale", 1.0))
    self._level_down_threshold = float(params.get("level_down_threshold", 150.0))
    self._level_up_threshold = float(params.get("level_up_threshold", 750.0))
    self._degree = float(params.get("degree", 0.05))

    # Hold live references; store original weights to allow correct rescaling.
    self._original_weights: dict[str, float] = {}
    self._reward_cfgs: dict[str, Any] = {}
    for name in reward_names:
      term_cfg = env.reward_manager.get_term_cfg(name)
      self._original_weights[name] = float(term_cfg.weight)
      self._reward_cfgs[name] = term_cfg

    self._current_scale = self._initial_scale
    self._avg_episode_length = 0.0
    self._apply_scale()

  # ------------------------------------------------------------------

  def _apply_scale(self) -> None:
    for name, term_cfg in self._reward_cfgs.items():
      term_cfg.weight = self._original_weights[name] * self._current_scale

  def reset(self, env_ids: torch.Tensor) -> None:
    """Called by CurriculumManager before episode_length_buf is zeroed."""
    if env_ids is None or len(env_ids) == 0:
      return

    # Read lengths of the just-finished episodes.
    lengths = self._env.episode_length_buf[env_ids].float()
    batch_avg = float(lengths.mean().item())

    # EMA: update weight = fraction of envs that just reset (num_envs-invariant).
    alpha = min(len(env_ids) / self._env.num_envs, 1.0)
    self._avg_episode_length = self._avg_episode_length * (1.0 - alpha) + batch_avg * alpha

    if self._avg_episode_length < self._level_down_threshold:
      self._current_scale *= 1.0 - self._degree
    elif self._avg_episode_length > self._level_up_threshold:
      self._current_scale *= 1.0 + self._degree

    self._current_scale = float(
      np.clip(self._current_scale, self._min_scale, self._max_scale)
    )
    self._apply_scale()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_names: list[str],
    **kwargs: Any,
  ) -> dict[str, torch.Tensor]:
    del env, env_ids, reward_names, kwargs
    return {
      "penalty_scale": torch.tensor(self._current_scale),
      "avg_episode_length": torch.tensor(self._avg_episode_length),
    }
