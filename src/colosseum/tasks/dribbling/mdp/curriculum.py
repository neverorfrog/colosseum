"""Curriculum terms for the dribbling task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class yaw_reset_curriculum:
  """Gradually widen the yaw randomization range on the ``reset_base`` event.

  Modifies ``event_manager["reset_base"].params["pose_range"]["yaw"]`` in
  place so that the robot starts always facing forward and progressively
  faces random directions as training proceeds.

  Stages example (in ``cact_cfg.py``)::

    CurriculumTermCfg(
      func=mdp.yaw_reset_curriculum,
      params={
        "event_name": "reset_base",
        "stages": [
          {"step":     0, "half_range": 0.0},          # always forward
          {"step":  2000, "half_range": math.pi / 6},  # ±30°
          {"step":  6000, "half_range": math.pi / 3},  # ±60°
          {"step": 12000, "half_range": math.pi / 2},  # ±90°
          {"step": 20000, "half_range": math.pi},      # ±180° (full)
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    event_name: str = cfg.params["event_name"]
    self._term_cfg = env.event_manager.get_term_cfg(event_name)
    self._stages = cfg.params["stages"]
    # Validate stage ordering.
    for i in range(1, len(self._stages)):
      if self._stages[i]["step"] < self._stages[i - 1]["step"]:
        raise ValueError(
          f"yaw_reset_curriculum stages must be in nondecreasing step order, "
          f"but stage {i} (step={self._stages[i]['step']}) < "
          f"stage {i-1} (step={self._stages[i-1]['step']})."
        )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[dict],
  ) -> dict[str, torch.Tensor]:
    step = env.common_step_counter
    half_range = 0.0
    for stage in stages:
      if step >= stage["step"]:
        half_range = stage["half_range"]
    self._term_cfg.params["pose_range"]["yaw"] = (-half_range, half_range)
    return {"yaw_half_range_deg": torch.tensor(math.degrees(half_range))}


class push_ball_curriculum:
  """Gradually increase the random ball push magnitude.

  Modifies ``event_manager["push_ball"].params["velocity_range"]`` in place.

  Stages example::

    CurriculumTermCfg(
      func=mdp.push_ball_curriculum,
      params={
        "event_name": "push_ball",
        "stages": [
          {"step":     0, "max_speed": 0.3},
          {"step":  5000, "max_speed": 0.6},
          {"step": 12000, "max_speed": 1.0},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    event_name: str = cfg.params["event_name"]
    self._term_cfg = env.event_manager.get_term_cfg(event_name)
    self._stages = cfg.params["stages"]
    for i in range(1, len(self._stages)):
      if self._stages[i]["step"] < self._stages[i - 1]["step"]:
        raise ValueError(
          f"push_ball_curriculum stages must be in nondecreasing step order."
        )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[dict],
  ) -> dict[str, torch.Tensor]:
    step = env.common_step_counter
    max_speed = 0.3
    for stage in stages:
      if step >= stage["step"]:
        max_speed = stage["max_speed"]
    self._term_cfg.params["velocity_range"]["x"] = (-max_speed, max_speed)
    self._term_cfg.params["velocity_range"]["y"] = (-max_speed, max_speed)
    return {"push_ball_max_speed": torch.tensor(max_speed)}
