"""Curriculum terms for the dribbling task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from colosseum.mdp.curriculums import _validate_transition_stages

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

from colosseum.tasks.dribbling.mdp.obstacle_commands import ObstacleCommand


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
            {"transitions": 0,            "half_range": 0.0},          # always forward
            {"transitions": 8_000_000,    "half_range": math.pi / 6},  # ±30°
            {"transitions": 24_000_000,   "half_range": math.pi / 3},  # ±60°
            {"transitions": 48_000_000,   "half_range": math.pi / 2},  # ±90°
            {"transitions": 80_000_000,   "half_range": math.pi},      # ±180° (full)
          ],
        },
      )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    event_name: str = cfg.params["event_name"]
    self._term_cfg = env.event_manager.get_term_cfg(event_name)
    self._stages: list[dict[str, Any]] = cfg.params["stages"]
    self._num_envs = env.num_envs
    for i in range(1, len(self._stages)):
      if self._stages[i]["transitions"] < self._stages[i - 1]["transitions"]:
        raise ValueError(
          f"yaw_reset_curriculum stages must be in nondecreasing transition order, "
          f"but stage {i} (transitions={self._stages[i]['transitions']}) < "
          f"stage {i - 1} (transitions={self._stages[i - 1]['transitions']})."
        )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[dict],
  ) -> dict[str, torch.Tensor]:
    del env_ids, event_name, stages
    total = env.common_step_counter * self._num_envs
    half_range = 0.0
    for stage in self._stages:
      if total >= stage["transitions"]:
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
            {"transitions": 0,             "max_speed": 0.3},
            {"transitions": 20_000_000,    "max_speed": 0.6},
            {"transitions": 48_000_000,    "max_speed": 1.0},
          ],
        },
      )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    event_name: str = cfg.params["event_name"]
    self._term_cfg = env.event_manager.get_term_cfg(event_name)
    self._stages: list[dict[str, Any]] = cfg.params["stages"]
    self._num_envs = env.num_envs
    for i in range(1, len(self._stages)):
      if self._stages[i]["transitions"] < self._stages[i - 1]["transitions"]:
        raise ValueError(
          "push_ball_curriculum stages must be in nondecreasing transition order."
        )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[dict],
  ) -> dict[str, torch.Tensor]:
    del env_ids, event_name, stages
    total = env.common_step_counter * self._num_envs
    max_speed = 0.3
    for stage in self._stages:
      if total >= stage["transitions"]:
        max_speed = stage["max_speed"]
    self._term_cfg.params["velocity_range"]["x"] = (-max_speed, max_speed)
    self._term_cfg.params["velocity_range"]["y"] = (-max_speed, max_speed)
    return {"push_ball_max_speed": torch.tensor(max_speed)}


class obstacle_curriculum:
  """Curriculum term for staged obstacle behaviors."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    command_name: str = cfg.params["command_name"]
    self._term: ObstacleCommand = env.command_manager.get_term(command_name)
    self._stages: list[dict[str, Any]] = cfg.params["stages"]
    self._num_envs = env.num_envs
    _validate_transition_stages(self._term.cfg, command_name, self._stages)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    stages: list[dict],
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, stages
    total = env.common_step_counter * self._num_envs
    stage_index = 0
    num_active = 0
    behavior = "none"
    distance_range = (1.5, 3.0)
    lateral_offset_range = (-0.3, 0.3)
    forward_fraction_range = (0.35, 0.75)
    min_speed = 0.0
    max_speed = 0.0
    velocity_resample_time_range = (0.5, 1.0)
    for idx, stage in enumerate(self._stages):
      if total >= stage["transitions"]:
        stage_index = idx
        num_active = stage.get("num_active", num_active)
        behavior = stage.get("behavior", behavior)
        distance_range = stage.get("distance_range", distance_range)
        lateral_offset_range = stage.get("lateral_offset_range", lateral_offset_range)
        forward_fraction_range = stage.get(
          "forward_fraction_range", forward_fraction_range
        )
        min_speed = stage.get("min_speed", min_speed)
        max_speed = stage.get("max_speed", max_speed)
        velocity_resample_time_range = stage.get(
          "velocity_resample_time_range", velocity_resample_time_range
        )

    self._term.cfg.num_active = num_active
    self._term.cfg.behavior = behavior
    self._term.cfg.distance_range = distance_range
    self._term.cfg.lateral_offset_range = lateral_offset_range
    self._term.cfg.forward_fraction_range = forward_fraction_range
    self._term.cfg.min_speed = min_speed
    self._term.cfg.max_speed = max_speed
    self._term.cfg.velocity_resample_time_range = velocity_resample_time_range

    behavior_to_id = {
      "none": 0.0,
      "static_blocker": 1.0,
      "lateral_blocker": 2.0,
      "ball_attacker": 3.0,
      "mixed_attackers": 4.0,
    }

    return {
      "obstacle_stage_index": torch.tensor(float(stage_index)),
      "obstacle_behavior_id": torch.tensor(behavior_to_id.get(behavior, -1.0)),
      "num_active_obstacles": torch.tensor(float(num_active)),
      "obstacle_min_dist_m": torch.tensor(float(distance_range[0])),
      "obstacle_min_speed": torch.tensor(float(min_speed)),
      "obstacle_max_speed": torch.tensor(float(max_speed)),
    }
