"""Booster T1 actuators: explicit PD with a flat-top torque-speed limit.

Single canonical actuator set, ported 1-to-1 from booster_train's T1 config
(``BOOSTER_T1_CFG``). One mjlab actuator per joint: each joint is driven by its
datasheet motor model (``BoosterJoint``), with stiffness/damping derived the
booster_train way (``kp = I*(2*pi*f)^2``, ``kd = 2*zeta*I*(2*pi*f)``; ``f`` =
10 Hz, ``zeta`` = 2). The T1 ankles are parallel (``BoosterT1AnkleParaWrapper``):
pitch and roll share a doubled armature. Command delay is the 2-8 physics-step
bus lag baked into ``BoosterPdActuatorCfg``.

See docs/research/t1_actuator_comparison.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from mjlab.actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
  import mujoco
  import mujoco_warp as mjwarp

  from mjlab.actuator.actuator import ActuatorCmd


##
# Actuator: explicit PD + flat-top torque-speed (T-N) limit.
##


@dataclass(kw_only=True)
class BoosterPdActuatorCfg(IdealPdActuatorCfg):
  """``IdealPdActuator`` plus a flat-top torque-speed limit."""

  velocity_limit: float = float("inf")
  """Joint speed at which the available torque reaches zero (rad/s)."""

  knee_point_velocity: float = float("inf")
  """Joint speed below which the full ``effort_limit`` is available (rad/s).
  Between the knee and ``velocity_limit`` the torque cap drops linearly to zero.
  """

  # booster_train command delay: 2-8 physics steps, resampled at reset.
  delay_min_lag: int = 2
  delay_max_lag: int = 8

  def build(
    self, entity, target_ids: list[int], target_names: list[str]
  ) -> BoosterPdActuator:
    return BoosterPdActuator(self, entity, target_ids, target_names)


class BoosterPdActuator(IdealPdActuator[BoosterPdActuatorCfg]):
  """Explicit PD with a speed-dependent torque cap (booster_train T-N curve)."""

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)
    # force_limit is (num_envs, num_targets); broadcast the speed params to match.
    assert self.force_limit is not None
    shape = self.force_limit.shape
    self._vmax = torch.full(shape, self.cfg.velocity_limit, device=device)
    knee = torch.full(shape, self.cfg.knee_point_velocity, device=device)
    self._knee = torch.minimum(knee, self._vmax).clamp_min(0.0)
    self._denom = (self._vmax - self._knee).clamp_min(1e-6)
    self._joint_vel = torch.zeros(shape, device=device)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    # Stash measured velocity so the overridden _clip_effort can shape the cap.
    self._joint_vel = cmd.vel
    return super().compute(cmd)

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    # Full force_limit for |v| <= knee, then linear to zero at |v| = velocity_limit.
    assert self.force_limit is not None
    v = self._joint_vel.abs()
    tau_max = self.force_limit
    tau_linear = tau_max * (self._vmax - v) / self._denom
    max_effort = tau_linear.clamp(min=0.0).clamp(max=tau_max)
    # velocity_limit = inf -> no speed-dependent reduction (plain force box).
    max_effort = torch.where(torch.isinf(self._vmax), tau_max, max_effort)
    return torch.clamp(effort, -max_effort, max_effort)


##
# Motor models (booster_train datasheets; see external/booster_train .../actuator.py).
##


@dataclass(frozen=True)
class BoosterJoint:
  """Datasheet motor model. stiffness/damping are derived from the armature and
  the closed-loop natural frequency, matching booster_train's ``BoosterJointCfg``.
  """

  effort_limit: float  # Nm
  velocity_limit: float  # rad/s; available torque reaches zero here
  knee_point_velocity: float  # rad/s; full effort below here
  armature: float  # kg*m^2 reflected inertia
  natural_freq: float = 10.0  # Hz (booster T1 default; K1 uses 4.0)
  damping_ratio: float = 2.0

  @property
  def stiffness(self) -> float:
    return self.armature * (2.0 * math.pi * self.natural_freq) ** 2

  @property
  def damping(self) -> float:
    return (
      2.0 * self.damping_ratio * self.armature * (2.0 * math.pi * self.natural_freq)
    )


def parallel(
  base: BoosterJoint,
  serial_index: int,
  *,
  effort_ratio: tuple[float, float] = (1.0, 1.0),
  velocity_ratio: tuple[float, float] = (1.0, 1.0),
  armature_ratio: tuple[float, float] = (2.0, 2.0),
  knee_ratio: tuple[float, float] = (1.0, 1.0),
) -> BoosterJoint:
  """Parallel-mechanism joint from a base motor (booster_train
  ``ParallelJointWrapperCfg``). ``serial_index`` selects pitch (0) or roll (1);
  defaults are the T1 ankle ratios (armature doubled)."""
  i = serial_index
  return replace(
    base,
    effort_limit=base.effort_limit * effort_ratio[i],
    velocity_limit=base.velocity_limit * velocity_ratio[i],
    knee_point_velocity=base.knee_point_velocity * knee_ratio[i],
    armature=base.armature * armature_ratio[i],
  )


# Named by motor model: preserves identity and keeps the datasheet in one place.
E4310 = BoosterJoint(38.3, 17.59, 7.85, 0.0282528)  # arms
E6408 = BoosterJoint(68.0, 14.66, 1.88, 0.0478125)  # waist, hip roll/yaw
E8112 = BoosterJoint(96.0, 16.76, 7.54, 0.0523908)  # hip pitch
E8116 = BoosterJoint(130.0, 14.66, 6.28, 0.0636012)  # knee
DM4310 = BoosterJoint(7.0, 12.57, 41.89, 0.0018)  # neck/head
E4315 = BoosterJoint(76.0, 12.57, 2.62, 0.0339552)  # ankle base motor
ANKLE_PITCH = parallel(E4315, 0)
ANKLE_ROLL = parallel(E4315, 1)


##
# One mjlab actuator per joint.
##

_SIDES = ("Left", "Right")
_ARM_JOINTS = ("Shoulder_Pitch", "Shoulder_Roll", "Elbow_Pitch", "Elbow_Yaw")

# Each joint -> its physical motor model.
JOINT_MOTORS: dict[str, BoosterJoint] = {
  "AAHead_yaw": DM4310,
  "Head_pitch": DM4310,
  **{f"{s}_{j}": E4310 for s in _SIDES for j in _ARM_JOINTS},
  "Waist": E6408,
  **{f"{s}_Hip_Pitch": E8112 for s in _SIDES},
  **{f"{s}_Hip_Roll": E6408 for s in _SIDES},
  **{f"{s}_Hip_Yaw": E6408 for s in _SIDES},
  **{f"{s}_Knee_Pitch": E8116 for s in _SIDES},
  **{f"{s}_Ankle_Pitch": ANKLE_PITCH for s in _SIDES},
  **{f"{s}_Ankle_Roll": ANKLE_ROLL for s in _SIDES},
}


def actuator(joint: str, motor: BoosterJoint) -> BoosterPdActuatorCfg:
  return BoosterPdActuatorCfg(
    target_names_expr=(joint,),
    stiffness=motor.stiffness,
    damping=motor.damping,
    effort_limit=motor.effort_limit,
    armature=motor.armature,
    velocity_limit=motor.velocity_limit,
    knee_point_velocity=motor.knee_point_velocity,
  )


ACTUATORS = tuple(actuator(j, m) for j, m in JOINT_MOTORS.items())


def action_scale(factor: float = 0.25) -> dict[str, float]:
  """Per-joint action scale, booster_train recipe: ``factor * effort / stiffness``.
  With ``kp * scale = factor * effort``, every joint gets torque authority
  proportional to its motor's effort, independent of the chosen stiffness."""
  return {
    cfg.target_names_expr[0]: factor * cfg.effort_limit / cfg.stiffness
    for cfg in ACTUATORS
  }
