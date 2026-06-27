"""Booster T1 actuator sets: MANUFACTURER_ACTUATORS and LOCOMOTION_ACTUATORS."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from mjlab.actuator import DcMotorActuatorCfg, IdealPdActuator, IdealPdActuatorCfg

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
# Manufacturer datasheet motor models (Booster T1 official spec sheet).
# Mapped onto mjlab's DcMotorActuator following booster_train's convention: the
# per-step clamp is the *peak* (availability) torque, so peak -> both
# saturation_effort and effort_limit, with the DC speed-droop tapering it toward
# zero at peak speed -> velocity_limit (no-load). Rated/continuous torque is a
# thermal/duty-cycle spec, not an instantaneous clamp; it is left to the reward's
# torque penalties rather than hard-capped here. Torque/speed are joint-side
# (post-gearbox); rotor inertia is motor-side, reflected via the gear ratio.
##

_RPM_TO_RAD_S = math.pi / 30.0


@dataclass(frozen=True)
class ManufacturerMotor:
  """One motor model from the Booster T1 datasheet. stiffness/damping follow
  booster_train's closed-loop convention (kp = I*(2*pi*f)^2,
  kd = 2*zeta*I*(2*pi*f); f=10 Hz, zeta=2).
  """

  peak_torque: float  # Nm; stall torque -> saturation_effort AND effort_limit
  rated_torque: float  # Nm; continuous/thermal rating (datasheet ref; not clamped)
  peak_speed_rpm: float  # rpm; no-load speed -> velocity_limit
  rotor_inertia: float  # kg*m^2, motor side
  gear_ratio: float
  natural_freq: float = 10.0  # Hz
  damping_ratio: float = 2.0

  @property
  def armature(self) -> float:
    return self.rotor_inertia * self.gear_ratio**2

  @property
  def velocity_limit(self) -> float:
    return self.peak_speed_rpm * _RPM_TO_RAD_S

  @property
  def stiffness(self) -> float:
    return self.armature * (2.0 * math.pi * self.natural_freq) ** 2

  @property
  def damping(self) -> float:
    return (
      2.0 * self.damping_ratio * self.armature * (2.0 * math.pi * self.natural_freq)
    )


# Datasheet columns: peak/rated torque (Nm), peak speed (rpm), rotor inertia
# (kg*m^2, motor side), gear ratio.
NECK = ManufacturerMotor(7.0, 3.0, 250, 18.0e-6, 10)
ARM = ManufacturerMotor(30.0, 10.0, 173, 21.8e-6, 36)
WAIST_HIP_ROLL_YAW = ManufacturerMotor(60.0, 13.0, 68, 76.5e-6, 25)
HIP_PITCH = ManufacturerMotor(90.0, 30.0, 146, 161.7e-6, 18)
KNEE = ManufacturerMotor(120.0, 40.0, 141, 196.3e-6, 18)
ANKLE = ManufacturerMotor(50.0, 16.0, 123, 26.2e-6, 36)


##
# One mjlab actuator per joint.
##

_SIDES = ("Left", "Right")
_ARM_JOINTS = ("Shoulder_Pitch", "Shoulder_Roll", "Elbow_Pitch", "Elbow_Yaw")

# Leg PD tuning. booster_train's T1 config leaves the legs on the library default
# (f=10 Hz, zeta=2 -> stiff and heavily over-damped, the "sluggish" feel), but its
# K1 config -- the platform they actually tuned -- softens the legs to f=4 Hz with
# zeta=1.5 on hips/ankles and zeta=1.0 (critical) on the knee. We adopt the K1
# recipe for the T1 legs; arms/head/waist stay on the manufacturer default.
# Lowering f drops kp (kp ~ f^2), so MANUFACTURER_ACTION_SCALE (0.25*peak/kp) rises
# accordingly -- it is derived from stiffness, so it stays consistent automatically.
_LEG_TUNING = {"natural_freq": 5.0, "damping_ratio": 1.5}
_KNEE_TUNING = {"natural_freq": 5.0, "damping_ratio": 1.25}
FRICTIONLOSS = 0.05

# Each joint -> its datasheet motor model.
JOINT_MOTORS: dict[str, ManufacturerMotor] = {
  "AAHead_yaw": NECK,
  "Head_pitch": NECK,
  **{f"{s}_{j}": ARM for s in _SIDES for j in _ARM_JOINTS},
  "Waist": WAIST_HIP_ROLL_YAW,
  **{f"{s}_Hip_Pitch": replace(HIP_PITCH, **_LEG_TUNING) for s in _SIDES},
  **{f"{s}_Hip_Roll": replace(WAIST_HIP_ROLL_YAW, **_LEG_TUNING) for s in _SIDES},
  **{f"{s}_Hip_Yaw": replace(WAIST_HIP_ROLL_YAW, **_LEG_TUNING) for s in _SIDES},
  **{f"{s}_Knee_Pitch": replace(KNEE, **_KNEE_TUNING) for s in _SIDES},
  **{f"{s}_Ankle_Pitch": replace(ANKLE, **_LEG_TUNING) for s in _SIDES},
  **{f"{s}_Ankle_Roll": replace(ANKLE, **_LEG_TUNING) for s in _SIDES},
}


def _actuator(joint: str, motor: ManufacturerMotor) -> DcMotorActuatorCfg:
  return DcMotorActuatorCfg(
    target_names_expr=(joint,),
    stiffness=motor.stiffness,
    damping=motor.damping,
    saturation_effort=motor.peak_torque,
    effort_limit=motor.peak_torque,
    velocity_limit=motor.velocity_limit,
    armature=motor.armature,
    frictionloss=FRICTIONLOSS,
  )


MANUFACTURER_ACTUATORS = tuple(_actuator(j, m) for j, m in JOINT_MOTORS.items())

##
# Locomotion actuator set: hand-tuned kp/kd (hip 200/5, knee 200/5, ankle 50/2).
# No T-N curve, no command delay.
##

LOCOMOTION_ACTUATORS = (
  # Head — held at default (not controlled by velocity policy).
  BoosterPdActuatorCfg(
    target_names_expr=("AAHead_yaw", "Head_pitch"),
    stiffness=4.0,
    damping=1.0,
    effort_limit=7.0,
    armature=0.0018,
    delay_min_lag=2,
    delay_max_lag=8,
  ),
  # Arms — held at default pose.
  BoosterPdActuatorCfg(
    target_names_expr=(".*Shoulder.*", ".*Elbow.*"),
    stiffness=50.0,
    damping=1.0,
    effort_limit=18.0,
    armature=0.0283,
    delay_min_lag=2,
    delay_max_lag=8,
  ),
  # Waist — held at default.
  BoosterPdActuatorCfg(
    target_names_expr=("Waist",),
    stiffness=200.0,
    damping=5.0,
    effort_limit=30.0,
    armature=0.0478,
    delay_min_lag=2,
    delay_max_lag=8,
  ),
  # Legs — locomotion control joints (booster_gym convention, no T-N curve).
  BoosterPdActuatorCfg(
    target_names_expr=(".*Hip_Pitch",),
    stiffness=200.0,
    damping=5.0,
    effort_limit=45.0,
    armature=0.0524,
    delay_min_lag=2,
    delay_max_lag=8,
  ),
  BoosterPdActuatorCfg(
    target_names_expr=(".*Hip_Roll",),
    stiffness=200.0,
    damping=5.0,
    effort_limit=30.0,
    armature=0.0478,
    delay_min_lag=2,
    delay_max_lag=8,
  ),
  BoosterPdActuatorCfg(
    target_names_expr=(".*Hip_Yaw",),
    stiffness=200.0,
    damping=5.0,
    effort_limit=30.0,
    armature=0.0478,
    delay_min_lag=2,
    delay_max_lag=8,
  ),
  BoosterPdActuatorCfg(
    target_names_expr=(".*Knee_Pitch",),
    stiffness=200.0,
    damping=5.0,
    effort_limit=60.0,
    armature=0.0636,
    delay_min_lag=2,
    delay_max_lag=8,
  ),
  BoosterPdActuatorCfg(
    target_names_expr=(".*Ankle_Pitch",),
    stiffness=50.0,
    damping=2.5,
    effort_limit=24.0,
    armature=0.0340,
    delay_min_lag=2,
    delay_max_lag=8,
  ),
  BoosterPdActuatorCfg(
    target_names_expr=(".*Ankle_Roll",),
    stiffness=50.0,
    damping=2.5,
    effort_limit=15.0,
    armature=0.0340,
    delay_min_lag=2,
    delay_max_lag=8,
  ),
)
