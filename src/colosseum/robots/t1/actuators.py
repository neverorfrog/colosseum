"""Booster T1 actuator set"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from mjlab.actuator import DcMotorActuatorCfg

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
class Motor:
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
NECK = Motor(7.0, 3.0, 250, 18.0e-6, 10)
ARM = Motor(30.0, 10.0, 173, 21.8e-6, 36)
WAIST_HIP_ROLL_YAW = Motor(60.0, 13.0, 68, 76.5e-6, 25)
HIP_PITCH = Motor(90.0, 30.0, 146, 161.7e-6, 18)
KNEE = Motor(120.0, 40.0, 141, 196.3e-6, 18)
ANKLE = Motor(50.0, 16.0, 123, 26.2e-6, 36)


##
# One mjlab actuator per joint.
##

_SIDES = ("Left", "Right")
_ARM_JOINTS = ("Shoulder_Pitch", "Shoulder_Roll", "Elbow_Pitch", "Elbow_Yaw")
_LEG_TUNING = {"natural_freq": 5.0, "damping_ratio": 1.5}
_KNEE_TUNING = {"natural_freq": 5.0, "damping_ratio": 1.25}
FRICTIONLOSS = 0.05

# Each joint -> its datasheet motor model.
JOINT_MOTORS: dict[str, Motor] = {
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


def _actuator(joint: str, motor: Motor) -> DcMotorActuatorCfg:
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


ACTUATORS = tuple(_actuator(j, m) for j, m in JOINT_MOTORS.items())