"""Booster T1 actuator configurations based on manufacturer motor specifications."""

import math
from dataclasses import dataclass

from mjlab.actuator import BuiltinPositionActuatorCfg

##
# Motor Specifications (from manufacturer data)
##


@dataclass
class MotorSpec:
  """Motor specifications from manufacturer."""

  gear_ratio: float
  rated_voltage: float  # V
  rated_torque: float  # Nm
  peak_torque: float  # Nm
  rated_speed: float  # rpm
  peak_speed: float  # rpm
  rotor_inertia: float  # kg·mm²

  @property
  def reflected_inertia(self) -> float:
    """Reflected inertia at joint (converted to kg·m²)."""
    return (self.rotor_inertia * 1e-6) * (self.gear_ratio**2)

  @property
  def effort_limit(self) -> float:
    """Maximum torque at joint (Nm)."""
    return self.peak_torque

  @property
  def velocity_limit(self) -> float:
    """Maximum velocity at joint (rad/s)."""
    return (self.peak_speed / 60.0) * 2.0 * math.pi / self.gear_ratio


# Motor specifications from manufacturer data
MOTOR_SPECS = {
  "hip_pitch": MotorSpec(
    gear_ratio=18,
    rated_voltage=48,
    rated_torque=30,
    peak_torque=90,
    rated_speed=140,
    peak_speed=160,
    rotor_inertia=161.7,
  ),
  "waist": MotorSpec(
    gear_ratio=25,
    rated_voltage=48,
    rated_torque=13,
    peak_torque=40,
    rated_speed=57,
    peak_speed=70,
    rotor_inertia=76.5,
  ),
  "knee": MotorSpec(
    gear_ratio=18,
    rated_voltage=48,
    rated_torque=39,
    peak_torque=118,
    rated_speed=120,
    peak_speed=140,
    rotor_inertia=196.3,
  ),
  "ankle_pitch": MotorSpec(
    gear_ratio=36,
    rated_voltage=48,
    rated_torque=19,
    peak_torque=57,
    rated_speed=104,
    peak_speed=123,
    rotor_inertia=26.2,
  ),
  "ankle_roll": MotorSpec(
    gear_ratio=36,
    rated_voltage=48,
    rated_torque=19,
    peak_torque=57,
    rated_speed=104,
    peak_speed=123,
    rotor_inertia=26.2,
  ),
  "arm": MotorSpec(
    gear_ratio=36,
    rated_voltage=48,
    rated_torque=10,
    peak_torque=30,
    rated_speed=147,
    peak_speed=167,
    rotor_inertia=21.8,
  ),
  "neck": MotorSpec(
    gear_ratio=10,
    rated_voltage=48,
    rated_torque=3,
    peak_torque=7,
    rated_speed=120,
    peak_speed=400,
    rotor_inertia=18.0,
  ),
}

##
# Actuator Configurations for Full-Body (23-DOF)
##

T1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Hip_Pitch",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=MOTOR_SPECS["hip_pitch"].effort_limit,
  armature=MOTOR_SPECS["hip_pitch"].reflected_inertia,
  frictionloss=0.2,
)

T1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Hip_Roll",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=MOTOR_SPECS["waist"].effort_limit,
  armature=MOTOR_SPECS["waist"].reflected_inertia,
  frictionloss=0.2,
)

T1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Hip_Yaw",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=MOTOR_SPECS["waist"].effort_limit,
  armature=MOTOR_SPECS["waist"].reflected_inertia,
  frictionloss=0.2,
)

T1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Knee_Pitch",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=MOTOR_SPECS["knee"].effort_limit,
  armature=MOTOR_SPECS["knee"].reflected_inertia,
  frictionloss=0.2,
)

T1_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Ankle_Pitch",),
  stiffness=50.0,
  damping=3.0,
  effort_limit=MOTOR_SPECS["ankle_pitch"].effort_limit,
  armature=MOTOR_SPECS["ankle_pitch"].reflected_inertia,
  frictionloss=0.1,
)

T1_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Ankle_Roll",),
  stiffness=50.0,
  damping=3.0,
  effort_limit=MOTOR_SPECS["ankle_roll"].effort_limit,
  armature=MOTOR_SPECS["ankle_roll"].reflected_inertia,
  frictionloss=0.1,
)

T1_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
  target_names_expr=("AAHead_yaw", "Head_pitch"),
  stiffness=5.0,
  damping=0.5,
  effort_limit=MOTOR_SPECS["neck"].effort_limit,
  armature=MOTOR_SPECS["neck"].reflected_inertia,
  frictionloss=0.2,
)

T1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Shoulder.*", ".*Elbow.*"),
  stiffness=20.0,
  damping=0.5,
  effort_limit=MOTOR_SPECS["arm"].effort_limit,
  armature=MOTOR_SPECS["arm"].reflected_inertia,
  frictionloss=0.2,
)

T1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("Waist",),
  stiffness=150.0,
  damping=5.0,
  effort_limit=MOTOR_SPECS["waist"].effort_limit,
  armature=MOTOR_SPECS["waist"].reflected_inertia,
  frictionloss=0.2,
)
