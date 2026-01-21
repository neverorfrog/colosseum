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
    "ankle": MotorSpec(
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
# PD Gain Computation (following Unitree G1 method)
##

# G1 uses: natural_freq = 10Hz, damping_ratio = 2.0
# This is overdamped (ζ > 1), which prevents oscillations
NATURAL_FREQ = 10.0 * 2.0 * math.pi  # 10Hz in rad/s
DAMPING_RATIO = 2.0  # Overdamped (same as G1)


def compute_pd_gains(
    motor: MotorSpec,
    natural_freq: float = NATURAL_FREQ,
    damping_ratio: float = DAMPING_RATIO,
) -> tuple[float, float]:
    """
    Compute PD gains from motor specs using G1/GO1 method.

    Formula (from Unitree G1):
        stiffness = reflected_inertia × ωₙ²
        damping = 2 × ζ × reflected_inertia × ωₙ

    Args:
        motor: Motor specification
        natural_freq: Natural frequency in rad/s (default: 10Hz = 62.83 rad/s)
        damping_ratio: Damping ratio (default: 2.0 for overdamping, same as G1)

    Returns:
        (stiffness, damping) tuple in (Nm/rad, Nm·s/rad)
    """
    stiffness = motor.reflected_inertia * (natural_freq**2)
    damping = 2.0 * damping_ratio * motor.reflected_inertia * natural_freq
    return stiffness, damping


##
# Actuator Configurations for 12-DOF Locomotion
##

T1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(".*Hip_Pitch",),
    stiffness=200.0,
    damping=5.0,
    effort_limit=MOTOR_SPECS["hip_pitch"].effort_limit,
    armature=MOTOR_SPECS["hip_pitch"].reflected_inertia,
)

T1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(".*Hip_Roll",),
    stiffness=200.0,
    damping=5.0,
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)

T1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(".*Hip_Yaw",),
    stiffness=200.0,
    damping=5.0,
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)

T1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*Knee_Pitch",),
    stiffness=200.0,
    damping=5.0,
    effort_limit=MOTOR_SPECS["knee"].effort_limit,
    armature=MOTOR_SPECS["knee"].reflected_inertia,
)

T1_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(".*Ankle_Pitch",),
    stiffness=50.0,
    damping=3.0,
    effort_limit=MOTOR_SPECS["ankle"].effort_limit,
    armature=MOTOR_SPECS["ankle"].reflected_inertia,
)

T1_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(".*Ankle_Roll",),
    stiffness=50.0,
    damping=3.0,
    effort_limit=MOTOR_SPECS["ankle"].effort_limit,
    armature=MOTOR_SPECS["ankle"].reflected_inertia,
)

##
# Actuator Configurations for Full-Body (23-DOF)
##

T1_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
    target_names_expr=("AAHead_yaw", "Head_pitch"),
    stiffness=5.0,     # Holosoma T1 (was computed 15.99)
    damping=0.5,       # Holosoma T1 (was computed 0.68)
    effort_limit=MOTOR_SPECS["neck"].effort_limit,
    armature=MOTOR_SPECS["neck"].reflected_inertia,
)

T1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
    target_names_expr=(".*Shoulder.*", ".*Elbow.*"),
    stiffness=20.0,    # Holosoma T1 (was computed 160.61!)
    damping=0.5,       # Holosoma T1 (was computed 8.52!)
    effort_limit=MOTOR_SPECS["arm"].effort_limit,
    armature=MOTOR_SPECS["arm"].reflected_inertia,
)

T1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=("Waist",),
    stiffness=200.0,   # Holosoma T1 (was computed 188.76) ✓
    damping=5.0,       # Holosoma T1 (was computed 12.02)
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)