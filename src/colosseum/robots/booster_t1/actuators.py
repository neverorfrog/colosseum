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

# TODO: Check kinematic limits (ctrllimited)

# Hip Pitch (uses hip_pitch motor)
_hip_pitch_stiffness, _hip_pitch_damping = compute_pd_gains(MOTOR_SPECS["hip_pitch"])
T1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Hip_Pitch",),
    stiffness=_hip_pitch_stiffness,
    damping=_hip_pitch_damping,
    effort_limit=MOTOR_SPECS["hip_pitch"].effort_limit,
    armature=MOTOR_SPECS["hip_pitch"].reflected_inertia,
)

# Hip Roll (uses waist motor)
_hip_roll_stiffness, _hip_roll_damping = compute_pd_gains(MOTOR_SPECS["waist"])
T1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Hip_Roll",),
    stiffness=_hip_roll_stiffness,
    damping=_hip_roll_damping,
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)

# Hip Yaw (uses waist motor)
_hip_yaw_stiffness, _hip_yaw_damping = compute_pd_gains(MOTOR_SPECS["waist"])
T1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Hip_Yaw",),
    stiffness=_hip_yaw_stiffness,
    damping=_hip_yaw_damping,
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)

# Knee (uses knee motor)
_knee_stiffness, _knee_damping = compute_pd_gains(MOTOR_SPECS["knee"])
T1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Knee_Pitch",),
    stiffness=_knee_stiffness,
    damping=_knee_damping,
    effort_limit=MOTOR_SPECS["knee"].effort_limit,
    armature=MOTOR_SPECS["knee"].reflected_inertia,
)

# Ankle Pitch (uses ankle motor)
_ankle_stiffness, _ankle_damping = compute_pd_gains(MOTOR_SPECS["ankle"])
T1_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Ankle_Pitch",),
    stiffness=_ankle_stiffness,
    damping=_ankle_damping,
    effort_limit=MOTOR_SPECS["ankle"].effort_limit,
    armature=MOTOR_SPECS["ankle"].reflected_inertia,
)

# Ankle Roll (uses ankle motor)
T1_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Ankle_Roll",),
    stiffness=_ankle_stiffness,
    damping=_ankle_damping,
    effort_limit=MOTOR_SPECS["ankle"].effort_limit,
    armature=MOTOR_SPECS["ankle"].reflected_inertia,
)

##
# Actuator Configurations for Full-Body (23-DOF)
##

# Neck (uses neck motor) - Higher frequency for faster response
_neck_natural_freq = 15.0 * 2.0 * math.pi  # 15Hz
_neck_stiffness, _neck_damping = compute_pd_gains(
    MOTOR_SPECS["neck"], natural_freq=_neck_natural_freq
)
T1_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
    joint_names_expr=("AAHead_yaw", "Head_pitch"),  # Explicit names to avoid conflicts
    stiffness=_neck_stiffness,
    damping=_neck_damping,
    effort_limit=MOTOR_SPECS["neck"].effort_limit,
    armature=MOTOR_SPECS["neck"].reflected_inertia,
)

# Arms (uses arm motor) - Moderate frequency
_arm_natural_freq = 12.0 * 2.0 * math.pi  # 12Hz
_arm_stiffness, _arm_damping = compute_pd_gains(
    MOTOR_SPECS["arm"], natural_freq=_arm_natural_freq
)
T1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
    joint_names_expr=(".*Shoulder.*", ".*Elbow.*"),
    stiffness=_arm_stiffness,
    damping=_arm_damping,
    effort_limit=MOTOR_SPECS["arm"].effort_limit,
    armature=MOTOR_SPECS["arm"].reflected_inertia,
)

# Waist (uses waist motor)
_waist_stiffness, _waist_damping = compute_pd_gains(MOTOR_SPECS["waist"])
T1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    joint_names_expr=("Waist",),
    stiffness=_waist_stiffness,
    damping=_waist_damping,
    effort_limit=MOTOR_SPECS["waist"].effort_limit,
    armature=MOTOR_SPECS["waist"].reflected_inertia,
)
