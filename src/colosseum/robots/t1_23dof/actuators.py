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
    """Maximum velocity at joint (rad/s). peak_speed is output-side rpm, so no
    gear division (matches mjlab and the datasheet convention)."""
    return self.peak_speed / 60.0 * 2.0 * math.pi


# Motor specifications from manufacturer data
MOTOR_SPECS = {
  "neck": MotorSpec(
    gear_ratio=10,
    rated_voltage=48,
    rated_torque=3,
    peak_torque=7,
    rated_speed=120,
    peak_speed=400,
    rotor_inertia=18.0,
  ),
  "arm": MotorSpec(
    gear_ratio=36,
    rated_voltage=48,
    rated_torque=10,
    peak_torque=36,
    rated_speed=75,
    peak_speed=89,
    rotor_inertia=21.8,
  ),
  "hip_pitch": MotorSpec(
    gear_ratio=18,
    rated_voltage=48,
    rated_torque=20,
    peak_torque=55,
    rated_speed=155,
    peak_speed=157,
    rotor_inertia=161.7,
  ),
  "waist": MotorSpec(
    gear_ratio=25,
    rated_voltage=48,
    rated_torque=12,
    peak_torque=40,
    rated_speed=55,
    peak_speed=70,
    rotor_inertia=76.5,
  ),
  "knee": MotorSpec(
    gear_ratio=18,
    rated_voltage=48,
    rated_torque=25,
    peak_torque=65,
    rated_speed=132,
    peak_speed=140,
    rotor_inertia=196.3,
  ),
  "ankle_pitch": MotorSpec(
    gear_ratio=36,
    rated_voltage=48,
    rated_torque=15,
    peak_torque=50,
    rated_speed=109,
    peak_speed=117,
    rotor_inertia=26.2,
  ),
  "ankle_roll": MotorSpec(
    gear_ratio=36,
    rated_voltage=48,
    rated_torque=15,
    peak_torque=50,
    rated_speed=109,
    peak_speed=117,
    rotor_inertia=26.2,
  ),
}

##
# Actuator Configurations for Full-Body (23-DOF)
#
# stiffness/damping = the real robot's deployed WALK gains (booster_deploy
# T1WalkControllerCfg, registered as the `t1_walk` task — it overrides the
# softer placeholder gains in the base T1_23DOF_CFG via .replace()). Matched so
# the policy trains against the same joint compliance it runs on. armature =
# manufacturer reflected inertia (rotor_inertia * gear_ratio^2) for MuJoCo
# solver stability; the real robot doesn't model this, it's a sim-only term.
##

T1_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
  target_names_expr=("AAHead_yaw", "Head_pitch"),
  stiffness=4.0,
  damping=1.0,
  effort_limit=7.0,
  armature=0.0018,
  frictionloss=0.03,
)

T1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Shoulder.*", ".*Elbow.*"),
  stiffness=50.0,
  damping=1.0,
  effort_limit=18.0,
  armature=0.0283,
  frictionloss=0.03,
)

T1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("Waist",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=30.0,
  armature=0.0478,
  frictionloss=0.03,
)

T1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Hip_Pitch",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=45.0,
  armature=0.0524,
  frictionloss=0.03,
)

T1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Hip_Roll",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=30.0,
  armature=0.0478,
  frictionloss=0.03,
)

T1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Hip_Yaw",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=30.0,
  armature=0.0478,
  frictionloss=0.03,
)

T1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Knee_Pitch",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=65.0,
  armature=0.0636,
  frictionloss=0.03,
)

T1_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Ankle_Pitch",),
  stiffness=50.0,
  damping=2.0,
  effort_limit=24.0,
  armature=0.0340,
  frictionloss=0.03,
)

T1_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Ankle_Roll",),
  stiffness=50.0,
  damping=2.0,
  effort_limit=15.0,
  armature=0.0340,
  frictionloss=0.03,
)

##
# Actuator set selection
#
# Two complete, comparable 9-group actuator sets to A/B:
#   - DEPLOY_ACTUATORS (default): kp/kd from the real robot firmware
#     (booster_deploy T1_23DOF_CFG). Transfer-faithful. Conservative effort
#     limits; frictionloss 0.03.
#   - MJLAB_ACTUATORS: kp/kd derived from each motor's reflected inertia
#     assuming a 5 Hz closed-loop natural frequency and damping ratio 2
#     (mjlab_playground convention). Manufacturer peak torque as effort limit;
#     no frictionloss.
# Both use manufacturer reflected inertia for armature.
# See docs/research/t1_model_comparison.md.
##

DEPLOY_ACTUATORS = (
  T1_ACTUATOR_NECK,
  T1_ACTUATOR_ARM,
  T1_ACTUATOR_WAIST,
  T1_ACTUATOR_HIP_PITCH,
  T1_ACTUATOR_HIP_ROLL,
  T1_ACTUATOR_HIP_YAW,
  T1_ACTUATOR_KNEE,
  T1_ACTUATOR_ANKLE_PITCH,
  T1_ACTUATOR_ANKLE_ROLL,
)

_MJLAB_NATURAL_FREQ = 5.0 * 2.0 * math.pi  # 5 Hz, rad/s
_MJLAB_DAMPING_RATIO = 2.0


def _mjlab_actuator(
  motor: str, target_names_expr: tuple[str, ...]
) -> BuiltinPositionActuatorCfg:
  """Actuator whose gains come from the motor's reflected inertia I:
  kp = I * w^2, kv = 2 * zeta * I * w (mjlab_playground convention)."""
  spec = MOTOR_SPECS[motor]
  i = spec.reflected_inertia
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=i * _MJLAB_NATURAL_FREQ**2,
    damping=2.0 * _MJLAB_DAMPING_RATIO * i * _MJLAB_NATURAL_FREQ,
    effort_limit=spec.effort_limit,
    armature=i,
  )


# Waist, Hip_Roll, and Hip_Yaw share the same physical motor ("waist").
MJLAB_ACTUATORS = (
  _mjlab_actuator("neck", ("AAHead_yaw", "Head_pitch")),
  _mjlab_actuator("arm", (".*Shoulder.*", ".*Elbow.*")),
  _mjlab_actuator("waist", ("Waist",)),
  _mjlab_actuator("hip_pitch", (".*Hip_Pitch",)),
  _mjlab_actuator("waist", (".*Hip_Roll",)),
  _mjlab_actuator("waist", (".*Hip_Yaw",)),
  _mjlab_actuator("knee", (".*Knee_Pitch",)),
  _mjlab_actuator("ankle_pitch", (".*Ankle_Pitch",)),
  _mjlab_actuator("ankle_roll", (".*Ankle_Roll",)),
)


def _motor_for_joint(name: str) -> str:
  """Which physical motor drives a given joint (for mjlab-style scaling)."""
  if name in ("AAHead_yaw", "Head_pitch"):
    return "neck"
  if "Shoulder" in name or "Elbow" in name:
    return "arm"
  if name == "Waist" or "Hip_Roll" in name or "Hip_Yaw" in name:
    return "waist"
  if "Hip_Pitch" in name:
    return "hip_pitch"
  if "Knee" in name:
    return "knee"
  if "Ankle_Pitch" in name:
    return "ankle_pitch"
  if "Ankle_Roll" in name:
    return "ankle_roll"
  raise KeyError(f"No motor mapping for joint {name!r}")


def mjlab_action_scale(
  joint_names: tuple[str, ...], factor: float = 0.25
) -> dict[str, float]:
  """Per-joint action scale = factor * effort / stiffness, with stiffness the
  mjlab-derived kp = I*w^2 (mjlab_playground / ant convention). Gives every
  joint the same torque authority per unit action. Pairs with MJLAB_ACTUATORS;
  do NOT use with the deploy gains (the soft arm kp=4 would blow the scale up)."""
  out: dict[str, float] = {}
  for n in joint_names:
    spec = MOTOR_SPECS[_motor_for_joint(n)]
    kp = spec.reflected_inertia * _MJLAB_NATURAL_FREQ**2
    out[n] = factor * spec.effort_limit / kp
  return out
