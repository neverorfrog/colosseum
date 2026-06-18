"""Booster T1 12-DOF locomotion actuators.

Legs-only, hand-tuned PD gains matching booster_gym's velocity-tracking config
(hip 200/5, knee 200/5, ankle 50/1). No T-N curve. The ``BoosterPdActuatorCfg``
class (explicit PD + optional torque-speed cap + command delay) is shared with
the 23-DOF model.
"""

from __future__ import annotations

from colosseum.robots.t1_23dof.actuators import BoosterPdActuatorCfg

# Legs only — the 12-DOF model has no head/arm/waist joints. Gains, effort
# limits and armatures follow booster_gym's T1_locomotion control block.
LOCOMOTION_ACTUATORS = (
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
    damping=2.0,
    effort_limit=24.0,
    armature=0.0340,
    delay_min_lag=2,
    delay_max_lag=8,
  ),
  BoosterPdActuatorCfg(
    target_names_expr=(".*Ankle_Roll",),
    stiffness=50.0,
    damping=2.0,
    effort_limit=15.0,
    armature=0.0340,
    delay_min_lag=2,
    delay_max_lag=8,
  ),
)
