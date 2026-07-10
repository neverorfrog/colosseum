# T1 Actuator Comparison: colosseum vs booster_train

Comparison of the Booster T1 actuator modeling in this repo
(`src/colosseum/robots/t1/`) against the manufacturer's own Isaac Lab
training repo (`external/booster_train`). Written ahead of porting the
booster_train actuator data into colosseum as the single canonical set, and
ahead of refactoring colosseum's actuator class to add the speed-dependent
torque modeling that booster_train has and we currently lack.

Sources:
- colosseum: `actuators.py` (`DEPLOY_ACTUATORS`, `MJLAB_ACTUATORS`), `constants.py` (`ACTION_SCALE`)
- booster_train: `source/booster_train/booster_train/assets/robots/actuator.py`, `.../booster.py` (`BOOSTER_T1_CFG`)

---

## 1. How each side derives gains

| Source | stiffness / damping | natural freq | effort source |
|---|---|---|---|
| colosseum `DEPLOY_ACTUATORS` | hand-tuned firmware deploy gains | — | conservative deploy limits |
| colosseum `MJLAB_ACTUATORS` | `kp = I·ω²`, `kd = 2ζIω` | **5 Hz**, ζ=2 | `MOTOR_SPECS.peak_torque` |
| **booster_train T1** | `kp = I·ω²`, `kd = 2ζIω` (identical formula) | **10 Hz**, ζ=2 | motor-model datasheet (`BoosterJoint*`) |

`MJLAB_ACTUATORS` and booster_train use the **same formula**. The only
derivation difference is the natural frequency: booster_train T1 leaves
`natural_freq` at its **10 Hz** default (`BoosterJointCfg.natural_freq = 10.`),
while `MJLAB_ACTUATORS` uses 5 Hz. That is a clean **4× stiffness** factor for
equal inertia.

> Note: booster_train's *K1* config overrides `natural_freq=4.0`, but the **T1**
> config passes no override, so every T1 group runs at the 10 Hz / ζ=2 default.

---

## 2. Per-joint parameter comparison

booster_train T1 values, computed with `ω = 2π·10 = 62.832`, `ω² = 3947.84`,
`kd = armature · 2ζω = armature · 251.33`:

| group | motor model | armature | stiffness | damping | effort | vel limit | knee vel (T-N) |
|---|---|---|---|---|---|---|---|
| head (neck) | DM4310 | 0.00180 | 7.11 | 0.45 | 7.0 | 12.57 | 41.89 |
| arms | E4310 | 0.02825 | 111.54 | 7.10 | 38.3 | 17.59 | 7.85 |
| waist | E6408 | 0.04781 | 188.76 | 12.02 | 68.0 | 14.66 | 1.88 |
| hip_pitch | E8112 | 0.05239 | 206.83 | 13.17 | 96.0 | 16.76 | 7.54 |
| hip_roll | E6408 | 0.04781 | 188.76 | 12.02 | 68.0 | 14.66 | 1.88 |
| hip_yaw | E6408 | 0.04781 | 188.76 | 12.02 | 68.0 | 14.66 | 1.88 |
| knee | E8116 | 0.06360 | 251.08 | 15.98 | 130.0 | 14.66 | 6.28 |
| ankle_pitch | E4315 × para(2.0) | 0.06791 | 268.11 | 17.07 | 76.0 | 12.57 | 2.62 |
| ankle_roll | E4315 × para(2.0) | 0.06791 | 268.11 | 17.07 | 76.0 | 12.57 | 2.62 |

colosseum's two existing sets, for the same joints:

| group | DEPLOY kp / kd / eff | MJLAB kp / kd / eff | MJLAB armature |
|---|---|---|---|
| head (neck) | 4 / 1 / 7 | 1.78 / 0.23 / 7 | 0.00180 |
| arms | 50 / 1 / 18 | 27.9 / 3.56 / 36 | 0.02825 |
| waist | 200 / 5 / 30 | 47.2 / 6.01 / 40 | 0.04781 |
| hip_pitch | 200 / 5 / 45 | 51.7 / 6.58 / 55 | 0.05239 |
| hip_roll | 200 / 5 / 30 | 47.2 / 6.01 / 40 | 0.04781 |
| hip_yaw | 200 / 5 / 30 | 47.2 / 6.01 / 40 | 0.04781 |
| knee | 200 / 5 / 65 | 62.8 / 7.99 / 65 | 0.06360 |
| ankle_pitch | 50 / 2 / 24 | 33.6 / 4.27 / 50 | 0.03396 |
| ankle_roll | 50 / 2 / 15 | 33.6 / 4.27 / 50 | 0.03396 |

### 2.1 Armature

colosseum's reflected-inertia calc reproduces the booster_train base-motor
armatures almost exactly (0.00180, 0.02825, 0.04781, 0.05239, 0.06360 all line
up). **One exception — the ankles.** booster_train wraps each ankle in a
`BoosterT1AnkleParaWrapperCfg` with `armature_ratio = (2.0, 2.0)`, doubling the
base E4315 armature: **0.0340 → 0.0679**. This models the parallel ankle
linkage's reflected inertia. colosseum does not double it.

### 2.2 Stiffness / damping

`DEPLOY` legs (kp=200) are loosely in range of booster's 10 Hz values
(189–268) but flat and hand-set. `MJLAB` is exactly ¼ of booster (5 Hz vs
10 Hz). booster's damping is higher than DEPLOY's kd=5 everywhere on the legs
(12–17).

### 2.3 Effort limit — the largest substantive gap

booster_train's effort limits are well above *either* colosseum set:

| joint | DEPLOY | MJLAB (`peak_torque`) | booster_train |
|---|---|---|---|
| hip_pitch | 45 | 55 | **96** |
| hip_roll / yaw | 30 | 40 | **68** |
| knee | 65 | 65 | **130** (exactly 2×) |
| ankle_pitch | 24 | 50 | **76** |
| ankle_roll | 15 | 50 | **76** |
| arm | 18 | 36 | 38.3 |
| waist | 30 | 40 | **68** |
| neck | 7 | 7 | 7 |

**Open question:** colosseum's `MOTOR_SPECS.peak_torque` is consistently below
booster_train's datasheet effort limits — knee a clean 2×, hip_pitch ~1.75×.
Either colosseum's figures are a rated/continuous reading vs booster's true
peak, or they come from a different datasheet revision. **Resolve which effort
is correct before treating the ported numbers as ground truth**, since effort
drives both the torque clamp and (via the action-scale identity below) the
policy's torque authority.

### 2.4 Velocity limit + torque-speed (T-N) curve

colosseum has **no velocity-dependent torque limit at all**. booster_train's
`BoosterDelayedPDActuator._clip_effort` applies a piecewise-linear T-N curve:
full `effort_limit` up to `knee_point_velocity`, then linear down to zero at
`velocity_limit`. Relevant at the knee/hip during fast swing — a real motor
cannot hold peak torque at high joint speed, and our sim currently lets it.
This is the main piece that cannot be expressed as data in the current
colosseum actuator (see §4).

---

## 3. Action scale coupling

booster_train's `T1_ACTION_SCALE` (booster.py) uses `0.25 · effort / stiffness`
per joint — the **same formula** as colosseum's `mjlab_action_scale`.
colosseum's default `ACTION_SCALE` (constants.py) is instead a flat **0.25**.

### The identity that makes this matter

With `target = scale·action + default`, the action's torque contribution is:

```
τ_action = kp·(target − q) ≈ kp · scale · action
```

so **torque-per-unit-action = `kp · scale`**. Substituting the booster scale:

```
kp · scale = stiffness · (0.25 · effort / stiffness) = 0.25 · effort
```

The stiffness cancels — the effort/stiffness recipe **decouples the policy's
torque authority from `kp`**. booster's per-action torque is exactly
`0.25 · effort`. colosseum's flat-0.25 deploy model does *not* have this:
there `kp·scale = 0.25·kp`, so authority tracks the chosen gain, not the motor.

### Torque-per-action (`kp · scale`) — what the policy actually feels

| joint | DEPLOY (`0.25·kp`) | booster_train (`0.25·effort`) | MJLAB (`0.25·peak`) |
|---|---|---|---|
| hip_pitch | 50.0 | 24.0 | 13.75 |
| hip_roll | 50.0 | 17.0 | 10.0 |
| hip_yaw | 50.0 | 17.0 | 10.0 |
| knee | 50.0 | 32.5 | 16.25 |
| ankle_pitch | 12.5 | 19.0 | 12.5 |
| ankle_roll | 12.5 | 19.0 | 12.5 |
| waist | 50.0 | 17.0 | 10.0 |
| arm | 12.5 | 9.6 | 9.0 |
| neck | 1.0 | 1.75 | 1.75 |

Three different authority *profiles*: DEPLOY is flat (50 across the whole leg,
because kp and scale are both uniform); booster is **effort-shaped** (knee 32.5
> hip_pitch 24 > hip_roll 17 — more authority where the motor is genuinely
stronger); MJLAB has booster's shape but lower magnitude (lower `peak_torque`
efforts).

booster_train T1 action-scale dict, computed with their armatures/efforts:

| joint(s) | action scale |
|---|---|
| AAHead_yaw, Head_pitch | 0.2463 |
| *_Shoulder_*, *_Elbow_* | 0.0859 |
| Waist | 0.0901 |
| *_Hip_Pitch | 0.1160 |
| *_Hip_Roll, *_Hip_Yaw | 0.0901 |
| *_Knee_Pitch | 0.1294 |
| *_Ankle_Pitch, *_Ankle_Roll | 0.0709 |

> These scales are inseparable from booster's efforts **and** their
> doubled-armature ankle. Computed against colosseum's current ankle armature
> (0.0340) the ankle scale would be 0.1418, not 0.0709. Scale and actuator set
> are not independent.

---

## 4. What ports as data vs what needs the class refactor

> This section is the pre-refactor analysis that motivated §5. At the time,
> colosseum used `BuiltinPositionActuatorCfg`; the conclusion (`velocity_limit`
> and the T-N curve need a custom actuator) is what §5 implements.

colosseum's actuator was mjlab's `BuiltinPositionActuatorCfg`: an **implicit**
PD servo computed inside MuJoCo. Its fields are:

```
target_names_expr, transmission_type, armature, frictionloss,
viscous_damping, delay_min_lag, delay_max_lag, delay_hold_prob,
delay_update_period, delay_per_env_phase, stiffness, damping, effort_limit
```

### Ports 1-to-1 as data (into BuiltinPositionActuatorCfg)

- `stiffness` ← booster computed kp (10 Hz)
- `damping` ← booster computed kd (10 Hz, ζ=2)
- `effort_limit` ← booster datasheet effort
- `armature` ← booster armature (including the doubled ankle)

### Cannot be expressed as data — needs the class refactor

- **`velocity_limit`**: no field on `BuiltinPositionActuatorCfg`.
- **`knee_point_velocity` / T-N torque-speed clipping**: no field and no
  mechanism. This is the core of booster's `BoosterDelayedPDActuator` and the
  most physically meaningful gap.

### Delay — now at the actuator level (matching booster_train)

- booster_train: `DelayBuffer` at the **actuator** level on pos+vel+effort,
  `min_delay=2`, `max_delay=8` *physics steps*, resampled per reset.
- colosseum: mjlab's base `Actuator` already buffers and delays the command
  (`apply_delay`, called per physics substep by the entity) via the
  `delay_min_lag`/`delay_max_lag`/`delay_hold_prob`/`delay_update_period` cfg
  fields. `BoosterPdActuatorCfg` defaults these to 2–8 substeps, so the delay is
  the same layer and lag as booster_train.
- The old action-level `DelayedJointPositionAction` was **removed** as redundant:
  it modeled the same policy→motor latency one layer up (stacking two delays
  would double-count). Its only loss was the steady-per-episode + sub-step
  resolution regime; mjlab's buffer resamples per substep instead.

---

## 5. Implemented (actuator class + single set)

The refactor has landed in `robots/t1/actuators.py`:

- **`BoosterPdActuator` / `BoosterPdActuatorCfg`** subclass mjlab's explicit
  `IdealPdActuator`. They add `velocity_limit` + `knee_point_velocity` and
  override `_clip_effort` with booster's flat-top T-N curve (full `effort_limit`
  for `|v| <= knee_point_velocity`, linear to zero at `velocity_limit`). This is
  the one piece the built-in position actuator cannot express.
- **`BoosterJoint`** is the lean datasheet motor model (effort / velocity_limit /
  knee_point_velocity / armature, with `kp = I·(2πf)²`, `kd = 2ζI·(2πf)` derived
  at f = 10 Hz, ζ = 2). It ports the *concept* of booster_train's
  `BoosterJointCfg`, not its multi-robot class hierarchy. The six T1 motors are
  named instances (`E4310`, `E6408`, `E8112`, `E8116`, `DM4310`, `E4315`).
- **The ankles are parallel.** `parallel()` reproduces
  `ParallelJointWrapperCfg`'s ratio math; the T1 ankle pitch/roll are built from
  the `E4315` base with the T1 armature ratio (×2), as distinct `serial_index`
  specs.
- **One mjlab actuator per joint** (23 total): a `{joint: BoosterJoint}` map
  expands to 23 `BoosterPdActuatorCfg`. booster_train's limb-grouping is not
  replicable in mjlab (its `ActuatorCfg` gains are scalar, one cfg = one gain
  set), so per-joint is the natural and cleanest structure.
- **`action_scale()`** lives here too, computing booster's `0.25 · effort /
  stiffness` directly off the actuator set.

`BoosterDelayedActuatorCfg` was deliberately **not** ported — it is an IsaacLab
adapter that unpacks motor objects into per-regex gain dicts, which the
per-joint `BoosterPdActuatorCfg` makes unnecessary.

`MotorSpec`/`MOTOR_SPECS`, `DEPLOY_ACTUATORS`, `MJLAB_ACTUATORS`, and the
`BuiltinPositionActuatorCfg` set are removed. `constants.py` exposes the single
`ACTUATORS` and `ACTION_SCALE = action_scale()`.
