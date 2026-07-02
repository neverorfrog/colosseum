"""Algorithm config for t1-dribbling-residual-kick: ResidualPPO, per-joint gate.

Same frozen walk + trainable residual as the base t1-dribbling-residual, with two
changes for the single-strong-kick variant:
  - ``orchestrator.per_joint=True``: the gate blends skills independently per
    joint, so the residual can take authority on just the swing-leg joints to
    elongate the strike stride without disturbing the rest of the walk.
  - ``init_favored_logit=2.0`` (down from 4.0): the per-joint gate starts less
    biased toward the frozen base, so it can actually open on the kicking leg.
"""

from dataclasses import replace

from ..t1_23dof.algo_cfg import booster_t1_residual_ppo_cfg as _base_cfg


def booster_t1_residual_ppo_cfg():
  base = _base_cfg()
  ra = base.residual_actor
  new_ra = replace(
    ra,
    orchestrator=replace(ra.orchestrator, per_joint=True),
    init_favored_logit=2.0,
  )
  return replace(base, residual_actor=new_ra)
