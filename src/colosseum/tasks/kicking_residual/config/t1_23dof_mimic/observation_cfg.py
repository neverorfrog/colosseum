"""Observation configuration for t1-kicking-residual-mimic.

Identical to the ``t1`` residual task's observation layout, plus 4
reference-motion terms (``motion_command``, ``motion_anchor_pos_b``,
``motion_anchor_ori_b``, ``motion_triggered``) added to ``kick_actor`` and
``critic``. These establish "how to move" once ``GatedHoldMotionCommand``
triggers near the ball; ``motion_triggered`` additionally tells the
orchestrator when to favor the kick branch.

``orchestrator_terms`` is recomputed as the deduplicated union of
``loco_actor_terms`` and the extended ``kick_actor_terms``, so the 4 new terms
flow into the deployable orchestrator automatically.

The 4 new terms use ``mirror_fn=None`` (treated as invariant under left-right
mirroring) — true L-R remapping of the reference clip is out of scope for v1.
"""

from mjlab.envs.mdp.observations import generated_commands
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.tracking.mdp.observations import motion_anchor_ori_b, motion_anchor_pos_b
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from colosseum.mdp.motion_command import motion_triggered
from colosseum.mdp.symmetry import MirrorableObservationTermCfg

from ..t1.observation_cfg import (
  critic_terms as _base_critic_terms,
)
from ..t1.observation_cfg import (
  kick_actor_terms as _base_kick_actor_terms,
)
from ..t1.observation_cfg import (
  loco_actor_terms,
)

# Reference-motion terms, added to both the trainable residual branch and the
# (privileged) critic.
motion_terms = {
  "motion_command": MirrorableObservationTermCfg(
    func=generated_commands,
    params={"command_name": "motion"},
  ),
  "motion_anchor_pos_b": MirrorableObservationTermCfg(
    func=motion_anchor_pos_b,
    params={"command_name": "motion"},
    noise=Unoise(n_min=-0.25, n_max=0.25),
  ),
  "motion_anchor_ori_b": MirrorableObservationTermCfg(
    func=motion_anchor_ori_b,
    params={"command_name": "motion"},
    noise=Unoise(n_min=-0.05, n_max=0.05),
  ),
  "motion_triggered": MirrorableObservationTermCfg(
    func=motion_triggered,
    params={"command_name": "motion"},
  ),
}

kick_actor_terms = {**_base_kick_actor_terms, **motion_terms}

# Orchestrator (gating net): deduplicated union of all skill observations,
# now including the reference-motion terms so it knows when mimic mode is
# active.
orchestrator_terms = {**loco_actor_terms, **kick_actor_terms}

critic_terms = {**_base_critic_terms, **motion_terms}

observations = {
  "loco_actor": ObservationGroupCfg(
    terms=loco_actor_terms,
    concatenate_terms=True,
    enable_corruption=True,
  ),
  "kick_actor": ObservationGroupCfg(
    terms=kick_actor_terms,
    concatenate_terms=True,
    enable_corruption=True,
  ),
  "orchestrator": ObservationGroupCfg(
    terms=orchestrator_terms,
    concatenate_terms=True,
    enable_corruption=False,
  ),
  "critic": ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  ),
}
