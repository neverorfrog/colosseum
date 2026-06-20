"""AMP observation group factory.

The discriminator's "state" must match ``AmpMotionDataset`` element-for-element.
Two layouts, both built from stock mjlab terms (no custom obs functions) so the
env side and the dataset stay aligned by construction:

  "classic"::

      [ joint_pos_rel (J), joint_vel_rel (J), base_lin_vel (3), base_ang_vel (3) ]

  "basic" (beyondAMP velocity recipe)::

      [ joint_pos_rel (J), joint_vel_rel (J) ]

The "basic" layout drops the base velocities so the discriminator is
*speed-agnostic*: it shapes posture/gait pattern but not body speed, letting the
velocity command set the speed freely (so a single, slow reference clip doesn't
fight the commanded velocity range). Pair it with ``include_base_vel=False`` on
``AmpMotionDataset`` so the expert side matches.

Term conventions (both layouts):
  * ``joint_pos_rel`` subtracts ``default_joint_pos`` (the dataset does the same: Fix A).
  * ``joint_vel_rel`` subtracts ``default_joint_vel`` (zero for T1; dataset uses raw).
  * ``base_lin_vel`` / ``base_ang_vel`` are ``root_link_*_vel_b`` — the full base frame
    the dataset rotates into (Fix B).

Attach to a task's observation dict as the ``"amp"`` group (corruption off — the
discriminator should see clean motion). It is NOT a skill group, so it never enters
the ResidualActor / orchestrator / symmetry paths.
"""

from __future__ import annotations

from collections.abc import Sequence

from mjlab.envs.mdp import observations as mdp
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg

# Body joints fed to the AMP discriminator: everything except the 2 head joints.
# The head is randomly perturbed during training (head_perturb) but static in the
# reference clip, so including it hands the discriminator a trivial,
# gait-irrelevant separator. Pass to the obs group's ``joint_names`` and mirror it
# on the dataset (``AmpPpoConfig.amp_joint_names``) so both sides stay aligned.
AMP_BODY_JOINTS = ("^(?!AAHead_yaw$|Head_pitch$).*$",)


def _amp_joint_terms(
  joint_names: Sequence[str] | None = None,
) -> dict[str, ObservationTermCfg]:
  # Separate cfg per term: the manager resolves (mutates) each one's joint_ids.
  return {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)},
    ),
  }


def amp_classic_obs_terms(
  joint_names: Sequence[str] | None = None,
) -> dict[str, ObservationTermCfg]:
  return {
    **_amp_joint_terms(joint_names),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
  }


def amp_basic_obs_terms(
  joint_names: Sequence[str] | None = None,
) -> dict[str, ObservationTermCfg]:
  return _amp_joint_terms(joint_names)


def _amp_obs_group(terms: dict[str, ObservationTermCfg], **group_kwargs) -> ObservationGroupCfg:
  group_kwargs.setdefault("concatenate_terms", True)
  group_kwargs.setdefault("enable_corruption", False)
  return ObservationGroupCfg(terms=terms, **group_kwargs)


def amp_classic_obs_group(
  joint_names: Sequence[str] | None = None, **group_kwargs
) -> ObservationGroupCfg:
  return _amp_obs_group(amp_classic_obs_terms(joint_names), **group_kwargs)


def amp_basic_obs_group(
  joint_names: Sequence[str] | None = None, **group_kwargs
) -> ObservationGroupCfg:
  """Joints-only AMP obs (speed-agnostic discriminator). 2*J dims.

  ``joint_names``: optional joint filter (default = all joints). Must be mirrored
  on the dataset via ``AmpPpoConfig.amp_joint_names`` so expert/policy align.
  """
  return _amp_obs_group(amp_basic_obs_terms(joint_names), **group_kwargs)
