"""Expert-transition dataset for AMP (Adversarial Motion Priors).

Loads one or more BeyondMimic-format ``.npz`` motion clips and yields
``(state_t, state_{t+1})`` transitions for the AMP discriminator's "expert"
branch. A *state* here is the AMP observation vector — NOT the RL observation.
For the kick we use the "classic" layout::

    [ joint_pos_rel (23), joint_vel (23), base_lin_vel_yaw (3), base_ang_vel_yaw (3) ]  = 52

This must match, element-for-element, the env's ``amp`` observation group, or the
discriminator separates expert from policy on a layout/convention artifact and the
AMP reward collapses. Three deliberate departures from stock beyondAMP keep the two
sides aligned (see the kick npz trace):

  Fix A — joint angles are stored ABSOLUTE in the npz, but mjlab's ``joint_pos_rel``
          emits ``joint_pos - default_joint_pos``. We subtract the robot default here
          so both sides are relative.
  Fix B — base velocities are rotated into the full base frame to match mjlab's
          built-in ``base_lin_vel`` / ``base_ang_vel`` obs terms (``root_link_*_vel_b``),
          which is also how the robot's own proprioception represents base velocity.
  Fix C — the anchor body is matched by ``body_names`` recorded in the npz, not via
          ``robot.find_bodies`` (which would assume npz body order == robot body order).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse


class AmpMotionDataset:
  """Holds expert AMP transitions and samples random ``(s_t, s_{t+1})`` minibatches."""

  def __init__(
    self,
    motion_files: list[str],
    default_joint_pos: torch.Tensor,
    device: torch.device | str,
    anchor_body: str = "Trunk",
    include_base_vel: bool = True,
    joint_ids: list[int] | None = None,
  ) -> None:
    """
    Args:
      motion_files: BeyondMimic-format ``.npz`` paths. Each must carry
        ``joint_pos (T, J)``, ``joint_vel (T, J)``, ``body_quat_w (T, B, 4)``,
        ``body_lin_vel_w (T, B, 3)``, ``body_ang_vel_w (T, B, 3)``, ``body_names (B,)``.
      default_joint_pos: ``(J,)`` robot default joint positions, in the robot's joint
        order (Fix A). Pull from ``env.scene["robot"].data.default_joint_pos[0]``.
      device: target device for the cached tensors.
      anchor_body: name of the base/root body in the npz (Fix C). T1: ``"Trunk"``.
      include_base_vel: if True, append ``[base_lin_vel(3), base_ang_vel(3)]`` to each
        state ("classic" layout, 2J+6). If False, states are joints-only ("basic"
        layout, 2J) so the discriminator is speed-agnostic. Must match the env's
        ``amp`` obs group (``amp_classic_obs_group`` vs ``amp_basic_obs_group``).
      joint_ids: optional column indices (into the robot's joint order) to keep,
        so the expert joints match a filtered env ``amp`` group (e.g. head excluded).
        ``None`` keeps all joints. Must mirror the obs group's ``joint_names``.
    """
    self.device = torch.device(device)
    self._default_joint_pos = default_joint_pos.to(self.device).reshape(1, -1)
    self._anchor_body = anchor_body
    self._include_base_vel = include_base_vel
    self._joint_ids = joint_ids

    states: list[torch.Tensor] = []
    index_t: list[torch.Tensor] = []
    index_tp1: list[torch.Tensor] = []
    offset = 0

    for path in motion_files:
      data = np.load(path, allow_pickle=True)
      state = self._build_states(data)  # (T, 52)
      states.append(state)

      t_len = state.shape[0]
      if t_len >= 2:
        # Fix transition indices so (t, t+1) never crosses a clip boundary.
        t = torch.arange(offset, offset + t_len - 1, device=self.device)
        index_t.append(t)
        index_tp1.append(t + 1)
      offset += t_len

    self.states = torch.cat(states, dim=0)  # (N, 52)
    self.index_t = torch.cat(index_t, dim=0)
    self.index_tp1 = torch.cat(index_tp1, dim=0)
    self.observation_dim = self.states.shape[-1]

  def _build_states(self, data) -> torch.Tensor:
    """Assemble the per-frame 52-dim AMP state from one clip."""
    body_names = [str(n) for n in data["body_names"]]
    anchor_idx = body_names.index(self._anchor_body)  # Fix C: match by name

    def t(key: str) -> torch.Tensor:
      return torch.tensor(data[key], dtype=torch.float32, device=self.device)

    joint_pos = t("joint_pos") - self._default_joint_pos  # Fix A: absolute -> relative
    joint_vel = t("joint_vel")

    if self._joint_ids is not None:  # keep only the joints the env amp group emits
      joint_pos = joint_pos[:, self._joint_ids]
      joint_vel = joint_vel[:, self._joint_ids]

    if not self._include_base_vel:
      return torch.cat([joint_pos, joint_vel], dim=-1)  # "basic" layout (2J)

    # Fix B: trunk world-frame velocities rotated into the full base frame
    # (matches mjlab's base_lin_vel / base_ang_vel == root_link_*_vel_b).
    anchor_quat_w = t("body_quat_w")[:, anchor_idx]  # (T, 4)
    anchor_lin_w = t("body_lin_vel_w")[:, anchor_idx]  # (T, 3)
    anchor_ang_w = t("body_ang_vel_w")[:, anchor_idx]  # (T, 3)
    base_lin_vel = quat_apply_inverse(anchor_quat_w, anchor_lin_w)
    base_ang_vel = quat_apply_inverse(anchor_quat_w, anchor_ang_w)

    return torch.cat([joint_pos, joint_vel, base_lin_vel, base_ang_vel], dim=-1)

  def feed_forward_generator(
    self, num_mini_batches: int, mini_batch_size: int
  ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``num_mini_batches`` random ``(state_t, state_{t+1})`` expert minibatches."""
    for _ in range(num_mini_batches):
      idx = torch.randint(0, self.index_t.shape[0], (mini_batch_size,), device=self.device)
      yield self.states[self.index_t[idx]], self.states[self.index_tp1[idx]]
