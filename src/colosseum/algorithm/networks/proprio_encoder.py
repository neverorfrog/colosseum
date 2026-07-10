"""Proprioceptive history encoder for classic (non-visual) RMA.

ProprioWindowEncoder — Conv1D over a rolling window of proprioceptive
observations, equivalent in capacity to the adaptation module in the
the original RMA paper.

Design notes
------------
- Input window: (B, T, obs_dim).  T is treated as the sequence length;
  obs_dim is the channel dimension fed into Conv1d.
- Vectorised training path: pad-then-unfold avoids a Python for-loop
  over time steps, giving a single batched Conv1d call.
- No internal state: unlike DepthEncoder (GRU), this module is stateless.
  The caller (RmaTerm) owns and maintains the rolling obs buffer.
- Output LayerNorm: matches PrivilegedEncoder so z_adapt and z_priv share
  the same output manifold and MSE regression is well-scaled.

Comparison with GRU (DepthEncoder)
------------------------------------
GRU       — stateful, handles arbitrary-length history, memory-efficient at
             inference (only hidden state needed), but gradients vanish over
             long windows and needs warmup steps after episode resets.
Conv1D    — stateless, operates on an explicit fixed-size window, better
             gradient flow within the window, no warmup needed (padding
             fills the gap at the start of an episode), but requires the
             caller to maintain and store a (B, T_window, obs_dim) buffer.

For classic RMA (static env params), Conv1D is arguably more appropriate:
the encoder just needs to identify the average physics effect on recent
observations, with no need to track time-varying dynamics.  GRU wastes
capacity learning to do that.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ProprioWindowEncoder(nn.Module):
  """Conv1D adaptation encoder for classic RMA (proprioceptive history).

  Single-step inference path (caller maintains rolling buffer):
    window (B, T_window, obs_dim) -> z (B, latent_dim)

  Sequence training path (vectorised, no for-loop):
    obs_seq (B, T_collect, obs_dim) -> z_seq (B, T_collect, latent_dim)
    Internally creates windowed views via pad + unfold.

  Args:
    obs_dim:         Proprioceptive observation dimension.
    latent_dim:      Output latent dimension (match PrivilegedEncoder's).
    window_size:     Number of past steps in the window (default 50).
    hidden_channels: Sizes of the two Conv1d channel dimensions.
  """

  def __init__(
    self,
    obs_dim: int,
    latent_dim: int = 8,
    window_size: int = 50,
    hidden_channels: tuple[int, int] = (32, 16),
  ) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    self.latent_dim = latent_dim
    self.window_size = window_size

    ch0, ch1 = hidden_channels
    self.conv = nn.Sequential(
      # obs_dim channels, window_size sequence length
      nn.Conv1d(obs_dim, ch0, kernel_size=3, padding=1),
      nn.ELU(),
      nn.Conv1d(ch0, ch1, kernel_size=3, padding=1),
      nn.ELU(),
      nn.AdaptiveAvgPool1d(1),
      nn.Flatten(),
      nn.Linear(ch1, latent_dim),
    )
    # Match PrivilegedEncoder's output norm so z_adapt lives on the same
    # manifold as z_priv and MSE regression starts in a reasonable range.
    self.output_norm = nn.LayerNorm(latent_dim)

  # ------------------------------------------------------------------
  # Single-step inference (deployment / per-step encoding)
  # ------------------------------------------------------------------

  def forward(self, window: Tensor) -> Tensor:
    """Encode a single observation window.

    Args:
      window: (B, T, obs_dim) — the last T proprio observations.
              T may be less than window_size (e.g. at episode start);
              the caller should pre-pad if needed.

    Returns:
      z: (B, latent_dim)
    """
    # (B, T, obs_dim) -> (B, obs_dim, T) for Conv1d
    x = window.transpose(-2, -1)
    return self.output_norm(self.conv(x))

  # ------------------------------------------------------------------
  # Sequence training (vectorised pad-then-unfold)
  # ------------------------------------------------------------------

  def encode_sequence(
    self,
    obs_seq: Tensor,
    reset_mask: Tensor | None = None,
  ) -> Tensor:
    """Encode a full collected rollout as windowed Conv1d views.

    For each time step t, the window is obs_seq[:, t-W+1 : t+1] left-padded
    by replicating obs_seq[:, 0] (same as ObservationsWrapper's done-env
    reset: fill all slots with the first observation of the new episode).

    Cross-episode contamination (observations from the previous episode
    leaking into the window of early steps in a new episode) is handled
    by the caller through the loss_mask: the first window_size steps after
    each reset are excluded from the loss rather than requiring special
    padding logic here.

    Args:
      obs_seq:    (B, T_collect, obs_dim) collected observations.
      reset_mask: (B, T_collect) bool — True where an episode reset
                  occurred.  Currently unused inside this method; the
                  caller (RmaTerm.compute_loss) applies the mask to the
                  returned z_seq when computing the MSE loss.

    Returns:
      z_seq: (B, T_collect, latent_dim)
    """
    B, T, D = obs_seq.shape
    W = self.window_size

    # Pad left with (W-1) copies of the first observation so that step 0
    # gets a full-width window of identical obs (no information leakage,
    # and matches the reset-fill behaviour of the reference wrapper).
    first = obs_seq[:, :1, :].expand(-1, W - 1, -1)  # (B, W-1, D)
    padded = torch.cat([first, obs_seq], dim=1)  # (B, T+W-1, D)

    # unfold(dim=1, size=W, step=1): sliding windows along the time axis.
    # Result: (B, T, D, W)  [torch puts the new "window" dim last]
    windows = padded.unfold(1, W, 1)  # (B, T, D, W)

    # Reshape to a single batched Conv1d call: (B*T, D, W)
    windows = windows.reshape(B * T, D, W)

    z_flat = self.output_norm(self.conv(windows))  # (B*T, latent_dim)
    return z_flat.reshape(B, T, self.latent_dim)
