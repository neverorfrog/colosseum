"""Empirical normalization for observations (from rsl-rl)."""

from abc import abstractmethod

import torch
import torch.nn as nn


class ObsNormalizer(nn.Module):
  """Base class for observation normalizers."""

  @abstractmethod
  def update(self, x: torch.Tensor) -> None: ...


class IdentityNormalizer(ObsNormalizer):
  """Pass-through normalizer (no-op)."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x

  def update(self, x: torch.Tensor) -> None:
    pass


class EmpiricalNormalization(ObsNormalizer):
  """Normalize mean and variance of values based on empirical values.

  Adapted from RSL-RL's EmpiricalNormalization for use with SAC.
  """

  def __init__(self, shape, eps=1e-2, until=None, device=None):
    """Initialize EmpiricalNormalization module.

    Args:
        shape (int or tuple of int): Shape of input values except batch axis.
        eps (float): Small value for stability.
        until (int or None): If this arg is specified, the module learns input values until the sum of batch sizes
            exceeds it.
        device (str or torch.device, optional): Device to place buffers on. If None, uses CPU.

    Note: The normalization parameters are computed over the whole batch, not for each environment separately.
    """
    super().__init__()
    self.eps: float = eps
    self.until: int | None = until

    # Create buffers on specified device
    if device is None:
      device = torch.device("cpu")
    elif isinstance(device, str):
      device = torch.device(device)

    self._mean: torch.Tensor
    self._var: torch.Tensor
    self._std: torch.Tensor
    self.count: torch.Tensor
    self.register_buffer("_mean", torch.zeros(shape, device=device).unsqueeze(0))
    self.register_buffer("_var", torch.ones(shape, device=device).unsqueeze(0))
    self.register_buffer("_std", torch.ones(shape, device=device).unsqueeze(0))
    self.register_buffer("count", torch.tensor(0, dtype=torch.long, device=device))

  @property
  def mean(self):
    return self._mean.squeeze(0).clone()

  @property
  def std(self):
    return self._std.squeeze(0).clone()

  def forward(self, x):
    """Normalize mean and variance of values based on empirical values."""

    return (x - self._mean) / (self._std + self.eps)

  @torch.jit.unused
  def update(self, x: torch.Tensor) -> None:
    """Learn input values without computing the output values of them"""

    if not self.training:
      return
    if self.until is not None and self.count >= self.until:
      return

    # Compute batch moments. In distributed mode we aggregate moments across
    # workers so all ranks keep identical normalization statistics.
    count_x = torch.tensor(x.shape[0], dtype=torch.long, device=x.device)
    sum_x = torch.sum(x, dim=0, keepdim=True)
    sum_x2 = torch.sum(x * x, dim=0, keepdim=True)

    if (
      torch.distributed.is_available()
      and torch.distributed.is_initialized()
      and torch.distributed.get_world_size() > 1
    ):
      torch.distributed.all_reduce(count_x, op=torch.distributed.ReduceOp.SUM)
      torch.distributed.all_reduce(sum_x, op=torch.distributed.ReduceOp.SUM)
      torch.distributed.all_reduce(sum_x2, op=torch.distributed.ReduceOp.SUM)

    if count_x.item() <= 0:
      return

    count_x_float = count_x.to(dtype=x.dtype)
    mean_x = sum_x / count_x_float
    var_x = torch.clamp(sum_x2 / count_x_float - mean_x * mean_x, min=0.0)

    self.count += count_x
    rate = count_x_float / self.count.to(dtype=x.dtype)
    delta_mean = mean_x - self._mean
    self._mean += rate * delta_mean
    self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
    self._std = torch.sqrt(self._var)

  @torch.jit.unused
  def inverse(self, y):
    """De-normalize values based on empirical values."""

    return y * (self._std + self.eps) + self._mean
