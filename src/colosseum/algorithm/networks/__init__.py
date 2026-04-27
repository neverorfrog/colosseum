"""Generic MLP backbone network.

Provides _resolve_activation() and the Network base class used by PPO networks.
"""

from typing import Callable, Sequence, cast

import numpy as np
import torch.nn as nn

from colosseum.config.types.networks import (
  NetworkConfig,  # noqa: F401 (re-exported for convenience)
)


def _resolve_activation(
  activation: str | Callable[[], nn.Module],
) -> Callable[[], nn.Module]:
  """Resolve activation from string or callable without fragile getattr cascades.

  Supported strings (case-insensitive): relu, tanh, gelu, silu. Custom callables
  can be passed directly (e.g., lambda: nn.LeakyReLU(0.1)).
  """
  if callable(activation):
    return cast(Callable[[], nn.Module], activation)

  name = activation.lower()
  registry: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
  }
  if name in registry:
    return registry[name]

  raise ValueError(
    f"Unsupported activation '{activation}'. Add it to the registry if needed."
  )


class Network(nn.Module):
  """Generic MLP backbone built from config.

  This handles stacking Linear + Activation layers based on `hidden_layers` and
  `activation` from the config. Children add their own heads on top.
  """

  last_dim: int

  def __init__(
    self,
    input_dim: int,
    hidden_layers: Sequence[int],
    activation_name: str | Callable[[], nn.Module],
  ):
    super().__init__()
    activation_cls = _resolve_activation(activation_name)
    layer_sizes = [int(input_dim), *[int(h) for h in hidden_layers]]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(layer_sizes, layer_sizes[1:]):
      layers.append(nn.Linear(in_dim, out_dim))
      layers.append(activation_cls())
    self.backbone = nn.Sequential(*layers)
    # Stash last layer size; object.__setattr__ avoids nn.Module __setattr__ typing complaints
    object.__setattr__(self, "last_dim", int(layer_sizes[-1]))

    # Initialize weights with orthogonal initialization (rsl_rl style)
    self._init_weights()

  def _init_weights(self) -> None:
    """Initialize network weights with orthogonal initialization.

    Matches rsl_rl's MLP initialization for better initial performance.
    Uses orthogonal init with gain=sqrt(2) for ReLU activations.
    """
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        nn.init.zeros_(module.bias)
