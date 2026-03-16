from __future__ import annotations

from dataclasses import field
from typing import List

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class NetworkConfig:
    """Base configuration for neural networks."""

    learning_rate: float = 3e-4
    """Learning rate for the network."""

    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    """List of hidden layer sizes."""

    activation: str = "relu"
    """Activation function to use in the network (e.g., 'relu', 'tanh')."""


@dataclass(frozen=True)
class PpoActorConfig(NetworkConfig):
    """Configuration for PPO actor (Gaussian policy) network."""

    init_noise_std: float = 1.0
    """Initial standard deviation of the action noise (state-independent)."""

    min_noise_std: float = 0.01
    """Minimum noise std clamp (holosoma pattern, prevents std collapsing to zero)."""


@dataclass(frozen=True)
class PpoCriticConfig(NetworkConfig):
    """Configuration for PPO critic (value function) network."""

    pass
