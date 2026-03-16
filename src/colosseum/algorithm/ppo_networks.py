"""PPO networks: Gaussian actor and value function critic.

Key differences from SAC networks:
- Actor: No tanh squashing, state-independent learned std parameter
- Critic: Outputs scalar value V(s) instead of Q(s,a)

Initialization follows RSL-RL/holosoma conventions:
- Backbone: orthogonal init with gain=sqrt(2)
- Actor mean head: orthogonal init with std=0.01 (near-zero initial actions)
- Critic value head: orthogonal init with std=1.0
- std parameterized directly (not log_std) for self-limiting gradient (1/std)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.algorithm.networks import Network
from colosseum.config.types.networks import PpoActorConfig, PpoCriticConfig


class PpoActor(Network):
    """Gaussian policy for PPO with state-independent std.

    Architecture:
        obs -> backbone (MLP) -> mean_head -> action_mean
        std: learned nn.Parameter (broadcast across batch)

    Uses direct std parameterization (RSL-RL/holosoma pattern) instead of
    log_std. Gradient w.r.t. std is 1/std, which is self-limiting: as std
    grows, gradient shrinks, preventing unbounded entropy growth.

    No tanh squashing: MuJoCo envs clip actions internally.
    """

    config: PpoActorConfig

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PpoActorConfig | None = None,
    ):
        cfg = config or PpoActorConfig()
        object.__setattr__(self, "config", cfg)

        super().__init__(
            input_dim=obs_dim,
            hidden_layers=cfg.hidden_layers,
            activation_name=cfg.activation,
        )

        self.mean_head = nn.Linear(int(self.last_dim), action_dim)
        self.std = nn.Parameter(cfg.init_noise_std * torch.ones(action_dim))
        self.min_noise_std = cfg.min_noise_std  # type: ignore[assignment]

        # Re-init specific heads (backbone already done by Network._init_weights)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

        # Disable default validation for speed (RSL-RL pattern)
        torch.distributions.Normal.set_default_validate_args(False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action mean."""
        return self.mean_head(self.backbone(obs))

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Return backbone features (for auxiliary heads)."""
        return self.backbone(obs)

    def get_distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Build Normal distribution from obs."""
        mean = self.forward(obs)
        std = torch.clamp(self.std, min=self.min_noise_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample stochastic action (for rollout collection)."""
        return self.get_distribution(obs).sample()

    def act_with_log_prob(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return all quantities in a single forward pass.

        Used during rollout collection to avoid redundant forward passes.
        RSL-RL caches these on the transition object; we return them directly.

        Returns:
            actions: [batch, action_dim] sampled actions
            log_probs: [batch] log probability of sampled actions
            means: [batch, action_dim] distribution means
            stds: [batch, action_dim] distribution stds
        """
        mean = self.forward(obs)
        std = torch.clamp(self.std, min=self.min_noise_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs, mean, std

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (mean) for evaluation."""
        return self.forward(obs)

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy for given (obs, actions) pairs.

        Used during PPO update to re-evaluate stored actions.

        Returns:
            log_prob: [batch] log probability of actions
            entropy: [batch] entropy of distribution
        """
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def get_action(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action with log probability and deterministic action.

        Used for inference and evaluation to match SAC interface.

        Returns:
            stochastic_action: [batch, action_dim] sampled action
            log_prob: [batch, 1] log probability of sampled action
            deterministic_action: [batch, action_dim] mean action (deterministic)
        """
        dist = self.get_distribution(obs)
        stochastic_action = dist.sample()
        log_prob = dist.log_prob(stochastic_action).sum(dim=-1, keepdim=True)
        deterministic_action = self.forward(obs)
        return stochastic_action, log_prob, deterministic_action

    def get_distribution_params(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) for KL divergence computation.

        Stored during rollout for adaptive KL LR scheduling.
        """
        mean = self.forward(obs)
        std = torch.clamp(self.std, min=self.min_noise_std).expand_as(mean)
        return mean, std


class PpoValueNet(Network):
    """Value function V(s) for PPO critic.

    Architecture:
        obs -> backbone (MLP) -> value_head -> scalar value
    """

    config: PpoCriticConfig

    def __init__(
        self,
        obs_dim: int,
        config: PpoCriticConfig | None = None,
    ):
        cfg = config or PpoCriticConfig()
        object.__setattr__(self, "config", cfg)

        super().__init__(
            input_dim=obs_dim,
            hidden_layers=cfg.hidden_layers,
            activation_name=cfg.activation,
        )

        self.value_head = nn.Linear(int(self.last_dim), 1)

        # Re-init value head (backbone already done by Network._init_weights)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns value estimate [batch, 1]."""
        return self.value_head(self.backbone(obs))
