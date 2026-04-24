from __future__ import annotations

from typing import Type, TYPE_CHECKING

from pydantic.dataclasses import dataclass

from colosseum.config.types.networks import (
    PpoActorConfig,
    PpoCriticConfig,
)


if TYPE_CHECKING:
    from colosseum.algorithm.base_algorithm import BaseAlgorithm

# Unified registry: algorithm name (lowercase) → (impl class, config class)
_ALGORITHM_REGISTRY: dict[str, tuple[Type["BaseAlgorithm"], Type["AlgorithmConfig"]]] = {}


def register_algorithm(name: str, config_class: Type["AlgorithmConfig"]):
    """Decorator to register an algorithm implementation alongside its config class.

    Usage:
        @register_algorithm("ppo", config_class=PpoConfig)
        class PPO(BaseAlgorithm):
            ...
    """

    def decorator(cls: Type["BaseAlgorithm"]) -> Type["BaseAlgorithm"]:
        _ALGORITHM_REGISTRY[name] = (cls, config_class)
        cls._algo_name = name
        return cls

    return decorator


def get_algorithm_class(name: str) -> Type["BaseAlgorithm"]:
    """Get algorithm implementation class by name (case-insensitive)."""
    key = name.lower()
    if key not in _ALGORITHM_REGISTRY:
        raise KeyError(
            f"Algorithm '{name}' not found. "
            f"Available: {list(_ALGORITHM_REGISTRY.keys())}"
        )
    return _ALGORITHM_REGISTRY[key][0]


def get_algorithm_config_class(name: str) -> Type["AlgorithmConfig"]:
    """Get algorithm config class by name (case-insensitive)."""
    key = name.lower()
    if key not in _ALGORITHM_REGISTRY:
        raise KeyError(
            f"Algorithm '{name}' not found. "
            f"Available: {list(_ALGORITHM_REGISTRY.keys())}"
        )
    return _ALGORITHM_REGISTRY[key][1]


@dataclass(frozen=True)
class AlgorithmConfig:
    """Configuration for the reinforcement learning algorithm."""

    name: str = "PPO"
    """Name of the RL algorithm (e.g., 'PPO')."""

    version: str = "v1"
    """Version of the algorithm."""

    target: str = "colosseum.algorithm.ppo:PPO"
    """Import path to the algorithm class."""

    learning_steps: int = 1_000_000
    """Total environment steps to run during training."""

    obs_normalization: bool = True
    """Whether to apply empirical normalization to observations."""

    seed: int | None = None
    """Optional global seed for reproducibility."""

    use_rich_logging: bool = True
    """Whether to use rich logging output in the console."""

    eval_interval: int = 0
    """How often (in env steps) to run evaluation. 0 disables evaluation."""

    eval_episodes: int = 10
    """Number of full episodes to run per evaluation."""

    eval_batch_size: int = 5
    """Number of parallel envs for evaluation."""

    @classmethod
    def reconstruct_from_dict(cls, data: dict) -> "AlgorithmConfig":
        """Reconstruct config from dict, handling nested configs."""
        return cls(**data)


@dataclass(frozen=True)
class PpoConfig(AlgorithmConfig):
    """Configuration for PPO (Proximal Policy Optimization).

    Combines patterns from RSL-RL, holosoma, and CleanRL:
    - Single joint optimizer for actor+critic (RSL-RL)
    - Adaptive KL learning rate scheduling (RSL-RL/holosoma)
    - GAE with timeout bootstrapping
    - Clipped surrogate and value losses
    """

    name: str = "PPO"
    target: str = "colosseum.algorithm.ppo:PPO"

    # On-policy rollout settings
    num_steps_per_env: int = 24
    """Steps collected per environment per rollout before an update."""

    # GAE
    gamma: float = 0.99
    """Discount factor for future rewards."""

    lam: float = 0.95
    """GAE lambda for advantage estimation."""

    # PPO clip
    clip_param: float = 0.2
    """PPO surrogate clipping parameter."""

    use_clipped_value_loss: bool = True
    """Whether to clip the value function loss (RSL-RL/holosoma style)."""

    # Loss coefficients
    value_loss_coef: float = 1.0
    """Coefficient for value function loss."""

    entropy_coef: float = 0.01
    """Coefficient for entropy bonus (encourages exploration)."""

    # Update epochs and mini-batches
    num_learning_epochs: int = 5
    """Number of PPO update epochs per rollout."""

    num_mini_batches: int = 4
    """Number of mini-batches per epoch."""

    # Gradient clipping
    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    # Learning rate (single optimizer, RSL-RL style)
    learning_rate: float = 1e-3
    """Learning rate for the joint actor+critic optimizer (RSL-RL style)."""

    # Adaptive KL learning rate scheduling (RSL-RL/holosoma style)
    desired_kl: float | None = 0.01
    """Target KL divergence for adaptive LR. None disables adaptive scheduling."""

    schedule: str = "adaptive"
    """LR schedule: 'adaptive' (KL-based) or 'fixed'."""

    # Advantage normalization
    normalize_advantage_per_mini_batch: bool = False
    """If True, normalize advantages per mini-batch. If False, normalize globally."""

    # Network configs
    actor: PpoActorConfig = PpoActorConfig()
    """Configuration for the PPO actor network."""

    critic: PpoCriticConfig = PpoCriticConfig()
    """Configuration for the PPO critic network."""

    @classmethod
    def reconstruct_from_dict(cls, data: dict) -> "PpoConfig":
        """Reconstruct PpoConfig from dict, handling nested network configs."""
        if "actor" in data and isinstance(data["actor"], dict):
            data["actor"] = PpoActorConfig(**data["actor"])
        if "critic" in data and isinstance(data["critic"], dict):
            data["critic"] = PpoCriticConfig(**data["critic"])
        return cls(**data)


@dataclass(frozen=True)
class DaggerPpoConfig(PpoConfig):
    """PPO with DAgger-style teacher-student imitation loss.

    Extends PpoConfig with a pre-trained teacher policy whose observation is a
    contiguous suffix of the student's observation tensor:

        student_obs[:, teacher_obs_start_idx:] == teacher_obs

    The imitation loss λ·MSE(student_mean, teacher_action) is added to the
    PPO objective and annealed linearly to zero over imitation_annealing_steps.
    """

    name: str = "DaggerPPO"
    target: str = "colosseum.algorithm.dagger_ppo:DaggerPPO"

    teacher_checkpoint: str = ""
    """Path to the pre-trained teacher policy checkpoint (.pt)."""

    teacher_obs_dim: int = 78
    """Observation dimension expected by the teacher network."""

    teacher_obs_start_idx: int = 7
    """Index into student obs where the teacher obs slice starts."""

    teacher_actor_hidden_layers: tuple[int, ...] = (512, 256, 128)
    """Hidden layer sizes of the teacher actor (must match checkpoint)."""

    teacher_actor_activation: str = "elu"
    """Activation function of the teacher actor (must match checkpoint)."""

    imitation_coef: float = 1.0
    """Initial weight λ for the imitation loss term."""

    imitation_annealing_steps: int = 10_000_000
    """Global steps over which λ is linearly annealed from imitation_coef to 0."""


@dataclass(frozen=True)
class RmaPPOConfig(PpoConfig):
    """PPO with RMA privileged + adaptation encoder pair.

    Extends PpoConfig with the inference_phase flag, which controls which encoder
    is used when the policy is run in play/eval mode (not during training).
    """

    name: str = "RmaPPO"
    target: str = "colosseum.algorithm.rma_ppo:RmaPPO"

    inference_phase: int = 1
    """Encoder to use at inference time.
    1 = privileged encoder (ground-truth obs, default).
    2 = adaptation encoder (sensor obs, for testing Phase 2 quality).
    """
