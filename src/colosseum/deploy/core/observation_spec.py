"""Base observation specification for task deployment.

This module provides base classes for defining observation contracts between
training (mjlab ObservationGroupCfg) and deployment (policy observation computation).
"""

from abc import ABC, abstractmethod
from typing import List
import torch


class ObservationSpec(ABC):
    """Abstract base class for observation specifications.

    An ObservationSpec defines the CONTRACT between training and deployment:
    - What observations are included
    - In what order they appear
    - How to validate them
    - How to compute their size

    Subclasses should implement task-specific observation structure.
    """

    @property
    @abstractmethod
    def observation_names(self) -> List[str]:
        """Get ordered list of observation component names.

        This should match the keys in ObservationGroupCfg.terms (in order).

        Returns:
            List of observation names in concatenation order.

        Example:
            ["velocity_commands", "base_ang_vel", "projected_gravity", ...]
        """
        pass

    @abstractmethod
    def get_component_size(self, component_name: str, num_joints: int) -> int:
        """Get size of a specific observation component.

        Args:
            component_name: Name of the observation component.
            num_joints: Number of actuated joints.

        Returns:
            Size (number of dimensions) of this component.

        Example:
            get_component_size("base_ang_vel", 23) -> 3
            get_component_size("joint_pos_rel", 23) -> 23
        """
        pass

    def compute_size(self, num_joints: int) -> int:
        """Compute total observation size.

        Args:
            num_joints: Number of actuated joints.

        Returns:
            Total observation dimension.
        """
        return sum(
            self.get_component_size(name, num_joints)
            for name in self.observation_names
        )

    def get_all_component_sizes(self, num_joints: int) -> dict[str, int]:
        """Get sizes of all observation components.

        Args:
            num_joints: Number of actuated joints.

        Returns:
            Dictionary mapping component names to sizes.
        """
        return {
            name: self.get_component_size(name, num_joints)
            for name in self.observation_names
        }

    def validate_observation(
        self,
        obs: torch.Tensor,
        num_joints: int,
        check_finite: bool = True,
    ) -> None:
        """Validate observation tensor matches specification.

        Args:
            obs: Observation tensor to validate.
            num_joints: Expected number of joints.
            check_finite: Whether to check for NaN/Inf values.

        Raises:
            ValueError: If observation doesn't match spec.
        """
        expected_size = self.compute_size(num_joints)

        # Check size
        if obs.shape[-1] != expected_size:
            raise ValueError(
                f"Observation size mismatch!\n"
                f"Expected: {expected_size} (for {num_joints} joints)\n"
                f"Got: {obs.shape[-1]}\n"
                f"Component sizes: {self.get_all_component_sizes(num_joints)}"
            )

        # Check finite values
        if check_finite:
            if not torch.isfinite(obs).all():
                nan_count = torch.isnan(obs).sum().item()
                inf_count = torch.isinf(obs).sum().item()
                raise ValueError(
                    f"Observation contains non-finite values!\n"
                    f"NaN count: {nan_count}\n"
                    f"Inf count: {inf_count}"
                )

    def split_observation(
        self,
        obs: torch.Tensor,
        num_joints: int,
    ) -> dict[str, torch.Tensor]:
        """Split concatenated observation into components.

        Useful for debugging and visualization.

        Args:
            obs: Concatenated observation tensor.
            num_joints: Number of joints.

        Returns:
            Dictionary mapping component names to tensors.
        """
        self.validate_observation(obs, num_joints)

        components = {}
        sizes = self.get_all_component_sizes(num_joints)

        idx = 0
        for name in self.observation_names:
            size = sizes[name]
            components[name] = obs[..., idx : idx + size]
            idx += size

        return components

    def describe(self, num_joints: int) -> str:
        """Generate human-readable description of observation structure.

        Args:
            num_joints: Number of joints.

        Returns:
            Formatted description string.
        """
        sizes = self.get_all_component_sizes(num_joints)
        total = self.compute_size(num_joints)

        lines = [
            f"{self.__class__.__name__} Observation Specification",
            "=" * 60,
            f"Total size: {total}",
            "",
            "Components (in order):",
        ]

        idx = 0
        for name in self.observation_names:
            size = sizes[name]
            lines.append(f"  [{idx:3d}:{idx+size:3d}] {name:25s} ({size:2d} dims)")
            idx += size

        return "\n".join(lines)

    @classmethod
    def from_observation_group_cfg(
        cls,
        observation_group_cfg,
        num_joints: int,
    ) -> "ObservationSpec":
        """Create ObservationSpec from mjlab ObservationGroupCfg.

        This allows automatic generation of the spec from training config.

        Args:
            observation_group_cfg: mjlab ObservationGroupCfg instance.
            num_joints: Number of joints (needed for size computation).

        Returns:
            ObservationSpec instance matching the training config.

        Note:
            This is a class method that subclasses can override to provide
            custom extraction logic from ObservationGroupCfg.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_observation_group_cfg. "
            "Either implement this method or create the spec manually."
        )


__all__ = [
    "ObservationSpec",
]
