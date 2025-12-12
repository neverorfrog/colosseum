"""Observation specification for velocity tracking task.

This module defines the CONTRACT between training and deployment.
It can be manually defined or derived from training ObservationGroupCfg.
"""

from typing import List

from colosseum.deploy.core.observation_spec import ObservationSpec


class VelocityObservationSpec(ObservationSpec):
    """Observation specification for velocity tracking task.

    This spec matches the "policy" observation group in velocity training configs.
    """

    @property
    def observation_names(self) -> List[str]:
        """Get ordered list of observation component names.

        This matches the order in ObservationGroupCfg.terms from training.
        """
        return [
            "velocity_commands",   # (3,) - [vx, vy, vyaw]
            "base_ang_vel",        # (3,) - [wx, wy, wz] in base frame
            "projected_gravity",   # (3,) - gravity in base frame
            "joint_pos_rel",       # (num_joints,) - relative to default
            "joint_vel",           # (num_joints,) - joint velocities
            "last_action",         # (num_joints,) - previous action
        ]

    def get_component_size(self, component_name: str, num_joints: int) -> int:
        """Get size of a specific observation component.

        Args:
            component_name: Name of the observation component.
            num_joints: Number of actuated joints.

        Returns:
            Size (number of dimensions) of this component.
        """
        # Fixed-size components
        if component_name in ["velocity_commands", "base_ang_vel", "projected_gravity"]:
            return 3

        # Joint-dependent components
        if component_name in ["joint_pos_rel", "joint_vel", "last_action"]:
            return num_joints

        raise ValueError(f"Unknown observation component: {component_name}")

    @classmethod
    def from_observation_group_cfg(
        cls,
        observation_group_cfg,
        num_joints: int,
    ) -> "VelocityObservationSpec":
        """Create VelocityObservationSpec from mjlab ObservationGroupCfg.

        This validates that the training config matches the expected structure.

        Args:
            observation_group_cfg: mjlab ObservationGroupCfg from training config.
            num_joints: Number of joints.

        Returns:
            VelocityObservationSpec instance.

        Raises:
            ValueError: If training config doesn't match expected structure.

        Example:
            >>> from colosseum.tasks.velocity.config.t1.env_cfgs import booster_t1_flat_env_cfg
            >>> cfg = booster_t1_flat_env_cfg()
            >>> spec = VelocityObservationSpec.from_observation_group_cfg(
            ...     cfg.observations["policy"],
            ...     num_joints=23
            ... )
            >>> print(spec.describe(num_joints=23))
        """
        spec = cls()

        # Validate that training config has expected components
        training_obs_names = list(observation_group_cfg.terms.keys())
        expected_obs_names = spec.observation_names

        # Check if all expected observations are present
        missing = set(expected_obs_names) - set(training_obs_names)
        if missing:
            raise ValueError(
                f"Training config missing observations: {missing}\n"
                f"Expected: {expected_obs_names}\n"
                f"Got: {training_obs_names}"
            )

        # Check if there are extra observations
        extra = set(training_obs_names) - set(expected_obs_names)
        if extra:
            raise ValueError(
                f"Training config has extra observations: {extra}\n"
                f"Expected: {expected_obs_names}\n"
                f"Got: {training_obs_names}"
            )

        # Validate concatenation settings
        if not observation_group_cfg.concatenate_terms:
            raise ValueError(
                "VelocityObservationSpec requires concatenate_terms=True in training"
            )

        print(f"✓ Training config validated against {spec.__class__.__name__}")
        print(f"  Total observation size: {spec.compute_total_size(num_joints)}")
        print(f"  Components: {', '.join(spec.observation_names)}")

        return spec


# Singleton instance
VELOCITY_OBS_SPEC = VelocityObservationSpec()


__all__ = [
    "VelocityObservationSpec",
    "VELOCITY_OBS_SPEC",
]
