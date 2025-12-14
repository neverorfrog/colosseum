"""Layer 2: Booster T1 robot-specific observation functions.

These functions are specific to the T1 humanoid platform and can be used
across multiple tasks (velocity tracking, manipulation, etc.).
"""

import torch


def compute_foot_contact_state(
    contact_forces: torch.Tensor,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Compute binary foot contact state for T1 robot.

    Uses contact force threshold to determine if feet are in contact with ground.

    Args:
        contact_forces: Contact forces from sensor data.
                       Shape: (2,) or (N, 2) for [left_foot, right_foot].
        threshold: Force threshold in Newtons. Default 1.0 N.

    Returns:
        Binary contact state [left, right] where 1.0 = contact, 0.0 = no contact.
        Shape: (2,) or (N, 2).

    Examples:
        >>> # Single instance (deployment)
        >>> forces = torch.tensor([15.0, 0.5])  # Left foot grounded, right in air
        >>> contacts = compute_foot_contact_state(forces, threshold=1.0)
        >>> contacts
        tensor([1., 0.])

        >>> # Batched (training)
        >>> forces = torch.randn(4096, 2) * 10
        >>> contacts = compute_foot_contact_state(forces)
        >>> contacts.shape
        torch.Size([4096, 2])
    """
    return (contact_forces > threshold).float()


__all__ = [
    "compute_foot_contact_state",
]
