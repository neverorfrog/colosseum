"""Layer 1: Universal observation functions (robot-agnostic).

These pure functions work across all robots and tasks, depending only on
torch and math utilities. They can be used in both training and deployment.
"""

import torch

from colosseum.deploy.utils.isaaclab import math as lab_math


def compute_projected_gravity(
    root_quat_w: torch.Tensor,
    gravity_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project gravity vector into base frame (works for any robot).

    This tells the robot which direction is "down" in its own coordinate frame.

    Args:
        root_quat_w: Base orientation quaternion in world frame (w, x, y, z).
                    Shape: (4,) for single instance or (N, 4) for batched.
        gravity_w: Optional gravity vector in world frame. Defaults to [0, 0, -1].
                  Shape: (3,) or (N, 3).

    Returns:
        Gravity vector projected into base frame.
        Shape: (3,) for single instance or (N, 3) for batched.

    Examples:
        >>> # Single instance (deployment)
        >>> quat = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity (upright)
        >>> gravity = compute_projected_gravity(quat)
        >>> # gravity ≈ [0, 0, -1] (pointing down in base frame)

        >>> # Batched (training)
        >>> quats = torch.randn(4096, 4)  # 4096 environments
        >>> gravities = compute_projected_gravity(quats)
        >>> gravities.shape
        torch.Size([4096, 3])
    """
    if gravity_w is None:
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)

    # Handle both batched and unbatched
    if root_quat_w.dim() == 1:
        # Single instance (deployment)
        return lab_math.quat_apply_inverse(root_quat_w, gravity_w)
    else:
        # Batched (training)
        batch_size = root_quat_w.shape[0]
        if gravity_w.dim() == 1:
            gravity_w = gravity_w.unsqueeze(0).expand(batch_size, -1)
        return lab_math.quat_apply_inverse(root_quat_w, gravity_w)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vector by inverse of quaternion (re-export from isaaclab.math).

    Args:
        quat: Quaternion (w, x, y, z). Shape: (4,) or (N, 4).
        vec: Vector to rotate. Shape: (3,) or (N, 3).

    Returns:
        Rotated vector. Shape matches input.
    """
    return lab_math.quat_apply_inverse(quat, vec)


__all__ = [
    "compute_projected_gravity",
    "quat_apply_inverse",
]
