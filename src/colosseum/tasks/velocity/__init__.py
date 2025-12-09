"""Shared definitions for velocity tracking task.

This module provides constants and functions that work in both
training (mjlab) and deployment (booster_deploy) contexts.
"""

__all__ = [
    "ACTION_SCALE",
    "DEFAULT_JOINT_POS",
]

# ============================================================================
# Action Configuration
# ============================================================================

# Action scale factor (applied to normalized policy output)
# This should match the scale in your training config's JointPositionActionCfg
ACTION_SCALE = 0.25

# Default joint positions (standing pose)
# This is defined per-robot in robot configs, but can be referenced here
# for consistency between training and deployment
