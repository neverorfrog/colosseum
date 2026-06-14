"""Curriculum terms.

booster_gym uses ``only_positive_rewards`` (not a penalty curriculum), and applies
disturbances at fixed magnitude, so there is no curriculum term here. The command
range is handled per-env by the grid-based CurriculumVelocityCommand (cat_cfg.py).
"""

curriculum: dict = {}
