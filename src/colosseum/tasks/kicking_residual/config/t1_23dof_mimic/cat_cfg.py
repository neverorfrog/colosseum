"""Commands, actions, and terminations for t1-kicking-residual-mimic.

``actions``/``terminations`` are re-exported unchanged from the
``t1`` sibling. ``commands`` extends the base ``{"twist", "ball_angle",
"gait_phase"}`` with a ``"motion"`` entry: a ``GatedHoldMotionCommand`` that
stays on frame 0 (no tracking signal) until the robot is within
``trigger_distance`` of the ball — the same 0.25 m threshold at which
``BallTwistCommand`` zeroes the walk command — then plays the reference kick
clip once and holds its final frame.
"""

from typing import Dict

from mjlab.managers import CommandTermCfg

from colosseum.mdp.motion_command import GatedHoldMotionCommandCfg
from colosseum.utils import project_root

from ..t1.cat_cfg import actions, terminations
from ..t1.cat_cfg import commands as _base_commands

# Bodies tracked by the motion-mimic reward/observation terms. Must all be
# present in `t1_motion.npz`'s body_names. Copied locally (matches
# kicking_5_mimic's own pattern of not sharing this tuple cross-task).
_TRACKED_BODY_NAMES = (
  "Trunk",
  "Waist",
  "H1",
  "H2",
  "Hip_Roll_Left",
  "Shank_Left",
  "left_foot_link",
  "Hip_Roll_Right",
  "Shank_Right",
  "right_foot_link",
)

commands: Dict[str, CommandTermCfg] = {
  **_base_commands,
  "motion": GatedHoldMotionCommandCfg(
    entity_name="robot",
    ball_entity="ball",
    motion_file=str(project_root() / "models" / "trajectories" / "kick_1.npz"),
    anchor_body_name="Trunk",
    body_names=_TRACKED_BODY_NAMES,
    trigger_distance=0.20,
    pose_range={},
    velocity_range={},
    resampling_time_range=(1e6, 1e6),
    sampling_mode="start",
  ),
}
