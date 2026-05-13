from mjlab.managers.curriculum_manager import CurriculumTermCfg

from colosseum.mdp.curriculums import (
    command_vel_curriculum,
    penalty_curriculum,
    push_curriculum_by_transitions,
    standing_curriculum_by_transitions,
)

curriculum = {
    "command_vel": CurriculumTermCfg(
        func=command_vel_curriculum,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {
                    "transitions": 0,
                    "lin_vel_x": (-0.5, 0.5),
                    "lin_vel_y": (-0.4, 0.4),
                    "ang_vel_z": (-0.5, 0.5),
                },
                {
                    "transitions": 200_000_000,
                    "lin_vel_x": (-0.7, 0.7),
                    "lin_vel_y": (-0.6, 0.6),
                    "ang_vel_z": (-0.75, 0.75),
                },
                {
                    "transitions": 400_000_000,
                    "lin_vel_x": (-1.0, 1.0),
                    "lin_vel_y": (-0.8, 0.8),
                    "ang_vel_z": (-1.0, 1.0),
                },
            ],
        },
    ),
    "standing_curriculum": CurriculumTermCfg(
        func=standing_curriculum_by_transitions,
        params={
            "command_name": "twist",
            "stages": [
                {"transitions": 0, "rel_standing_envs": 0.4},
                {"transitions": 200_000_000, "rel_standing_envs": 0.3},
                {"transitions": 300_000_000, "rel_standing_envs": 0.2},
            ],
        },
    ),
    "push_curriculum": CurriculumTermCfg(
        func=push_curriculum_by_transitions,
        params={
            "event_name": "push_robot",
            "stages": [
                {
                    "transitions": 0,
                    "velocity_range": {
                        "x": (0.0, 0.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.0),
                    },
                },
                {
                    "transitions": 200_000_000,
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.2, 0.2),
                        "roll": (-0.2, 0.2),
                        "pitch": (-0.2, 0.2),
                        "yaw": (-0.3, 0.3),
                    },
                },
                {
                    "transitions": 500_000_000,
                    "velocity_range": {
                        "x": (-1.0, 1.0),
                        "y": (-1.0, 1.0),
                        "z": (-0.4, 0.4),
                        "roll": (-0.52, 0.52),
                        "pitch": (-0.52, 0.52),
                        "yaw": (-0.78, 0.78),
                    },
                },
            ],
        },
    ),
    "penalty_curriculum": CurriculumTermCfg(
        func=penalty_curriculum,
        params={
            "reward_names": [
                "penalty_body_ang_vel",
                "penalty_orientation",
                "penalty_action_rate",
                "penalty_feet_distance",
                "penalty_feet_ori",
                "penalty_pose_deviation",
                "penalty_feet_yaw_diff",
                "penalty_feet_yaw_mean",
                "arm_swing",
            ],
            "initial_scale": 0.1,
            "min_scale": 0.01,
            "max_scale": 1.0,
            "level_down_threshold": 150,
            "level_up_threshold": 750,
            "degree": 0.00025,
        },
    ),
}
