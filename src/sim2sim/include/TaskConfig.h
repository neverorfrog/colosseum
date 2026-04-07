#pragma once
#include <string>
#include <array>
#include <booster/robot/b1/b1_api_const.hpp>

using booster::robot::b1::JointIndex;

struct TaskConfig {
    static constexpr int NUM_JOINTS = 23;

    std::string task_name = "t1-velocity-flat";
    std::string model_path;   // absolute or relative to executable

    float policy_dt = 0.02f;  // 50 Hz

    std::array<JointIndex, NUM_JOINTS> joints;

    // Per-joint action scale = policy_action_scale * effort_limit / stiffness
    // (matches Python Policy.__init__: self.action_scale = cfg.action_scale
    //                                                     * robot.effort_limit
    //                                                     / robot.joint_stiffness)
    // See "Action Decoding" section for precomputed values.
    std::array<float, NUM_JOINTS> action_scale;

    // Default standing pose (rad) — subtracted in joint_pos_rel observation,
    // added back after action decoding. Source: deploy.py default_joint_pos
    std::array<float, NUM_JOINTS> default_joint_pos;

    // PD gains sent in every LowCmd. Source: deploy.py joint_stiffness/damping
    std::array<float, NUM_JOINTS> kp;
    std::array<float, NUM_JOINTS> kd;
};
