#pragma once
#include "TaskConfig.h"
#include "RobotState.h"
#include "OnnxPolicy.h"

class Policy {
public:
    explicit Policy(TaskConfig cfg): config_(std::move(cfg)), onnx(config_.model_path) {
        observation.reserve(onnx.input_dim());
    }

    virtual ~Policy() = default;

    void reset() {
        std::fill(std::begin(last_action), std::end(last_action), 0.0f);
    }

    std::array<float, TaskConfig::NUM_JOINTS> get_action(const RobotState& state) {
        build_observation(state);
        Eigen::VectorXf obs_vec = Eigen::Map<Eigen::VectorXf>(observation.data(), observation.size());
        Eigen::VectorXf action_vec = onnx.infer(obs_vec);
        std::array<float, TaskConfig::NUM_JOINTS> action{};
        for (int i = 0; i < onnx.output_dim(); i++) {
            // Match Python: store raw network output as last_action (used in obs)
            last_action[i] = action_vec[i];
            // Decode: scale + default pose (matches Python inference())
            action[i] = action_vec[i] * config_.action_scale[i] + config_.default_joint_pos[i];
        }
        return action;
    }

    const TaskConfig& config() const { return config_; }


protected:
    TaskConfig config_;
    float last_action[TaskConfig::NUM_JOINTS]{};
    std::vector<float> observation;
    OnnxPolicy onnx;

    // Subclass fills obs[0..OBS_DIM-1].
    virtual void build_observation(const RobotState& state) = 0;

};
