#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

// Runs an ONNX policy: flat observation vector → joint position targets.
//
// Usage:
//   Policy policy("path/to/policy.onnx");
//   Eigen::VectorXf obs = ...;          // shape (obs_dim,)
//   Eigen::VectorXf targets = policy.infer(obs);  // shape (action_dim,)
class Policy {
public:
    explicit Policy(const std::string& model_path);

    // Forward pass. obs must have exactly obs_dim() elements.
    // Returns action_dim() joint position targets in simulation order.
    Eigen::VectorXf infer(const Eigen::VectorXf& obs);

    int obs_dim()    const { return obs_dim_; }
    int action_dim() const { return action_dim_; }

private:
    Ort::Env            env_;
    Ort::SessionOptions session_opts_;
    Ort::Session        session_;
    Ort::MemoryInfo     mem_info_;

    // Cached I/O names (ORT requires const char* that outlives Run())
    std::string input_name_;
    std::string output_name_;

    int obs_dim_;
    int action_dim_;
};
