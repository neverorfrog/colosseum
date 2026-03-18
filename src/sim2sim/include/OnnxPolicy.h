#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

class OnnxPolicy {
public:
    explicit OnnxPolicy(const std::string& model_path);

    // Forward pass. obs must have exactly obs_dim() elements.
    // Returns action_dim() joint position targets in simulation order.
    Eigen::VectorXf infer(const Eigen::VectorXf& input);

    int input_dim()    const { return input_dim_; }
    int output_dim() const { return output_dim_; }

private:
    Ort::Env            env_;
    Ort::SessionOptions session_opts_;
    Ort::Session        session_;
    Ort::MemoryInfo     mem_info_;

    // Cached I/O names (ORT requires const char* that outlives Run())
    std::string input_name_;
    std::string output_name_;

    int input_dim_;
    int output_dim_;
};
