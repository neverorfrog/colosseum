#include "Policy.h"

#include <filesystem>
#include <stdexcept>
#include <string>

// Validates model path before it's passed to Ort::Session (which segfaults on bad paths).
static const std::string& checked_path(const std::string& path) {
    if (!std::filesystem::exists(path))
        throw std::runtime_error("Policy: model file not found: " + path);
    return path;
}

Policy::Policy(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "Policy"),
      session_opts_(),
      session_(env_, checked_path(model_path).c_str(), session_opts_),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    Ort::AllocatorWithDefaultOptions allocator;

    // Cache input name and shape
    auto input_name_ptr = session_.GetInputNameAllocated(0, allocator);
    input_name_ = input_name_ptr.get();

    auto input_type_info  = session_.GetInputTypeInfo(0);
    auto input_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
    // shape is (batch, obs_dim) — use last dim to be safe
    obs_dim_ = static_cast<int>(input_shape.back());

    // Cache output name and shape
    auto output_name_ptr = session_.GetOutputNameAllocated(0, allocator);
    output_name_ = output_name_ptr.get();

    auto output_type_info  = session_.GetOutputTypeInfo(0);
    auto output_shape = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
    action_dim_ = static_cast<int>(output_shape.back());
}

Eigen::VectorXf Policy::infer(const Eigen::VectorXf& obs)
{
    if (obs.size() != obs_dim_) {
        throw std::runtime_error(
            "Policy::infer: expected obs_dim=" + std::to_string(obs_dim_) +
            ", got " + std::to_string(obs.size()));
    }

    // Build input tensor (batch_size=1)
    std::array<int64_t, 2> input_shape{1, obs_dim_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_,
        const_cast<float*>(obs.data()),
        static_cast<size_t>(obs_dim_),
        input_shape.data(),
        input_shape.size()
    );

    // Run
    const char* input_names[]  = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,  &input_tensor,  1,
        output_names, 1
    );

    // Copy output to Eigen vector
    const float* output_data = output_tensors[0].GetTensorData<float>();
    return Eigen::Map<const Eigen::VectorXf>(output_data, action_dim_);
}
