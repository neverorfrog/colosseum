#include "Policy.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <string>

// Identity model lives next to the test sources so PROJECT_ROOT always works.
static std::string model_path() {
    return std::string(PROJECT_ROOT) + "/models/identity.onnx";
}

TEST(PolicyTest, LoadsModel) {
    Policy policy(model_path());
    EXPECT_EQ(policy.obs_dim(),    82);
    EXPECT_EQ(policy.action_dim(), 82);
}

TEST(PolicyTest, IdentityForwardPass) {
    Policy policy(model_path());

    Eigen::VectorXf obs = Eigen::VectorXf::LinSpaced(policy.obs_dim(), 0.f, 1.f);
    Eigen::VectorXf out = policy.infer(obs);

    ASSERT_EQ(out.size(), obs.size());
    float max_err = (out - obs).cwiseAbs().maxCoeff();
    EXPECT_LT(max_err, 1e-6f) << "Identity model output differs from input";
}

TEST(PolicyTest, WrongObsDimThrows) {
    Policy policy(model_path());
    Eigen::VectorXf bad_obs(policy.obs_dim() + 1);
    EXPECT_THROW(policy.infer(bad_obs), std::runtime_error);
}
