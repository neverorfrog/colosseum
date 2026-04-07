#include "Policy.h"
#include "TaskRegistry.h"
#include <filesystem>

class T1VelocityFlat : public Policy {
    public:
        T1VelocityFlat() : Policy(make_config()) {}

    protected:

        void build_observation(const RobotState& state) {
            observation.clear();

            // 1. Base angular velocity (3)
            observation.push_back(state.gyro[0]);
            observation.push_back(state.gyro[1]);
            observation.push_back(state.gyro[2]);

            // 2. Projected gravity (3)
            observation.push_back(state.projected_gravity[0]);
            observation.push_back(state.projected_gravity[1]);
            observation.push_back(state.projected_gravity[2]);

            // 3. Joint positions relative to default (23)
            for (int i = 0; i < TaskConfig::NUM_JOINTS; i++) {
                observation.push_back(state.joint_pos[i] - config_.default_joint_pos[i]);
            }

            // 4. Joint velocities (23)
            for (int i = 0; i < TaskConfig::NUM_JOINTS; i++) {
                observation.push_back(state.joint_vel[i]);
            }

            // 5. Last action (23)
            for (int i = 0; i < TaskConfig::NUM_JOINTS; i++) {
                observation.push_back(last_action[i]);
            }

            // 6. Velocity command (3)
            observation.push_back(state.v[0]);
            observation.push_back(state.v[1]);
            observation.push_back(state.v[2]);
        }

        static std::string resolve_model_path(const std::string& filename) {
            std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe");
            return (exe.parent_path().parent_path() / "models" / filename).string();
        }

        static TaskConfig make_config() {
            TaskConfig cfg;
            cfg.task_name = "t1-velocity-flat_ppo_499580928";
            cfg.model_path = resolve_model_path(cfg.task_name + ".onnx");

            cfg.policy_dt = 0.02f;

            cfg.joints = {
                JointIndex::kHeadYaw, JointIndex::kHeadPitch,
                JointIndex::kLeftShoulderPitch,  JointIndex::kLeftShoulderRoll,
                JointIndex::kLeftElbowPitch,    JointIndex::kLeftElbowYaw,
                JointIndex::kRightShoulderPitch, JointIndex::kRightShoulderRoll,
                JointIndex::kRightElbowPitch,   JointIndex::kRightElbowYaw,
                JointIndex::kWaist,
                JointIndex::kLeftHipPitch, JointIndex::kLeftHipRoll, JointIndex::kLeftHipYaw,
                JointIndex::kLeftKneePitch, JointIndex::kCrankUpLeft, JointIndex::kCrankDownLeft,
                JointIndex::kRightHipPitch, JointIndex::kRightHipRoll, JointIndex::kRightHipYaw,
                JointIndex::kRightKneePitch, JointIndex::kCrankUpRight, JointIndex::kCrankDownRight
            };

            cfg.default_joint_pos = {
                0.0f, 0.0f,                             // AAHead_yaw, Head_pitch
                0.2f, -1.3f, 0.0f, -0.5f,               // Left arm
                0.2f,  1.3f, 0.0f,  0.5f,               // Right arm
                0.0f,                                   // Waist
                -0.2f, 0.0f, 0.0f, 0.4f, -0.2f, 0.0f,   // Left leg
                -0.2f, 0.0f, 0.0f, 0.4f, -0.2f, 0.0f,   // Right leg
            };

            // Uniform action scale = 0.25 for all joints
            cfg.action_scale.fill(0.25f);

            cfg.kp = {
                5.0f, 5.0f,                                         // Head
                20.0f, 20.0f, 20.0f, 20.0f,                         // Left arm
                20.0f, 20.0f, 20.0f, 20.0f,                         // Right arm
                200.0f,                                             // Waist
                200.0f, 200.0f, 200.0f, 200.0f, 50.0f, 50.0f,       // Left leg
                200.0f, 200.0f, 200.0f, 200.0f, 50.0f, 50.0f,       // Right leg
            };

            cfg.kd = {
                0.5f, 0.5f,
                0.5f, 0.5f, 0.5f, 0.5f,
                0.5f, 0.5f, 0.5f, 0.5f,
                5.0f,
                5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 3.0f,
                5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 3.0f,
            };

            return cfg;
        }
};

REGISTER_TASK("t1-velocity-flat", T1VelocityFlat);
