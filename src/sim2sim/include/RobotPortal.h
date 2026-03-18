#pragma once
#include "RobotState.h"
#include "booster/robot/b1/b1_loco_client.hpp"
#include "booster/robot/channel/channel_subscriber.hpp"
#include "booster/robot/channel/channel_publisher.hpp"
#include "booster/idl/b1/LowState.h"
#include "booster/idl/b1/LowCmd.h"
#include "booster/robot/b1/b1_api_const.hpp"

#include <array>
#include <atomic>
#include <booster/robot/b1/b1_loco_api.hpp>
#include <booster/robot/common/robot_shared.hpp>
#include <thread>

class RobotPortal {

    public:
        RobotPortal();
        ~RobotPortal();

        static RobotPortal& instance() {
            static RobotPortal instance;
            return instance;
        }

        void initialize();

        // Delete copy constructor and assignment operator
        RobotPortal(const RobotPortal&) = delete;
        RobotPortal& operator=(const RobotPortal&) = delete;

        const RobotState& getState() const {
            return state;
        }

        const int changeMode(booster::robot::RobotMode mode) {
            return client.ChangeMode(mode);
        }

    private:
        // State
        RobotState state;

        // LocoClient for synchronous RPC calls
        booster::robot::b1::B1LocoClient client;

        // DDS Channels
        ChannelSubscriber<booster_interface::msg::LowState> low_state_sub;
        void lowStateCallback(const booster_interface::msg::LowState& msg);
        ChannelPublisherPtr<booster_interface::msg::LowCmd> low_cmd_pub;

        // Joystick
        std::thread joystick_thread;
        std::atomic<bool> joystick_stop{false};
        void joystickLoop(const std::string& device);

        // Utility functions
        std::array<float, 4> rpy_to_quat(const float rpy[3]) const;
        std::array<float, 3> compute_projected_gravity(const float q[4]) const;
};
