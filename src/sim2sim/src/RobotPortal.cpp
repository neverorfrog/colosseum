#include "RobotPortal.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/joystick.h>
#include <poll.h>
#include <unistd.h>
#include <iostream>

using booster::robot::ChannelPublisher;

RobotPortal::RobotPortal()
    : state(),
      low_state_sub(booster::robot::b1::kTopicLowState) {}

void RobotPortal::initialize() {
    client.Init();
    low_state_sub.InitChannel([this](const void* msg) {
        lowStateCallback(*static_cast<const booster_interface::msg::LowState*>(msg));
    });
    low_cmd_pub = std::make_shared<ChannelPublisher<booster_interface::msg::LowCmd>>(booster::robot::b1::kTopicJointCtrl);
    low_cmd_pub->InitChannel();
    const char* js_device = std::getenv("JOYSTICK_DEVICE");
    if (!js_device) js_device = "/dev/input/js2";
    joystick_thread = std::thread(&RobotPortal::joystickLoop, this, std::string(js_device));
}

RobotPortal::~RobotPortal() {
    joystick_stop = true;
    if (joystick_thread.joinable())
        joystick_thread.join();
}

void RobotPortal::lowStateCallback(const booster_interface::msg::LowState& msg) {
    for (int i = 0; i < 3; i++) {
        state.rpy[i] = msg.imu_state().rpy()[i];
        state.gyro[i] = msg.imu_state().gyro()[i];
        state.acc[i] = msg.imu_state().acc()[i];
    }

    for (int i = 0; i < 23; i++) {
        state.joint_pos[i] = msg.motor_state_serial()[i].q();
        state.joint_vel[i] = msg.motor_state_serial()[i].dq();
        state.feedback_torque[i] = msg.motor_state_serial()[i].tau_est();
    }

    auto quat = rpy_to_quat(state.rpy);
    for (int i = 0; i < 4; i++) {
        state.root_quat[i] = quat[i];
    }
    auto pg = compute_projected_gravity(quat.data());
    for (int i = 0; i < 3; i++) {
        state.projected_gravity[i] = pg[i];
    }
}

void RobotPortal::joystickLoop(const std::string& device) {
    static constexpr float VX_MAX   = 1.0f;
    static constexpr float VY_MAX   = 0.5f;
    static constexpr float VYAW_MAX = 1.0f;

    int fd = open(device.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0) {
        std::cerr << "[joystick] Cannot open " << device << ": " << strerror(errno) << "\n";
        return;
    }
    std::cout << "[joystick] Opened " << device << "\n";

    static constexpr int kMaxAxes = 8;
    std::array<int16_t, kMaxAxes> axes{};

    auto norm = [](int16_t v) { return static_cast<float>(v) / 32767.0f; };
    auto deadzone = [](float v) { return fabsf(v) < 0.1f ? 0.0f : v; };

    pollfd pfd{fd, POLLIN, 0};
    js_event event;

    while (!joystick_stop) {
        if (poll(&pfd, 1, 50) <= 0) continue;

        if (read(fd, &event, sizeof(event)) != sizeof(event)) continue;

        if ((event.type & ~JS_EVENT_INIT) != JS_EVENT_AXIS) continue;
        if (event.number >= kMaxAxes) continue;

        axes[event.number] = event.value;

        state.v[0] =  deadzone(norm(axes[1])) * VX_MAX;
        state.v[1] = -deadzone(norm(axes[0])) * VY_MAX;
        state.v[2] = -deadzone(norm(axes[3])) * VYAW_MAX;
    }

    close(fd);
    std::cout << "[joystick] Thread stopped.\n";
}

std::array<float, 4> RobotPortal::rpy_to_quat(const float rpy[3]) const {
    float cr = cosf(rpy[0]*0.5f), sr = sinf(rpy[0]*0.5f);
    float cp = cosf(rpy[1]*0.5f), sp = sinf(rpy[1]*0.5f);
    float cy = cosf(rpy[2]*0.5f), sy = sinf(rpy[2]*0.5f);
    std::array<float, 4> q;
    q[0] = cr*cp*cy + sr*sp*sy;  // w
    q[1] = sr*cp*cy - cr*sp*sy;  // x
    q[2] = cr*sp*cy + sr*cp*sy;  // y
    q[3] = cr*cp*sy - sr*sp*cy;  // z
    return q;
}

std::array<float, 3> RobotPortal::compute_projected_gravity(const float q[4]) const {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    std::array<float, 3> pg;
    pg[0] =  2.0f * (w*y - x*z);
    pg[1] = -2.0f * (w*x + y*z);
    pg[2] = -(1.0f - 2.0f*(x*x + y*y));
    return pg;
}
