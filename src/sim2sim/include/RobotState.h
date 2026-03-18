#pragma once

#include <ostream>
#include <string>

struct RobotState {
    // From LowState.imu_state
    float rpy[3];            // roll, pitch, yaw (rad)
    float gyro[3];           // angular velocity in body frame (rad/s)
    float acc[3];            // linear acceleration (m/s^2)

    // From LowState.motor_state_serial
    float joint_pos[23];        // joint positions (rad)
    float joint_vel[23];       // joint velocities (rad/s)
    float feedback_torque[23]; // estimated torques (Nm)

    // Derived
    float projected_gravity[3]; // gravity vector in body frame
    float root_quat[4];         // quaternion from RPY (w,x,y,z)

    // Command
    float v[3];                // desired velocity in body frame [vx, vy, omega]

    friend std::ostream& operator<<(std::ostream& os, const RobotState& state) {
        os << "Velocity command: [vx: " << state.v[0] << ", vy: " << state.v[1] << ", omega: " << state.v[2] << "]\n";
        return os;
    }
};
