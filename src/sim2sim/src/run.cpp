#include "RobotPortal.h"
#include "TaskRegistry.h"
#include <booster/robot/channel/channel_factory.hpp>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

static volatile std::sig_atomic_t terminated = 0;

static void signal_handler(int) {
  terminated = 1;
}

extern "C" int run() {
  std::signal(SIGINT,  signal_handler);
  std::signal(SIGTERM, signal_handler);

  // 1. Initialize SDK and RobotPortal
  std::this_thread::sleep_for(std::chrono::seconds(5));
  std::cout << "Initializing RobotPortal...\n" << std::flush;
  booster::robot::ChannelFactory::Instance()->Init(0);
  RobotPortal::instance().initialize();
  std::cout << "RobotPortal initialized.\n" << std::flush;

  // 2. Wait for initial state
  while (!terminated && !RobotPortal::instance().hasState()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (terminated) return 0;

  // 3. Initialize policy
  auto policy = TaskRegistry::instance().create("t1-velocity-flat");
  auto cfg = policy->config();
  std::cout << "Policy '" << cfg.task_name << "' initialized with model: " << cfg.model_path << "\n" << std::flush;

  // 4. Change robot mode to Custom
  if (RobotPortal::instance().changeMode(booster::robot::RobotMode::kCustom) != 0) {
    std::cerr << "Failed to change robot mode to Custom.\n";
    return -1;
  }

  // 5. Policy loop
  policy->reset();
  std::array<float, 23> targets;
  auto next_tick = std::chrono::steady_clock::now();
  while (!terminated) {
      next_tick += std::chrono::microseconds(static_cast<long>(cfg.policy_dt * 1e6));
      std::this_thread::sleep_until(next_tick);
      targets = policy->get_action(RobotPortal::instance().getState());
      RobotPortal::instance().publishCommand(targets.data(), cfg.kp.data(), cfg.kd.data());
  }

  std::cout << "\nShutting down...\n" << std::flush;
  return 0;
}
