#include "RobotPortal.h"
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

  std::this_thread::sleep_for(std::chrono::seconds(1));
  std::cout << "Initializing RobotPortal...\n" << std::flush;
  booster::robot::ChannelFactory::Instance()->Init(0);
  RobotPortal::instance().initialize();
  std::cout << "RobotPortal initialized.\n" << std::flush;

  if (RobotPortal::instance().changeMode(booster::robot::RobotMode::kCustom) != 0) {
    std::cerr << "Failed to change robot mode to Custom.\n";
    return -1;
  }
  std::cout << "Robot mode changed to WALK.\n" << std::flush;

  while (!terminated) {
    std::cout << RobotPortal::instance().getState();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  std::cout << "\nShutting down...\n" << std::flush;
  return 0;
}
