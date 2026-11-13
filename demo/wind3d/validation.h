#pragma once

#include <chrono>
#include <iostream>
#include <thread>

#include "simulation.h"

namespace wind3d {
// Exercise ownership and step-boundary control changes on the actual worker,
// including a reset while paused. No additional simulation/backend is mocked.
inline bool ValidateLifecycle(Simulation &simulation, Controls controls) {
  auto wait_for = [&](auto predicate) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (std::chrono::steady_clock::now() < deadline) {
      if (!simulation.Error().empty())
        return false;
      if (predicate())
        return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
  };
  simulation.Configure(controls, true);
  if (!wait_for([&] { return simulation.Paused(); }))
    return false;
  auto old = simulation.Latest();
  if (!old)
    return false;
  auto checksum = [](const Snapshot &snapshot) {
    double sum = 0;
    for (const auto &cell : snapshot.cells)
      sum += cell.u + 3.0 * cell.v + 5.0 * cell.w + 7.0 * cell.pressure;
    return sum;
  };
  double old_checksum = checksum(*old);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  if (simulation.Latest() != old)
    return false;
  controls.speed = controls.speed > 15 ? 8 : 20;
  controls.angle = -8;
  controls.flap = 25;
  controls.aileron = 15;
  controls.elevator = -15;
  controls.rudder = 20;
  simulation.Configure(controls, true);
  simulation.Reset();
  if (!wait_for([&] {
        auto state = simulation.Latest();
        return state && state->generation == old->generation + 1 && simulation.Paused();
      }))
    return false;
  auto reset = simulation.Latest();
  if (reset->step != 1 || reset->controls.speed != controls.speed || reset->controls.angle != controls.angle ||
      reset->controls.flap != controls.flap || reset->controls.aileron != controls.aileron ||
      reset->controls.elevator != controls.elevator || reset->controls.rudder != controls.rudder ||
      reset->aircraft == old->aircraft || checksum(*old) != old_checksum)
    return false;
  simulation.Configure(controls, false);
  if (!wait_for([&] { return simulation.Latest()->step > reset->step; }))
    return false;
  std::cout << "lifecycle_validation=PASS (pause, reset, controls, resume, immutable snapshot)\n";
  return true;
}
}  // namespace wind3d
