#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

#include "simulation.h"

namespace wind3d {
// Cross-check the actual explicit mesh against the independent occupancy path,
// including combined extreme deflections.
inline bool ValidateGeometry() {
  const Controls poses[] = {{12, 5, 0, 0, 0, 0}, {12, -20, 40, 25, -25, 25}, {12, 25, 0, -25, 25, -25}};
  float maximum = 0;
  for (const auto &controls : poses) {
    auto mesh = AircraftMesh(controls);
    if (mesh.empty() || mesh.size() % 3)
      return false;
    for (size_t i = 0; i < mesh.size(); ++i) {
      const auto &vertex = mesh[i];
      if (!std::isfinite(Length(vertex.position) + Length(vertex.normal)) ||
          std::abs(Length(vertex.normal) - 1) > 0.001f) {
        std::cerr << "Invalid mesh vertex " << i << ", normal length=" << Length(vertex.normal) << '\n';
        return false;
      }
      if (i % 31 == 0) {
        maximum = std::max(maximum, AircraftDistance(vertex.position, controls));
        if (maximum > 0.0025f) {
          std::cerr << "Geometry / solid boundary mismatch: " << maximum << " m\n";
          return false;
        }
      }
    }
  }
  std::cout << "geometry_validation=PASS max_surface_offset=" << maximum << " m\n";
  return true;
}

// Exercise ownership and step-boundary control changes on the actual worker,
// including a reset while paused. No additional simulation/backend is mocked.
inline bool ValidateLifecycle(Simulation &simulation, Controls controls) {
  auto wait_for = [&](auto predicate) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(3);
    while (std::chrono::steady_clock::now() < deadline) {
      if (!simulation.Error().empty()) {
        std::cerr << "Lifecycle solver error: " << simulation.Error() << '\n';
        return false;
      }
      if (predicate())
        return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    auto state = simulation.Latest();
    std::cerr << "Lifecycle wait timed out: step=" << (state ? state->step : 0)
              << " generation=" << (state ? state->generation : 0) << " paused=" << simulation.Paused() << '\n';
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
    for (float value : snapshot.vorticity)
      sum += value;
    return sum;
  };
  double old_checksum = checksum(*old);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  if (simulation.Latest() != old)
    return false;
  controls.speed = controls.speed > 15 ? 8 : 20;
  controls.angle = controls.angle == -8 ? 8 : -8;
  controls.flap = 25;
  controls.wing_controls = {false, 15, -2, 3, 30, 10, false, 20, false};
  controls.aileron = 15;
  controls.elevator = -15;
  controls.rudder = 20;
  controls.airframe.tail_incidence = controls.airframe.tail_incidence == 3 ? -3 : 3;
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
      !SameAirframe(reset->controls.airframe, controls.airframe) ||
      !SameWingControls(reset->controls.wing_controls, controls.wing_controls) || reset->aircraft == old->aircraft ||
      checksum(*old) != old_checksum) {
    std::cerr << "Lifecycle reset/snapshot check failed: step=" << reset->step << " speed=" << reset->controls.speed
              << " angle=" << reset->controls.angle << '\n';
    return false;
  }
  simulation.Configure(controls, false);
  if (!wait_for([&] { return simulation.Latest()->step > reset->step; }))
    return false;
  // Exercise a changed inlet without a geometry rebuild or resetting maps.
  controls.speed = 10;
  auto before_change = simulation.Latest();
  simulation.Configure(controls, false);
  if (!wait_for([&] {
        auto state = simulation.Latest();
        return state->controls.speed == controls.speed && state->step >= before_change->step + 3;
      }))
    return false;
  auto changed = simulation.Latest();
  if (changed->generation != reset->generation || !changed->pressure_converged ||
      !std::isfinite(Length(changed->force) + Length(changed->moment)) ||
      changed->divergence_after >= std::max(1e-5f, changed->divergence_before * 0.001f)) {
    std::cerr << "Live inlet validation failed: generation=" << changed->generation << " expected=" << reset->generation
              << " pressure=" << changed->pressure_residual << " divergence=" << changed->divergence_before << " -> "
              << changed->divergence_after << " force=" << Length(changed->force)
              << " moment=" << Length(changed->moment) << '\n';
    return false;
  }
  std::cout << "lifecycle_validation=PASS (pause, reset, controls, resume, live inlet change, immutable snapshot)\n";
  return true;
}
}  // namespace wind3d
