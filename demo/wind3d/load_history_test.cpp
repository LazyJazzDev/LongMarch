#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

#include "simulation.h"

int main() {
  auto options = wind3d::OptimizedSolverOptions();
  options.display = false;
  options.record_loads = true;
  options.stop_after = 64;
  auto simulation = wind3d::Simulation::Create(0, 80, options);
  if (!simulation)
    return 1;
  auto wait = [&](uint64_t generation, uint64_t step) {
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (std::chrono::steady_clock::now() < end) {
      if (!simulation->Error().empty())
        return false;
      // Deliberately do not consume snapshots as steps run.
      if (simulation->Paused()) {
        auto state = simulation->Latest();
        if (state && state->generation == generation && state->step == step)
          return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    return false;
  };
  if (!wait(0, 64))
    return 2;
  auto history = simulation->RecentLoads();
  if (history.samples.size() != 64 || history.generation != 0)
    return 3;
  double impulse = 0;
  for (size_t i = 0; i < history.samples.size(); ++i) {
    const auto &sample = history.samples[i];
    if (sample.step != i + 1 || sample.dt <= 0 || (i && sample.time <= history.samples[i - 1].time))
      return 4;
    impulse += sample.force.x * double(sample.dt);
  }
  auto state = simulation->Latest();
  if (std::abs(impulse - state->impulse[0]) > 1e-4 * std::max(1.0, std::abs(impulse)))
    return 5;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  if (simulation->RecentLoads().samples.size() != 64)
    return 6;
  auto controls = state->controls;
  controls.elevator = 5;
  simulation->Configure(controls, true);
  simulation->Reset();
  if (!wait(1, 1))
    return 7;
  auto reset = simulation->RecentLoads();
  if (reset.generation != 1 || reset.samples.size() != 1 || reset.samples[0].controls.elevator != 5)
    return 8;
  simulation->Configure(controls, false);
  if (!wait(1, 64))
    return 9;
  auto resumed = simulation->RecentLoads();
  if (resumed.samples.size() != 64 || history.samples.size() != 64)
    return 10;
  for (size_t i = 0; i < resumed.samples.size(); ++i)
    if (resumed.samples[i].step != i + 1)
      return 11;
  std::cout << "PASS all solver steps retained without polling snapshots; impulse, pause, reset and resume verified\n";
}
