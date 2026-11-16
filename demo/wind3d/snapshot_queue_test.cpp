#include <chrono>
#include <iostream>
#include <thread>

#include "simulation.h"

int main() {
  using namespace wind3d;
  auto options = OptimizedSolverOptions();
  options.display = false;
  options.export_field = false;
  options.retain_snapshots = true;
  options.stop_after = 8;
  options.max_dt = .0002f;
  auto simulation = Simulation::Create(0, 64, options);
  if (!simulation)
    return 1;
  auto wait = [&](auto predicate) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
      if (!simulation->Error().empty())
        return false;
      if (predicate())
        return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
  };
  if (!wait([&] {
        auto s = simulation->Latest();
        return s && s->step == 2;
      }))
    return 2;
  auto second = simulation->Latest();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  if (simulation->Latest() != second)
    return 3;
  for (uint64_t step = 1; step <= 8; ++step) {
    std::shared_ptr<const Snapshot> sample;
    if (!wait([&] {
          sample = simulation->TakeNext();
          return bool(sample);
        }) ||
        sample->step != step)
      return 4;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (!wait([&] { return simulation->Paused(); }) || simulation->TakeNext())
    return 5;
  auto old = simulation->Latest();
  simulation->Reset();
  if (!wait([&] { return simulation->Latest()->generation != old->generation && simulation->Paused(); }))
    return 6;
  auto reset = simulation->TakeNext();
  if (!reset || reset->step != 1 || reset->generation != old->generation + 1 || simulation->TakeNext())
    return 7;
  simulation->Configure(reset->controls, false);
  if (!wait([&] { return simulation->Latest()->step == 3; }))
    return 8;
  // Destroy while the producer is blocked by a full queue.
  simulation.reset();
  if (second->step != 2 || old->step != 8)
    return 9;
  std::cout << "PASS lossless bounded snapshots, slow consumer, reset and blocked-worker destruction\n";
}
