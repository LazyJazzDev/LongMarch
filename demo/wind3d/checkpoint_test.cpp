#include <chrono>
#include <filesystem>
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
  auto path = std::filesystem::temp_directory_path() /
              ("wind3d-checkpoint-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  // A full snapshot queue must not deadlock a checkpoint request.
  if (simulation->Checkpoint(path.string()) != 2)
    return 3;
  auto drain = [&](uint64_t first) {
    for (uint64_t step = first; step <= 8; ++step) {
      std::shared_ptr<const Snapshot> sample;
      if (!wait([&] {
            sample = simulation->TakeNext();
            return bool(sample);
          }) ||
          sample->step != step)
        return false;
    }
    return wait([&] { return simulation->Paused(); });
  };
  if (!drain(1))
    return 4;
  auto reference = simulation->Latest();
  simulation->Restore(path.string());
  simulation->Reset();
  if (!wait([&] { return simulation->Latest()->generation != reference->generation && simulation->Paused(); }))
    return 5;
  simulation->Configure(reference->controls, false);
  if (!drain(3))
    return 6;
  auto resumed = simulation->Latest();
  std::filesystem::remove(path);
  if (resumed->time != reference->time || resumed->dt != reference->dt || resumed->force.x != reference->force.x ||
      resumed->force.y != reference->force.y || resumed->force.z != reference->force.z ||
      resumed->moment.x != reference->moment.x || resumed->moment.y != reference->moment.y ||
      resumed->moment.z != reference->moment.z)
    return 7;
  for (int k = 0; k < 3; ++k)
    if (resumed->impulse[k] != reference->impulse[k] || resumed->angular_impulse[k] != reference->angular_impulse[k])
      return 8;
  std::cout << "PASS uniform checkpoint resumes exact loads, integrals and clock with a bounded snapshot queue\n";
}
