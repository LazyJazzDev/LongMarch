#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "aircraft.h"

namespace wind3d {

struct Cell {
  float u{}, v{}, w{}, pressure{};
};

struct Vertex {
  Vec3 position, normal, color;
};

std::vector<Vertex> AircraftMesh(const Controls &controls);

struct Snapshot {
  int nx{}, ny{}, nz{};
  uint64_t step{}, generation{};
  float time{}, dt{}, divergence_before{}, divergence_after{}, step_seconds{};
  Vec3 force{}, moment{};
  Controls controls;
  std::vector<Cell> cells;
  std::shared_ptr<const std::vector<Vertex>> aircraft;
};

// Single worker owns all CUDA state. Published snapshots are immutable: the
// renderer never waits for a solver step or observes half-written velocity data.
class Simulation {
 public:
  static std::unique_ptr<Simulation> Create(int cuda_device, int nx);
  ~Simulation();
  Simulation(const Simulation &) = delete;
  Simulation &operator=(const Simulation &) = delete;

  void Configure(const Controls &controls, bool paused);
  void Reset();
  bool Paused() const;
  std::shared_ptr<const Snapshot> Latest() const;
  std::string Error() const;

 private:
  Simulation();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace wind3d
