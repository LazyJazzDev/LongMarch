#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "amr_advance.h"

namespace wind3d::amr {
struct FlowParameters {
  bool cubic{true};
  double density{1.225}, wall_viscosity{9.2e-7}, viscosity{1.5e-5}, sponge_length{2};
};

struct FlowInfo {
  size_t cells{}, faces{}, blocks{};
  uint64_t steps{}, generation{};
  double time{};
  bool pressure_valid{};
};

// Copies, not mutable views into the GPU flow. Readback is an explicit cold path.
struct FlowSnapshot {
  FlowInfo info;
  Topology topology;
  Mesh mesh;
  std::vector<double> velocity, pressure;
};

class GpuFlow {
 public:
  using Distance = std::function<double(std::array<double, 3>)>;
  GpuFlow();
  ~GpuFlow();
  GpuFlow(const GpuFlow &) = delete;
  GpuFlow &operator=(const GpuFlow &) = delete;
  // Distance is retained for regrids: capture immutable geometry by value.
  // Empty distance means all fluid. Initialization replaces state only on success.
  bool Initialize(int device,
                  const Topology &tree,
                  int cells_per_block,
                  double root_width,
                  std::array<double, 3> origin,
                  const Distance &distance,
                  double speed);
  // Same physical domain/body. Builds, transfers and projects in temporary
  // storage, then commits. Remap potential is discarded; pressure_valid remains
  // false until the next successful physical step. Time and step count persist.
  bool Regrid(const Topology &tree);
  AdvanceResult Advance(double dt, double speed, const FlowParameters &parameters = {});
  bool Readback(FlowSnapshot &snapshot) const;
  FlowInfo Info() const;
  std::string Error() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  mutable std::string error_;
};
}  // namespace wind3d::amr
