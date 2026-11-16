#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "amr_advance.h"
#include "amr_refinement.h"

namespace wind3d::amr {
struct FlowParameters {
  bool cubic{true};
  double pressure_tolerance{1e-8};
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
  bool compatible_pressure{};
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
                  double speed,
                  bool compatible_pressure = false);
  // Same physical domain/body. Builds, transfers and projects in temporary
  // storage, then commits. Remap potential is discarded; pressure_valid remains
  // false until the next successful physical step. Time and step count persist.
  bool Regrid(const Topology &tree);
  // Explicit adaptation interval: downloads velocity for the host policy, then
  // uses the transactional regrid path. Report is committed only on success.
  bool Adapt(const RefinementOptions &options, RefinementReport &report);
  AdvanceResult Advance(double dt, double speed, const FlowParameters &parameters = {});
  bool MaximumSpeed(double &speed);
  bool Readback(FlowSnapshot &snapshot) const;
  // Restore into an initialized flow with the same domain, body and pressure
  // operator. Rebuilds snapshot topology and validates geometry/finite fields
  // in temporary storage. Failure preserves the live state; no remap/projection.
  bool Restore(const FlowSnapshot &snapshot);
  // Versioned, checksummed file checkpoint; save replaces the destination only
  // after a complete temporary write. Load requires initialized matching domain,
  // body and pressure mode. Caller must retain the same physical/time-step options.
  bool SaveCheckpoint(const std::string &path) const;
  bool LoadCheckpoint(const std::string &path);
  FlowInfo Info() const;
  std::string Error() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  mutable std::string error_;
};
}  // namespace wind3d::amr
