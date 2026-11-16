#pragma once
#include <memory>
#include <optional>
#include <string>

#include "amr_mesh.h"

namespace wind3d::amr {
// Single-node advection with uniquely owned dynamic fields and peer sampling.
// Geometry is replicated. Uses the same sampling kernels as GpuTransport;
// topology changes require Build. Diffusion/projection/runtime are separate.
class GpuDistributedTransport {
 public:
  GpuDistributedTransport();
  ~GpuDistributedTransport();
  GpuDistributedTransport(const GpuDistributedTransport &) = delete;
  GpuDistributedTransport &operator=(const GpuDistributedTransport &) = delete;
  // Ownership matches PressurePartition. Required peer access remains enabled
  // for the CUDA context lifetime. Failure preserves the previous transport.
  bool Build(const Mesh &mesh,
             const Topology &tree,
             const std::vector<int> &devices,
             double root_width,
             std::array<double, 3> origin);
  const std::vector<std::vector<int>> &OwnedFaces() const;
  // Each allocation has at least OwnedFaces()[rank].size() entries, ordered by
  // those IDs. Only this prefix is read/written; pressure ghosts are untouched.
  // Producers must finish first. Per-rank input/output may alias. Returns after
  // all work finishes; numerical failure leaves every destination unchanged.
  // CUDA failures may leave outputs partial. No timestep/trace clipping.
  bool Advect(const std::vector<const double *> &velocity,
              const std::vector<double *> &destination,
              double dt,
              bool cubic = true,
              bool corrected = true,
              std::optional<double> inlet_speed = std::nullopt);

  const std::string &Error() const {
    return error_;
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string error_;
};
}  // namespace wind3d::amr
