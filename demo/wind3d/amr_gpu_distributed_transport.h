#pragma once
#include <memory>
#include <optional>
#include <string>

#include "amr_mesh.h"

namespace wind3d::amr {
struct TransportLoads {
  // Force XYZ followed by moment XYZ about the world/body origin.
  std::array<double, 6> pressure{}, friction{}, total{};
};

// Single-node advection with uniquely owned dynamic fields and peer sampling.
// Geometry is replicated. Uses the same sampling kernels as GpuTransport;
// topology changes require Build. Projection/runtime are separate.
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
  const std::vector<std::vector<int>> &OwnedCells() const;
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

  // Corrected advection followed by molecular diffusion from the original
  // slip-extended field, then outlet sponge and prescribed normal boundaries.
  // Requires dt > 0 and dt*viscosity/min(face area) <= 1/6. Same owned-prefix,
  // aliasing and transactional-output contract as Advect; no hidden prior call.
  bool AdvectDiffuse(const std::vector<const double *> &velocity,
                     const std::vector<double *> &destination,
                     double dt,
                     double speed,
                     bool cubic = true,
                     double viscosity = 1.5e-5,
                     double sponge_length = 2);

  // Read-only owned prefixes: velocity uses OwnedFaces, physical pressure uses
  // OwnedCells. Extend pre-step velocity with the same two slip layers as
  // transport before evaluating load-only wall friction. No prior call needed.
  // Producers must finish first. Every failure leaves loads and inputs unchanged.
  bool EvaluateLoads(const std::vector<const double *> &velocity,
                     const std::vector<const double *> &pressure,
                     TransportLoads &loads,
                     double density = 1.225,
                     double wall_viscosity = 9.2e-7);

  // Area-weighted cell-centre speed maximum, matching GpuTransport. No slip
  // extension is applied. Read-only owned prefixes; failure preserves speed.
  bool MaximumSpeed(const std::vector<const double *> &velocity, double &speed);

  const std::string &Error() const {
    return error_;
  }

 private:
  struct Diffusion {
    double viscosity, sponge_length, speed;
  };

  bool Apply(const std::vector<const double *> &velocity,
             const std::vector<double *> &destination,
             double dt,
             bool cubic,
             bool corrected,
             std::optional<double> inlet_speed,
             std::optional<Diffusion> diffusion);
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string error_;
};
}  // namespace wind3d::amr
