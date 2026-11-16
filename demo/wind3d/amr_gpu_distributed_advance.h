#pragma once
#include <memory>
#include <string>

#include "amr_advance.h"
#include "amr_gpu_exchange.h"

namespace wind3d::amr {
// Transactional single-node timestep: distributed transport, compatible
// pressure projection, physical pressure and pre-step wall-friction loads.
class GpuDistributedAdvance {
 public:
  GpuDistributedAdvance();
  ~GpuDistributedAdvance();
  GpuDistributedAdvance(const GpuDistributedAdvance &) = delete;
  GpuDistributedAdvance &operator=(const GpuDistributedAdvance &) = delete;
  // Failure preserves the previous stepper. Peer access follows the component
  // solvers' context-lifetime contract. Topology changes require Build.
  bool Build(const Mesh &mesh,
             const Topology &tree,
             const std::vector<int> &devices,
             double root_width,
             std::array<double, 3> origin);
  const std::vector<PressureLocalLayout> &Layouts() const;
  // Disjoint layout-sized velocity/pressure fields per rank. Producers must
  // finish first. Only owned prefixes commit after every numerical stage
  // succeeds. Numerical failure leaves all caller entries unchanged; CUDA
  // errors during final copies may leave partial outputs. All work is drained.
  AdvanceResult Advance(const std::vector<double *> &velocity,
                        const std::vector<double *> &pressure,
                        double dt,
                        double speed,
                        bool cubic = true,
                        double density = 1.225,
                        double wall_viscosity = 9.2e-7,
                        double viscosity = 1.5e-5,
                        double sponge_length = 2,
                        double pressure_tolerance = 1e-8,
                        int max_iterations = 2000);

  const std::string &Error() const {
    return error_;
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string error_;
};
}  // namespace wind3d::amr
