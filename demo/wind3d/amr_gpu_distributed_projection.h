#pragma once
#include "amr_gpu_operators.h"
#include "amr_projection.h"

namespace wind3d::amr {
// Compatible pressure projection with distributed vectors and host-reduced
// device dot products. Jacobi is a correctness baseline, not the final
// distributed multilevel preconditioner.
class GpuDistributedProjection {
 public:
  GpuDistributedProjection();
  ~GpuDistributedProjection();
  GpuDistributedProjection(const GpuDistributedProjection &) = delete;
  GpuDistributedProjection &operator=(const GpuDistributedProjection &) = delete;
  bool Build(const Mesh &mesh, const Topology &tree, const std::vector<int> &devices);
  const std::vector<PressureLocalLayout> &Layouts() const;
  // Layout-sized disjoint fields per rank. Producers must finish first.
  // Only owned velocity/potential entries commit on convergence; all caller
  // entries remain unchanged on numerical nonconvergence. CUDA errors can
  // leave outputs partial. Returns after device work completes.
  ProjectionResult Project(const std::vector<double *> &velocity,
                           const std::vector<double *> &potential,
                           double tolerance = 1e-9,
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
