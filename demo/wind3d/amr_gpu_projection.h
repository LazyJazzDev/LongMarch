#pragma once
#include <memory>
#include <string>

#include "amr_projection.h"

namespace wind3d::amr {
class GpuProjection {
 public:
  GpuProjection();
  ~GpuProjection();
  GpuProjection(const GpuProjection &) = delete;
  GpuProjection &operator=(const GpuProjection &) = delete;
  // Fixed linear aggregation V-cycle by default; false selects Jacobi for comparison.
  // Four threads share a matrix row by default; false retains scalar rows for comparison.
  // Compatible mode uses D M^-1 D^T with a two-point multilevel preconditioner.
  // In compatible mode the supplied gradient is unused and may be empty.
  // It uses CG by default; conjugate=false retains BiCGStab for comparison.
  // The original nonsymmetric gradient always uses BiCGStab.
  bool Build(int device,
             const Mesh &mesh,
             const FaceGradient &gradient,
             bool multilevel = true,
             bool cooperative = true,
             bool compatible = false,
             bool conjugate = true);
  ProjectionResult Project(std::vector<double> &velocity,
                           std::vector<double> &potential,
                           double tolerance = 1e-9,
                           int max_iterations = 2000);
  // Caller-owned, disjoint buffers on the Build device, in mesh face/cell order.
  // Uses the default stream; caller must finish writes on other streams first.
  // Numerical nonconvergence leaves both buffers unchanged. Runtime/device
  // errors are reported by Error(); output contents are then unspecified.
  ProjectionResult ProjectDevice(double *velocity,
                                 size_t faces,
                                 double *potential,
                                 size_t cells,
                                 double tolerance = 1e-9,
                                 int max_iterations = 2000);
  std::string Error() const;

 private:
  ProjectionResult Solve(double tolerance, int max_iterations);
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string build_error_;
};
}  // namespace wind3d::amr
