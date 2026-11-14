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
  bool Build(int device, const Mesh &mesh, const FaceGradient &gradient);
  ProjectionResult Project(std::vector<double> &velocity,
                           std::vector<double> &potential,
                           double tolerance = 1e-9,
                           int max_iterations = 2000);
  std::string Error() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string build_error_;
};
}  // namespace wind3d::amr
