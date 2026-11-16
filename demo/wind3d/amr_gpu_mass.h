#pragma once
#include <memory>
#include <string>

#include "amr_mesh.h"

namespace wind3d::amr {
class GpuMassInverse {
 public:
  GpuMassInverse();
  ~GpuMassInverse();
  // Closed faces are fixed at zero. Optionally fix inlet/wall normal pressure
  // corrections as well, leaving the pressure outlet free. Failed builds
  // retain the previous inverse.
  bool Build(int device, const Mesh &mesh, bool pressure_boundaries = true);
  // Enqueues on the default CUDA stream; caller synchronizes before reading.
  // Input/output must be distinct. The instance owns non-concurrent workspace.
  bool ApplyDevice(const double *rhs, double *result, size_t faces);
  size_t CoupledFaces() const;
  std::string Error() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string error_;
};
}  // namespace wind3d::amr
