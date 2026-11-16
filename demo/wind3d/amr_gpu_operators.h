#pragma once
#include "amr_gpu_exchange.h"

namespace wind3d::amr {
// Distributed building blocks for the compatible pressure operator. D is
// negative integrated flux divergence; D^T is not yet the velocity gradient
// (which requires M^-1). Gradient and Pressure compose these building blocks;
// Pressure pins inactive rows. A pressure iteration is not performed here.
class GpuPressureOperators {
 public:
  GpuPressureOperators();
  ~GpuPressureOperators();
  GpuPressureOperators(const GpuPressureOperators &) = delete;
  GpuPressureOperators &operator=(const GpuPressureOperators &) = delete;
  bool Build(const Mesh &mesh, const Topology &tree, const std::vector<int> &devices, bool pressure_boundaries = true);
  const std::vector<PressureLocalLayout> &Layouts() const;
  // Layout-sized device arrays, disjoint input/output allocations. Input
  // imports are refreshed; only owned output entries are written. Producers
  // must finish before calling; all rank kernels finish before return.
  bool Divergence(const std::vector<double *> &faces, const std::vector<double *> &cells);
  bool TransposeDivergence(const std::vector<double *> &cells, const std::vector<double *> &faces);
  bool Mass(const std::vector<double *> &input, const std::vector<double *> &output);

  // Fixed symmetric Chebyshev inverse of the free-face principal mass matrix.
  // Closed faces, and optionally fixed inlet/wall normals, return zero.
  bool InverseMass(const std::vector<double *> &input, const std::vector<double *> &output);

  bool Gradient(const std::vector<double *> &cells, const std::vector<double *> &faces);
  // Two-point D diag(area * distance * aperture)^-1 D^T approximation,
  // with the same free-face boundary mask and inactive identity rows.
  // Used for multilevel preconditioning; not a replacement for Pressure.
  bool TwoPointPressure(const std::vector<double *> &input, const std::vector<double *> &output);
  bool Pressure(const std::vector<double *> &input, const std::vector<double *> &output);

  const std::string &Error() const {
    return error_;
  }

 private:
  bool Apply(const std::vector<double *> &input, const std::vector<double *> &output, int operation);
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string error_;
};
}  // namespace wind3d::amr
