#pragma once
#include "amr_gpu_exchange.h"

namespace wind3d::amr {
// Distributed building blocks for the compatible pressure operator. D is
// negative integrated flux divergence; D^T is not yet the velocity gradient
// (which requires M^-1). No pressure boundary pinning or solve is performed.
class GpuPressureOperators {
 public:
  GpuPressureOperators();
  ~GpuPressureOperators();
  GpuPressureOperators(const GpuPressureOperators &) = delete;
  GpuPressureOperators &operator=(const GpuPressureOperators &) = delete;
  bool Build(const Mesh &mesh, const Topology &tree, const std::vector<int> &devices);
  const std::vector<PressureLocalLayout> &Layouts() const;
  // Layout-sized device arrays, disjoint input/output allocations. Input
  // imports are refreshed; only owned output entries are written. Producers
  // must finish before calling; all rank kernels finish before return.
  bool Divergence(const std::vector<double *> &faces, const std::vector<double *> &cells);
  bool TransposeDivergence(const std::vector<double *> &cells, const std::vector<double *> &faces);
  bool Mass(const std::vector<double *> &input, const std::vector<double *> &output);

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
