#pragma once
#include <memory>
#include <string>

#include "amr_pressure_hierarchy.h"

namespace wind3d::amr {
// Distributed symmetric V-cycle for the two-point pressure preconditioner.
class GpuMultilevel {
 public:
  GpuMultilevel();
  ~GpuMultilevel();
  GpuMultilevel(const GpuMultilevel &) = delete;
  GpuMultilevel &operator=(const GpuMultilevel &) = delete;
  bool Build(const PressureHierarchy &hierarchy, const std::vector<int> &devices);
  // Arrays follow the finest level's owned_cells order. Only owned entries are
  // read/written; caller ghosts are untouched. Producers must finish first.
  // Input/output allocations must be disjoint. All work completes on return.
  bool Apply(const std::vector<double *> &input, const std::vector<double *> &output);

  const std::string &Error() const {
    return error_;
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string error_;
};
}  // namespace wind3d::amr
