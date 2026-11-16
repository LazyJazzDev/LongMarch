#pragma once
#include "amr_gpu_flow.h"

namespace wind3d::amr {
// Distributed compatible-flow runtime. Only explicit snapshots/adaptation
// gather fields on the host; ordinary steps and speed reduction stay on GPUs.
class GpuDistributedFlow {
 public:
  using Distance = GpuFlow::Distance;
  GpuDistributedFlow();
  ~GpuDistributedFlow();
  GpuDistributedFlow(const GpuDistributedFlow &) = delete;
  GpuDistributedFlow &operator=(const GpuDistributedFlow &) = delete;
  // Geometry is retained by value through the callback, as for GpuFlow.
  // Initialization replaces state only on success.
  bool Initialize(const std::vector<int> &devices,
                  const Topology &tree,
                  int cells_per_block,
                  double root_width,
                  std::array<double, 3> origin,
                  const Distance &distance,
                  double speed);
  // Transactional same-domain remapping and projection. Physical pressure is
  // cleared and invalidated; time and step count persist across successful regrids.
  bool Regrid(const Topology &tree);
  bool Adapt(const RefinementOptions &options, RefinementReport &report);
  AdvanceResult Advance(double dt, double speed, const FlowParameters &parameters = {});
  bool MaximumSpeed(double &speed);
  bool Readback(FlowSnapshot &snapshot) const;
  FlowInfo Info() const;
  std::string Error() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  mutable std::string error_;
};
}  // namespace wind3d::amr
