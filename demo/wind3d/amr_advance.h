#pragma once
#include "amr_projection.h"
#include "amr_transport.h"

namespace wind3d::amr {
struct AdvanceResult {
  bool advanced{};
  ProjectionResult projection;
  std::array<double, 6> pressure_loads{};
};

// CPU inviscid reference time step. Wall friction, viscous diffusion, sponge,
// MacCormack correction and regridding are not supplied by this reference.
// A failed transport/projection leaves the caller's complete state untouched.
AdvanceResult AdvanceReference(const Mesh &mesh,
                               const Topology &tree,
                               double root_width,
                               std::array<double, 3> origin,
                               double dt,
                               double speed,
                               std::vector<double> &velocity,
                               std::vector<double> &pressure,
                               bool cubic = true,
                               double density = 1.225);
}  // namespace wind3d::amr
