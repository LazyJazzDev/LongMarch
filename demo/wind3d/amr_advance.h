#pragma once
#include "amr_projection.h"
#include "amr_transport.h"
#include "amr_wall_loads.h"

namespace wind3d::amr {
struct AdvanceResult {
  bool advanced{};
  ProjectionResult projection;
  // GPU stage wall times, including host synchronization; zero in CPU reference.
  double transport_seconds{}, pressure_seconds{}, loads_seconds{};
  std::array<double, 6> pressure_loads{}, friction_loads{}, total_loads{};
};

// CPU reference with corrected transport, slip walls, molecular diffusion,
// outlet sponge and load-only wall friction. Dynamic regridding is separate.
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
                               double density = 1.225,
                               double wall_viscosity = 9.2e-7,
                               double viscosity = 1.5e-5,
                               double sponge_length = 2,
                               bool corrected_transport = true);
}  // namespace wind3d::amr
