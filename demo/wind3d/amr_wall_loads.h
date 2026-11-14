#pragma once
#include "amr_mesh.h"

namespace wind3d::amr {
// Load-only log-law split: extended pre-step velocities, half-cell sample
// distance, closed-area normal and moment about the world/body origin.
bool WallLoads(const Mesh &mesh,
               const std::vector<double> &extended_velocity,
               double density,
               double molecular_viscosity,
               std::array<double, 6> &loads);
}  // namespace wind3d::amr
