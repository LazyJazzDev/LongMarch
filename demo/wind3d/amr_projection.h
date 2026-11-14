#pragma once
#include "amr_gradient.h"

namespace wind3d::amr {
struct ProjectionResult {
  bool converged{};
  int iterations{};
  double relative_residual{};
};

// CPU reference for the composite operator; used to validate the GPU port.
// Unit dt/rho: pressure is the velocity-potential correction. Fixed flux on
// inlet/lateral boundaries, zero potential at the +x outlet. All-fluid only.
ProjectionResult ProjectReference(const Mesh &mesh,
                                  const FaceGradient &gradient,
                                  std::vector<double> &velocity,
                                  std::vector<double> &potential,
                                  double tolerance = 1e-9,
                                  int max_iterations = 2000);
}  // namespace wind3d::amr
