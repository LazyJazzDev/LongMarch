#pragma once
#include "amr_transport.h"

namespace wind3d::amr {
// Transfer between hierarchies of the same physical domain and cells/block.
// Coarsening integrates open-face flux. Refinement preserves each old face's
// aggregate flux while bounding reconstructed child values. New interior faces
// are sampled from the old field. Fluid-adjacent cut faces must retain their
// locations, areas and apertures, preserving wall distance as well as flux.
// The result requires projection before use as a divergence-free flow state;
// projection potential is a remapping correction, not physical pressure.
// Host-side reference/cold regrid path; failure leaves destination unchanged.
bool TransferVelocity(const Mesh &source,
                      const Topology &source_tree,
                      const std::vector<double> &velocity,
                      const Mesh &target,
                      const Topology &target_tree,
                      double root_width,
                      std::array<double, 3> origin,
                      std::vector<double> &destination);
}  // namespace wind3d::amr
