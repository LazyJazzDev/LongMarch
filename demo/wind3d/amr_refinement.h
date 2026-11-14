#pragma once
#include "amr_mesh.h"

namespace wind3d::amr {
struct ProtectedBox {
  std::array<double, 3> lower{}, upper{};
};

struct RefinementOptions {
  int max_level{2}, buffer_blocks{1};
  size_t capacity{100000};
  double refine_threshold{.08}, coarsen_threshold{.03}, reference_speed{1}, protect_padding{};
  std::vector<ProtectedBox> protected_regions;
};

struct RefinementReport {
  size_t old_blocks{}, new_blocks{}, marked_refine{};
  double max_indicator{};
  bool changed{};
};

// Conservative body bounds, not pointwise distance tests: aircraft distance
// functions may clamp far-field values and need not be true signed distances.
bool ProtectedTopology(std::array<int, 3> roots,
                       double root_width,
                       std::array<double, 3> origin,
                       const RefinementOptions &options,
                       Topology &destination);
// Relative cell-scale velocity variation is a refinement heuristic, not a
// certified load-error bound. Coarsening uses a lower threshold (hysteresis),
// at most one level per update. Existing cut-cell blocks retain their spacing.
// All outputs remain unchanged on invalid data/capacity exhaustion.
bool PlanRefinement(const Mesh &mesh,
                    const Topology &tree,
                    const std::vector<double> &velocity,
                    double root_width,
                    std::array<double, 3> origin,
                    const RefinementOptions &options,
                    Topology &destination,
                    RefinementReport &report);
}  // namespace wind3d::amr
