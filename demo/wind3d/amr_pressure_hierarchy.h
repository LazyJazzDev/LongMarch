#pragma once
#include "amr_mesh.h"

namespace wind3d::amr {
struct PressureLevelExchange {
  int source{}, destination{};
  std::vector<int> cells;
};

struct PressureLevel {
  std::vector<int> offsets, columns, owners, parent;
  std::vector<double> weights, diagonal;
  std::vector<std::vector<int>> owned_cells;
  std::vector<PressureLevelExchange> exchanges;
  bool identity{};
};

// Immutable host assembly for distributed geometric aggregation. R=P^T sums
// integrated residuals; P is constant per aggregate and zero on inactive cells.
// Each exchange plan covers A, restriction into the next level, and prolongation
// from this level into the previous one. IDs are global within a level.
class PressureHierarchy {
 public:
  // Fine owners come from the matching pressure partition. Failure preserves
  // the previous hierarchy. Only the standard inlet/wall/outlet mask is used.
  bool Build(const Mesh &mesh, const std::vector<int> &fine_owners, int ranks, int coarse_limit = 64);

  const std::vector<PressureLevel> &Levels() const {
    return levels_;
  }

 private:
  std::vector<PressureLevel> levels_;
};
}  // namespace wind3d::amr
