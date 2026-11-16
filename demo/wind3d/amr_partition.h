#pragma once
#include "amr_mass.h"

namespace wind3d::amr {
struct PressureExchange {
  int source{}, destination{};
  std::vector<int> cells, faces;
};

// Block-balanced ownership and directed imports for D, D^T and the face mass
// matrix, plus the fine two-point pressure stencil. IDs remain global; exchange payloads are sorted and duplicate-free.
// This does not include transport halos or coarse-level preconditioner exchanges.
class PressurePartition {
 public:
  // Inputs must describe the same mesh. Failure preserves the previous plan.
  bool Build(const Mesh &mesh, const Topology &tree, const FaceMass &mass, int ranks);

  const std::vector<int> &CellOwners() const {
    return cell_owners_;
  }

  const std::vector<int> &FaceOwners() const {
    return face_owners_;
  }

  const std::vector<std::vector<int>> &OwnedCells() const {
    return cells_;
  }

  const std::vector<std::vector<int>> &OwnedFaces() const {
    return faces_;
  }

  const std::vector<PressureExchange> &Exchanges() const {
    return exchanges_;
  }

 private:
  std::vector<int> cell_owners_, face_owners_;
  std::vector<std::vector<int>> cells_, faces_;
  std::vector<PressureExchange> exchanges_;
};
}  // namespace wind3d::amr
