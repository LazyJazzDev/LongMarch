#pragma once
#include "amr_topology.h"

namespace wind3d::amr {
struct MeshCell {
  std::array<double, 3> centre{};
  double width{};
};

// A face is a unique flux degree of freedom shared by both cells. A coarse
// face has four fine patches, not a duplicate coarse flux. -1 denotes exterior.
struct MeshFace {
  int low{-1}, high{-1}, axis{};
  std::array<double, 3> centre{};
  double area{}, distance{};
};

class Mesh {
 public:
  // Physical root blocks have side root_width; cells per block must be even.
  // Failure does not alter an existing mesh. No field allocations occur here.
  bool Build(const Topology &tree, int cells_per_block, double root_width, std::array<double, 3> origin = {});

  const std::vector<MeshCell> &Cells() const {
    return cells_;
  }

  const std::vector<MeshFace> &Faces() const {
    return faces_;
  }

  // Flat CSR cell-to-face incidence for GPU cell gathers: +(face+1) at the
  // low cell, -(face+1) at the high cell. No floating-point atomics needed.
  const std::vector<int> &Offsets() const {
    return offsets_;
  }

  const std::vector<int> &Incidence() const {
    return incidence_;
  }

  int CellsPerBlock() const {
    return cells_per_block_;
  }

 private:
  int cells_per_block_{};
  std::vector<MeshCell> cells_;
  std::vector<MeshFace> faces_;
  std::vector<int> offsets_, incidence_;
};
}  // namespace wind3d::amr
