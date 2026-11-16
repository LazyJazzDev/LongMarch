#pragma once
#include "amr_mesh.h"

namespace wind3d::amr {
// Compatible face-velocity mass matrix. Uniform cut cells retain diagonal
// aperture-weighted face volumes; fully open coarse cells with split faces
// use a mimetic local mass. Closed-face rows are pinned to one for inversion.
// This is an operator building block, not a replacement pressure solver.
class FaceMass {
 public:
  bool Build(const Mesh &mesh);

  const std::vector<int> &Offsets() const {
    return offsets_;
  }

  const std::vector<int> &Columns() const {
    return columns_;
  }

  const std::vector<double> &Weights() const {
    return weights_;
  }

  const std::vector<double> &Diagonal() const {
    return diagonal_;
  }

 private:
  std::vector<int> offsets_, columns_;
  std::vector<double> weights_, diagonal_;
};
}  // namespace wind3d::amr
