#pragma once
#include "amr_mesh.h"

namespace wind3d::amr {
// Interior normal pressure derivative on each unique face. Boundary rows are
// empty: boundary conditions must be supplied by the projection operator.
// Equal-size faces use the centred difference. Coarse/fine faces reproduce
// quadratic polynomials using a local weighted reconstruction at the face.
// The resulting composite operator need not be symmetric; plain CG is invalid.
class FaceGradient {
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

 private:
  std::vector<int> offsets_, columns_;
  std::vector<double> weights_;
};
}  // namespace wind3d::amr
