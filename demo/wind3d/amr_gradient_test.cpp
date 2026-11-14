#include "amr_gradient.h"

#include <algorithm>
#include <cmath>
#include <iostream>

int main() {
  using namespace wind3d::amr;
  double errors[2]{};
  for (int resolution = 0; resolution < 2; ++resolution) {
    Topology tree;
    if (!tree.Regrid({2, 2, 2}, 1, 100, [](Block b) { return b.x == 0 && b.y == 0; }))
      return 1;
    Mesh mesh;
    if (!mesh.Build(tree, 4 << resolution, 1, {-1, -1, -1}))
      return 2;
    FaceGradient gradient;
    if (!gradient.Build(mesh))
      return 3;
    for (size_t f = 0; f < mesh.Faces().size(); ++f) {
      auto face = mesh.Faces()[f];
      if (face.low < 0 || face.high < 0)
        continue;
      // Independently exercise all ten polynomial basis fields.
      for (int term = 0; term < 10; ++term) {
        auto polynomial = [&](std::array<double, 3> p) {
          if (term == 0)
            return 1.0;
          if (term < 4)
            return p[term - 1];
          if (term < 7)
            return p[term - 4] * p[term - 4];
          int a = term == 7 ? 0 : term == 8 ? 0 : 1, b = term == 7 ? 1 : 2;
          return p[a] * p[b];
        };
        double actual = 0;
        for (int k = gradient.Offsets()[f]; k < gradient.Offsets()[f + 1]; ++k)
          actual += gradient.Weights()[k] * polynomial(mesh.Cells()[gradient.Columns()[k]].centre);
        auto a = face.centre, b = a;
        a[face.axis] += .5;
        b[face.axis] -= .5;
        double exact = polynomial(a) - polynomial(b);
        if (std::abs(actual - exact) > 1e-10) {
          std::cerr << "polynomial failure " << term << ' ' << actual - exact << '\n';
          return 4;
        }
      }
      double actual = 0;
      for (int k = gradient.Offsets()[f]; k < gradient.Offsets()[f + 1]; ++k) {
        auto p = mesh.Cells()[gradient.Columns()[k]].centre;
        actual += gradient.Weights()[k] * (p[0] * p[0] * p[0] + p[1] * p[1] * p[1] + p[2] * p[2] * p[2]);
      }
      double exact = 3 * face.centre[face.axis] * face.centre[face.axis];
      errors[resolution] = std::max(errors[resolution], std::abs(actual - exact));
    }
  }
  std::cout << "cubic manufactured-field max errors " << errors[0] << ' ' << errors[1] << '\n';
  if (errors[1] > .3 * errors[0])
    return 5;
  std::cout
      << "PASS constant, linear and quadratic pressure derivatives including coarse/fine interfaces; second-order cubic convergence\n";
}
