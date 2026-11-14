#include "amr_projection.h"

#include <cmath>
#include <iostream>

int main() {
  using namespace wind3d::amr;
  Topology tree;
  if (!tree.Regrid({2, 2, 2}, 1, 100, [](Block b) { return b.x == 0 && b.y == 0; }))
    return 1;
  Mesh mesh;
  if (!mesh.Build(tree, 4, 1, {-1, -1, -1}))
    return 2;
  FaceGradient g;
  if (!g.Build(mesh))
    return 3;
  std::vector<double> u(mesh.Faces().size()), p;
  constexpr double pi = 3.141592653589793;
  for (size_t i = 0; i < u.size(); ++i) {
    auto f = mesh.Faces()[i];
    u[i] = std::sin((f.centre[f.axis] + 1) * pi * (f.axis == 0 ? .25 : 1));
  }
  auto before = u;
  auto solve = ProjectReference(mesh, g, u, p);
  std::cout << "projection iterations=" << solve.iterations << " true relative residual=" << solve.relative_residual
            << '\n';
  if (!solve.converged)
    return 4;
  double initial = 0, final = 0;
  for (size_t i = 0; i < mesh.Cells().size(); ++i) {
    double a = 0, b = 0;
    for (int k = mesh.Offsets()[i]; k < mesh.Offsets()[i + 1]; ++k) {
      int id = mesh.Incidence()[k], f = std::abs(id) - 1;
      double area = (id > 0 ? 1 : -1) * mesh.Faces()[f].area;
      a += area * before[f];
      b += area * u[f];
    }
    initial += a * a;
    final += b * b;
  }
  if (std::sqrt(final / initial) > 1.1e-9)
    return 5;
  for (size_t i = 0; i < u.size(); ++i) {
    auto f = mesh.Faces()[i];
    if ((f.low < 0 || f.high < 0) && !(f.axis == 0 && f.high < 0) && u[i] != before[i])
      return 6;
  }
  u = before;
  p.clear();
  auto fail = ProjectReference(mesh, g, u, p, 1e-12, 1);
  if (fail.converged || u != before || !p.empty())
    return 7;
  std::cout
      << "PASS composite projection removes divergence, preserves fixed boundary flux and rolls back nonconvergence\n";
}
