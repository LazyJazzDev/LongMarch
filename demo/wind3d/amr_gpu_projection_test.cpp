#include "amr_gpu_projection.h"

#include <cmath>
#include <iostream>

int Run(bool embedded) {
  using namespace wind3d::amr;
  Topology tree;
  if (!tree.Regrid({2, 2, 2}, 1, 100, [](Block b) { return b.x == 0 && b.y == 0; }))
    return 1;
  Mesh mesh;
  if (!mesh.Build(tree, 4, 1, {-1, -1, -1}))
    return 2;
  if (embedded && !mesh.Embed([](std::array<double, 3> p) {
        return std::max({std::abs(p[0] + .5), std::abs(p[1] + .5), std::abs(p[2])}) - .21;
      }))
    return 16;
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
  auto reference_u = u;
  std::vector<double> reference_p;
  auto reference = ProjectReference(mesh, g, reference_u, reference_p);
  if (!reference.converged)
    return 8;
  GpuProjection gpu;
  if (!gpu.Build(0, mesh, g)) {
    std::cerr << gpu.Error() << '\n';
    return 9;
  }
  auto solve = gpu.Project(u, p);
  if (solve.converged) {
    double error = 0, scale = 0;
    for (size_t i = 0; i < u.size(); ++i) {
      error = std::max(error, std::abs(u[i] - reference_u[i]));
      scale = std::max(scale, std::abs(reference_u[i]));
    }
    std::cout << "CPU/GPU corrected velocity max difference=" << error << '\n';
    if (error > 1e-7 * std::max(1.0, scale))
      return 10;
  }
  std::cout << "projection iterations=" << solve.iterations << " true relative residual=" << solve.relative_residual
            << '\n';
  if (!solve.converged)
    return 4;
  double initial = 0, final = 0;
  for (size_t i = 0; i < mesh.Cells().size(); ++i) {
    double a = 0, b = 0;
    for (int k = mesh.Offsets()[i]; k < mesh.Offsets()[i + 1]; ++k) {
      int id = mesh.Incidence()[k], f = std::abs(id) - 1;
      double area = (id > 0 ? 1 : -1) * mesh.Faces()[f].area * mesh.Faces()[f].aperture;
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
  auto fail = gpu.Project(u, p, 1e-12, 1);
  if (fail.converged || u != before || !p.empty())
    return 7;
  // Persistent workspace recovers after an iteration-limit failure.
  auto again = gpu.Project(u, p);
  if (!again.converged)
    return 11;
  std::vector<double> zero(u.size());
  p.clear();
  if (!gpu.Project(zero, p).converged)
    return 12;
  for (double value : p)
    if (value != 0)
      return 13;
  Mesh invalid;
  if (gpu.Build(0, invalid, g) || gpu.Error().empty())
    return 14;
  u = before;
  if (!gpu.Project(u, p).converged)
    return 15;
  std::cout
      << "PASS composite projection removes divergence, preserves fixed boundary flux and rolls back nonconvergence\n";
  return 0;
}

int main() {
  int status = Run(false);
  return status ? status : Run(true);
}
