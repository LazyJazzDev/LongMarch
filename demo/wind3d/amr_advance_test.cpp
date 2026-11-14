#include "amr_advance.h"

#include <cmath>
#include <iostream>

#include "aircraft.h"

int main() {
  using namespace wind3d::amr;
  Topology tree;
  if (!tree.Regrid({8, 4, 6}, 1, 1000,
                   [](Block b) { return b.x >= 1 && b.x <= 4 && b.y >= 1 && b.y <= 2 && b.z >= 1 && b.z <= 4; }))
    return 1;
  Mesh mesh;
  if (!mesh.Build(tree, 4, 2, {-6, -4, -6}))
    return 2;
  std::vector<double> u(mesh.Faces().size()), p;
  for (size_t f = 0; f < u.size(); ++f)
    u[f] = mesh.Faces()[f].axis == 0 ? 1 : 0;
  auto before = u;
  auto free = AdvanceReference(mesh, tree, 2, {-6, -4, -6}, .01, 1, u, p);
  if (!free.advanced)
    return 3;
  for (size_t i = 0; i < u.size(); ++i)
    if (std::abs(u[i] - before[i]) > 1e-12)
      return 4;
  wind3d::Controls controls;
  controls.angle = 4;
  if (!mesh.Embed([&](std::array<double, 3> x) {
        return double(wind3d::AircraftDistance({float(x[0]), float(x[1]), float(x[2])}, controls));
      }))
    return 5;
  for (int step = 0; step < 3; ++step) {
    auto result = AdvanceReference(mesh, tree, 2, {-6, -4, -6}, .01, 1, u, p);
    std::cout << "aircraft reference step=" << step + 1 << " residual=" << result.projection.relative_residual
              << " Fx=" << result.pressure_loads[0] << " Fy=" << result.pressure_loads[1] << '\n';
    if (!result.advanced)
      return 6;
    if (!(result.friction_loads[0] > 0))
      return 9;
    for (int k = 0; k < 6; ++k)
      if (result.total_loads[k] != result.pressure_loads[k] + result.friction_loads[k])
        return 10;
    for (double value : u)
      if (!std::isfinite(value))
        return 7;
  }
  before = u;
  auto old_pressure = p;
  if (AdvanceReference(mesh, tree, 2, {-6, -4, -6}, -1, 1, u, p).advanced || u != before || p != old_pressure)
    return 8;
  std::cout
      << "PASS uniform freestream preservation and coarse-aircraft slip-wall time-step smoke test (not steady-load validation)\n";
}
