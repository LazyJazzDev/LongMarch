#include <algorithm>
#include <cmath>
#include <iostream>

#include "aircraft.h"
#include "amr_projection.h"

int main() {
  using namespace wind3d::amr;
  Topology tree;
  if (!tree.Regrid({4, 4, 4}, 1, 1000,
                   [](Block b) { return b.x >= 1 && b.x <= 2 && b.y >= 1 && b.y <= 2 && b.z >= 1 && b.z <= 2; }))
    return 1;
  Mesh mesh;
  if (!mesh.Build(tree, 4, 1, {-2, -2, -2}))
    return 2;
  auto box = [](std::array<double, 3> p) { return std::max({std::abs(p[0]), std::abs(p[1]), std::abs(p[2])}) - .61; };
  if (!mesh.Embed(box))
    return 3;
  size_t partial = 0, closed = 0, inactive = 0;
  for (auto &f : mesh.Faces()) {
    partial += f.aperture > 0 && f.aperture < 1;
    closed += f.aperture == 0;
  }
  for (auto &c : mesh.Cells())
    inactive += !c.active;
  if (!partial || !closed || !inactive)
    return 4;
  std::vector<double> pressure(mesh.Cells().size(), 100);
  std::array<double, 6> loads{};
  if (!mesh.PressureLoads(pressure, loads))
    return 5;
  for (double f : loads)
    if (std::abs(f) > 1e-10)
      return 6;
  FaceGradient g;
  if (!g.Build(mesh))
    return 7;
  std::vector<double> u(mesh.Faces().size()), p;
  for (size_t i = 0; i < u.size(); ++i)
    u[i] = mesh.Faces()[i].axis == 0 ? 1 : 0;
  auto solve = ProjectReference(mesh, g, u, p, 1e-8, 2000);
  std::cout << "embedded box partial=" << partial << " closed=" << closed << " inactive=" << inactive
            << " iterations=" << solve.iterations << " residual=" << solve.relative_residual << '\n';
  if (!solve.converged)
    return 8;
  for (size_t i = 0; i < u.size(); ++i)
    if (mesh.Faces()[i].aperture == 0 && u[i] != 0)
      return 9;
  // A boundary crossing a refinement interface must be rejected, not silently
  // change the body representation to a coarse approximation.
  auto before = mesh.Faces();
  if (mesh.Embed([](std::array<double, 3> p) { return p[0] + .95; }))
    return 10;
  for (size_t i = 0; i < before.size(); ++i)
    if (before[i].aperture != mesh.Faces()[i].aperture)
      return 11;
  // Actual aircraft geometry, with all components inside a uniform fine region.
  Topology aircraft_tree;
  if (!aircraft_tree.Regrid({8, 4, 6}, 0, 192, [](Block) { return false; }))
    return 12;
  Mesh aircraft;
  if (!aircraft.Build(aircraft_tree, 4, 2, {-6, -4, -6}))
    return 13;
  wind3d::Controls controls;
  controls.angle = 4;
  controls.flap = 20;
  controls.elevator = -10;
  if (!aircraft.Embed([&](std::array<double, 3> p) {
        return double(wind3d::AircraftDistance({float(p[0]), float(p[1]), float(p[2])}, controls));
      }))
    return 14;
  pressure.assign(aircraft.Cells().size(), 100);
  if (!aircraft.PressureLoads(pressure, loads))
    return 15;
  for (double f : loads)
    if (std::abs(f) > 1e-8)
      return 16;
  std::cout
      << "PASS stationary cut-face projection, closed-surface aircraft force/moment cancellation and interface rollback\n";
}
