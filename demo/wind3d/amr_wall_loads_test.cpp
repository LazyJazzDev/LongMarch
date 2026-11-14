#include "amr_wall_loads.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "amr_transport.h"
#include "wall_model.h"

// Independent bracketed double-precision evaluation of the implicit law.
double ReferenceViscosity(double nu, double speed, double y) {
  double re = speed * y / nu;
  if (re <= 1)
    return nu;
  double lo = 0, hi = 100;
  for (int i = 0; i < 100; ++i) {
    double u = .5 * (lo + hi), k = .41 * u;
    double yp = u + .1185 * (std::expm1(k) - k - .5 * k * k - k * k * k / 6);
    if (u * yp > re)
      hi = u;
    else
      lo = u;
  }
  double u = .5 * (lo + hi);
  return nu * re / (u * u);
}

int main() {
  using namespace wind3d::amr;
  for (double re : {.01, 1., 10., 1000., 1e5, 1e8}) {
    double expected = ReferenceViscosity(1e-6, re * 1e-6, .9999999999999999);
    double actual = wind3d::WallEffectiveViscosity(1e-6f, float(re * 1e-6), 1.f);
    if (std::abs(actual - expected) > 5e-5 * expected)
      return 1;
  }
  Topology tree;
  tree.Regrid({2, 2, 2}, 0, 8, [](Block) { return false; });
  Mesh mesh;
  if (!mesh.Build(tree, 4, 1, {-1, -1, -1}) || !mesh.Embed([](std::array<double, 3> p) { return p[1]; }))
    return 2;
  std::vector<double> velocity(mesh.Faces().size());
  for (size_t f = 0; f < velocity.size(); ++f)
    velocity[f] = mesh.Faces()[f].axis == 0 && mesh.Faces()[f].aperture > 0 ? 2 : 0;
  FaceField field(mesh, tree, 1, {-1, -1, -1});
  if (!field.ExtendSlip(velocity))
    return 3;
  for (size_t f = 0; f < velocity.size(); ++f) {
    auto face = mesh.Faces()[f];
    if (face.axis != 0)
      continue;
    if ((face.centre[1] == -.125 || face.centre[1] == -.375) && std::abs(velocity[f] - 2) > 1e-12)
      return 4;
    if (face.centre[1] == -.625 && velocity[f] != 0)
      return 5;
  }
  std::array<double, 6> loads{};
  if (!WallLoads(mesh, velocity, 1.225, 9.2e-7, loads))
    return 6;
  double expected = 4 * 1.225 * ReferenceViscosity(9.2e-7, 2, .125) * 2 / .125;
  if (std::abs(loads[0] - expected) > 5e-5 * expected || std::abs(loads[5] + .125 * expected) > 5e-5 * expected)
    return 7;
  for (int k : {1, 2, 3, 4})
    if (std::abs(loads[k]) > 1e-12)
      return 8;
  std::cout << "flat-wall shear force=" << loads[0] << " independent expected=" << expected << " Mz=" << loads[5]
            << '\n';
  // Velocity normal to the wall must not produce tangential friction.
  for (size_t f = 0; f < velocity.size(); ++f)
    velocity[f] = mesh.Faces()[f].axis == 1 ? 3 : 0;
  if (!WallLoads(mesh, velocity, 1.225, 9.2e-7, loads))
    return 9;
  for (double f : loads)
    if (std::abs(f) > 1e-12)
      return 10;
  auto before = loads;
  if (WallLoads(mesh, velocity, 0, 9.2e-7, loads) || loads != before)
    return 11;
  std::cout
      << "PASS shared wall law, two-layer slip extension, tangential shear, moment arm and failed-input rollback\n";
}
