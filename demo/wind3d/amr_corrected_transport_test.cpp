#include <algorithm>
#include <cmath>
#include <iostream>

#include "amr_transport.h"

int main() {
  using namespace wind3d::amr;
  Topology tree;
  tree.Regrid({2, 2, 2}, 0, 8, [](Block) { return false; });
  Mesh mesh;
  if (!mesh.Build(tree, 8, 1, {-1, -1, -1}))
    return 1;
  FaceField field(mesh, tree, 1, {-1, -1, -1});
  std::vector<double> u(mesh.Faces().size()), first, corrected;
  for (size_t f = 0; f < u.size(); ++f) {
    auto face = mesh.Faces()[f];
    u[f] = face.axis == 0 ? std::sin(2 * face.centre[1]) : face.axis == 1 ? .2 : 0;
  }
  if (!field.Update(u) || !field.Advect(.1, first) || !field.AdvectCorrected(.1, corrected))
    return 2;
  double error_first = 0, error_corrected = 0;
  for (size_t f = 0; f < u.size(); ++f) {
    auto face = mesh.Faces()[f];
    if (face.axis != 0 || std::abs(face.centre[1]) > .6)
      continue;
    double exact = std::sin(2 * (face.centre[1] - .02));
    error_first += std::pow(first[f] - exact, 2);
    error_corrected += std::pow(corrected[f] - exact, 2);
  }
  std::cout << "smooth-wave squared errors first=" << error_first << " corrected=" << error_corrected << '\n';
  if (error_corrected >= .1 * error_first)
    return 3;
  // Repeated small-CFL corrections of a discontinuity must not grow extrema.
  for (size_t f = 0; f < u.size(); ++f) {
    auto face = mesh.Faces()[f];
    u[f] = face.axis == 0 ? (face.centre[1] < 0 ? 0 : 1) : face.axis == 1 ? .2 : 0;
  }
  for (int step = 0; step < 20; ++step) {
    if (!field.Update(u) || !field.AdvectCorrected(.001, corrected, true))
      return 4;
    for (double v : corrected)
      if (v < -1e-12 || v > 1 + 1e-12)
        return 5;
    u = corrected;
  }
  Topology adaptive;
  adaptive.Regrid({2, 2, 2}, 1, 100, [](Block b) { return b.x == 0 && b.y == 0; });
  Mesh mixed;
  mixed.Build(adaptive, 4, 1, {-1, -1, -1});
  FaceField constant(mixed, adaptive, 1, {-1, -1, -1});
  u.assign(mixed.Faces().size(), .3);
  if (!constant.Update(u) || !constant.AdvectCorrected(.01, corrected, true))
    return 6;
  for (double v : corrected)
    if (std::abs(v - .3) > 1e-12)
      return 7;
  std::cout << "PASS original-velocity reverse tracing, small-CFL limiter bounds and adaptive constant preservation\n";
}
