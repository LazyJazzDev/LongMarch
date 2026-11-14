#include "amr_transfer.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "amr_projection.h"

int main() {
  using namespace wind3d::amr;
  Topology coarse_tree, fine_tree, mixed_tree;
  if (!coarse_tree.Regrid({2, 2, 2}, 0, 8, [](Block) { return false; }) ||
      !fine_tree.Regrid({2, 2, 2}, 1, 64, [](Block) { return true; }) ||
      !mixed_tree.Regrid({2, 2, 2}, 1, 64, [](Block b) { return b.x == 0; }))
    return 1;
  Mesh coarse, fine, mixed;
  if (!coarse.Build(coarse_tree, 4, 1, {-1, -1, -1}) || !fine.Build(fine_tree, 4, 1, {-1, -1, -1}) ||
      !mixed.Build(mixed_tree, 4, 1, {-1, -1, -1}))
    return 2;
  for (const Topology *tree : {&fine_tree, &mixed_tree}) {
    const Mesh &target = tree == &fine_tree ? fine : mixed;
    std::vector<double> u(coarse.Faces().size(), .3), mapped, back;
    if (!TransferVelocity(coarse, coarse_tree, u, target, *tree, 1, {-1, -1, -1}, mapped))
      return 3;
    for (double v : mapped)
      if (std::abs(v - .3) > 1e-14)
        return 4;
    for (size_t i = 0; i < u.size(); ++i) {
      auto f = coarse.Faces()[i];
      u[i] = .2 + std::sin(f.centre[(f.axis + 1) % 3]) + .1 * f.centre[f.axis];
    }
    if (!TransferVelocity(coarse, coarse_tree, u, target, *tree, 1, {-1, -1, -1}, mapped) ||
        !TransferVelocity(target, *tree, mapped, coarse, coarse_tree, 1, {-1, -1, -1}, back))
      return 5;
    auto bounds = std::minmax_element(u.begin(), u.end());
    for (double v : mapped)
      if (v < *bounds.first - 1e-12 || v > *bounds.second + 1e-12)
        return 14;
    double error = 0;
    for (size_t i = 0; i < u.size(); ++i)
      error = std::max(error, std::abs(u[i] - back[i]));
    std::cout << "refine/coarsen face-flux roundtrip error=" << error << '\n';
    if (error > 1e-12)
      return 6;
    FaceGradient gradient;
    if (!gradient.Build(target))
      return 7;
    std::vector<double> potential;
    auto solved = ProjectReference(target, gradient, mapped, potential);
    if (!solved.converged)
      return 8;
    double divergence = 0;
    for (size_t c = 0; c < target.Cells().size(); ++c) {
      double flux = 0;
      for (int k = target.Offsets()[c]; k < target.Offsets()[c + 1]; ++k) {
        int id = target.Incidence()[k], f = std::abs(id) - 1;
        flux += (id > 0 ? 1 : -1) * target.Faces()[f].area * target.Faces()[f].aperture * mapped[f];
      }
      divergence = std::max(divergence, std::abs(flux) / std::pow(target.Cells()[c].width, 3));
    }
    std::cout << "projected transferred max divergence=" << divergence << '\n';
    if (divergence > 1e-7)
      return 9;
    auto saved = mapped;
    u.pop_back();
    if (TransferVelocity(coarse, coarse_tree, u, target, *tree, 1, {-1, -1, -1}, mapped) || mapped != saved)
      return 10;
  }
  // Move the refined region, simultaneously coarsening one side and refining the other.
  Topology shifted_tree;
  Mesh shifted;
  if (!shifted_tree.Regrid({2, 2, 2}, 1, 64, [](Block b) { return b.x == 1; }) ||
      !shifted.Build(shifted_tree, 4, 1, {-1, -1, -1}))
    return 15;
  std::vector<double> moved_input(mixed.Faces().size(), .7), moved;
  if (!TransferVelocity(mixed, mixed_tree, moved_input, shifted, shifted_tree, 1, {-1, -1, -1}, moved))
    return 16;
  for (double v : moved)
    if (std::abs(v - .7) > 1e-14)
      return 17;
  // Unchanged embedded geometry must retain both cut-face flux and solid zeros.
  if (!mixed.Embed([](std::array<double, 3> p) {
        return std::max({std::abs(p[0] + .5), std::abs(p[1]), std::abs(p[2])}) - .21;
      }))
    return 11;
  std::vector<double> u(mixed.Faces().size()), mapped;
  for (size_t i = 0; i < u.size(); ++i)
    u[i] = mixed.Faces()[i].aperture > 0 ? .4 : 0;
  if (!TransferVelocity(mixed, mixed_tree, u, mixed, mixed_tree, 1, {-1, -1, -1}, mapped))
    return 12;
  for (size_t i = 0; i < u.size(); ++i)
    if (std::abs(mapped[i] - u[i]) > 1e-14)
      return 13;
  auto saved = mapped;
  if (TransferVelocity(mixed, mixed_tree, u, coarse, coarse_tree, 1, {-1, -1, -1}, mapped) || mapped != saved)
    return 18;
  if (!fine.Embed([](std::array<double, 3> p) {
        return std::max({std::abs(p[0] + .5), std::abs(p[1]), std::abs(p[2])}) - .21;
      }) ||
      !TransferVelocity(mixed, mixed_tree, u, fine, fine_tree, 1, {-1, -1, -1}, mapped))
    return 19;
  for (size_t i = 0; i < mapped.size(); ++i)
    if (fine.Faces()[i].aperture == 0 && mapped[i] != 0)
      return 20;
  // Even equal aggregate apertures must not permit a changed wall spacing.
  if (!coarse.Embed([](std::array<double, 3> p) {
        return std::max({std::abs(p[0] + .5), std::abs(p[1]), std::abs(p[2])}) - .21;
      }))
    return 21;
  saved = mapped;
  if (TransferVelocity(mixed, mixed_tree, u, coarse, coarse_tree, 1, {-1, -1, -1}, mapped) || mapped != saved)
    return 22;
  std::cout << "PASS conservative transfer, constant preservation, post-transfer projection and rollback\n";
}
