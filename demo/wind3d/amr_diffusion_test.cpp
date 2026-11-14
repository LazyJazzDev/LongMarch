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
  std::vector<double> u(mesh.Faces().size());
  for (size_t f = 0; f < u.size(); ++f)
    u[f] = mesh.Faces()[f].axis == 0 ? std::sin(2 * mesh.Faces()[f].centre[1]) : 0;
  if (!field.Update(u))
    return 2;
  auto diffused = u;
  if (!field.DiffuseAndSponge(.1, .01, 0, 0, diffused))
    return 3;
  double h = .125, decay = 1 - .1 * .01 * 4 * std::pow(std::sin(h), 2) / (h * h);
  for (size_t f = 0; f < u.size(); ++f) {
    auto face = mesh.Faces()[f];
    if (face.axis == 0 && std::abs(face.centre[1]) < .7 && std::abs(diffused[f] - decay * u[f]) > 1e-12)
      return 4;
  }
  auto saved = diffused;
  if (field.DiffuseAndSponge(100, .01, 0, 0, diffused) || diffused != saved)
    return 5;
  std::vector<double> sponge(u.size());
  if (!field.DiffuseAndSponge(.1, 0, .5, 2, sponge))
    return 6;
  for (size_t f = 0; f < u.size(); ++f) {
    auto face = mesh.Faces()[f];
    if (face.axis == 0 && face.high < 0 && std::abs(sponge[f] - .5) > 1e-12)
      return 7;
  }
  if (!mesh.Embed([](std::array<double, 3> p) { return p[1]; }))
    return 8;
  for (size_t f = 0; f < u.size(); ++f)
    u[f] = mesh.Faces()[f].axis == 0 && mesh.Faces()[f].aperture > 0 ? 2 : 0;
  auto extended = u;
  if (!field.ExtendSlip(extended) || !field.Update(extended) || !field.DiffuseAndSponge(.1, .01, 0, 2, u))
    return 9;
  for (size_t f = 0; f < u.size(); ++f) {
    auto face = mesh.Faces()[f];
    double exact = face.axis == 0 && face.aperture > 0 ? 2 : 0;
    if (std::abs(u[f] - exact) > 1e-12)
      return 10;
  }
  std::cout
      << "PASS discrete sine diffusion, stable-step guard, outlet sponge and no artificial slip-wall momentum sink\n";
}
