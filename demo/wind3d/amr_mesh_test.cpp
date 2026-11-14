#include "amr_mesh.h"

#include <cmath>
#include <iostream>
#include <random>

int main() {
  using namespace wind3d::amr;
  Topology tree;
  if (!tree.Regrid({4, 2, 3}, 2, 10000, [](Block b) { return b.x < 2 && b.y < 2 && b.z < 2; }))
    return 1;
  Mesh mesh;
  if (!mesh.Build(tree, 4, 2, {-6, -4, -6}))
    return 2;
  const auto &cells = mesh.Cells();
  const auto &faces = mesh.Faces();
  const auto &offsets = mesh.Offsets();
  const auto &incidence = mesh.Incidence();
  std::vector<int> uses(faces.size());
  std::mt19937 rng(43);
  std::uniform_real_distribution<double> uniform(-1, 1);
  std::vector<double> flux(faces.size()), p(cells.size()), q(cells.size());
  for (auto &v : flux)
    v = uniform(rng);
  for (auto &v : p)
    v = uniform(rng);
  for (auto &v : q)
    v = uniform(rng);
  double volume = 0, total = 0, boundary = 0, pAq = 0, qAp = 0, energy = 0;
  int transitions = 0;
  for (size_t i = 0; i < cells.size(); ++i) {
    std::array<double, 3> balance{};
    double linear = 0, random = 0, ap = 0, aq = 0;
    for (int k = offsets[i]; k < offsets[i + 1]; ++k) {
      int index = std::abs(incidence[k]) - 1, sign = incidence[k] > 0 ? 1 : -1;
      const auto &f = faces[index];
      ++uses[index];
      if ((sign > 0 ? f.low : f.high) != int(i))
        return 3;
      balance[f.axis] += sign * f.area;
      linear += sign * f.area * f.centre[f.axis];
      random += sign * f.area * flux[index];
      int other = sign > 0 ? f.high : f.low;
      if (other >= 0) {
        ap += f.area / f.distance * (p[i] - p[other]);
        aq += f.area / f.distance * (q[i] - q[other]);
      }
    }
    double v = std::pow(cells[i].width, 3);
    volume += v;
    for (double sum : balance)
      if (std::abs(sum) > 1e-12)
        return 4;
    // div(x,y,z)=3 in every cell, including coarse/fine interfaces.
    if (std::abs(linear - 3 * v) > 1e-11)
      return 5;
    total += random;
    pAq += p[i] * aq;
    qAp += q[i] * ap;
    energy += p[i] * ap;
  }
  for (size_t i = 0; i < faces.size(); ++i) {
    const auto &f = faces[i];
    if (uses[i] != (f.low >= 0) + (f.high >= 0))
      return 6;
    if (f.low < 0)
      boundary -= f.area * flux[i];
    if (f.high < 0)
      boundary += f.area * flux[i];
    if (f.low >= 0 && f.high >= 0 && cells[f.low].width != cells[f.high].width)
      ++transitions;
  }
  if (!transitions || std::abs(volume - 192) > 1e-10)
    return 7;
  if (std::abs(total - boundary) > 1e-10)
    return 8;
  if (std::abs(pAq - qAp) > 1e-9 || energy <= 0)
    return 9;
  const size_t before = cells.size();
  if (mesh.Build(tree, 3, 2) || mesh.Cells().size() != before)
    return 10;
  std::cout
      << "PASS exact face coverage, constant/linear flux divergence, global flux cancellation and graph symmetry/positive energy; cells="
      << cells.size() << " interfaces=" << transitions << '\n';
}
