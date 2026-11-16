#include "amr_mass.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

std::vector<double> Apply(const wind3d::amr::FaceMass &mass, const std::vector<double> &u) {
  std::vector<double> result(u.size());
  for (size_t i = 0; i < u.size(); ++i)
    for (int k = mass.Offsets()[i]; k < mass.Offsets()[i + 1]; ++k)
      result[i] += mass.Weights()[k] * u[mass.Columns()[k]];
  return result;
}

int main() {
  using namespace wind3d::amr;
  for (int split_axis = 0; split_axis < 4; ++split_axis)
    for (double width : {1., 1.3})
      for (bool mixed : {false, true})
        for (bool embedded : {false, true}) {
          Topology tree;
          Mesh mesh;
          FaceMass mass;
          std::array<int, 3> roots{1, 1, 1};
          if (split_axis < 3)
            roots[split_axis] = 2;
          else
            roots = {2, 2, 2};
          if (!tree.Regrid(roots, 1, 64,
                           [mixed, split_axis](Block b) {
                             int coordinate[3]{b.x, b.y, b.z};
                             return !mixed || (split_axis < 3 ? coordinate[split_axis] == 0 : b.x == 0 && b.y == 0);
                           }) ||
              !mesh.Build(tree, 4, width))
            return 1;
          if (embedded && !mesh.Embed([width](std::array<double, 3> p) {
                return std::max(
                           {std::abs(p[0] - .5 * width), std::abs(p[1] - .5 * width), std::abs(p[2] - .5 * width)}) -
                       .2 * width;
              }))
            return 2;
          if (!mass.Build(mesh))
            return 3;
          const auto &faces = mesh.Faces();
          for (size_t i = 0; i < faces.size(); ++i) {
            if (!(mass.Diagonal()[i] > 0))
              return 4;
            for (int k = mass.Offsets()[i]; k < mass.Offsets()[i + 1]; ++k) {
              int j = mass.Columns()[k];
              auto begin = mass.Columns().begin() + mass.Offsets()[j];
              auto end = mass.Columns().begin() + mass.Offsets()[j + 1];
              auto mirror = std::lower_bound(begin, end, int(i));
              if (mirror == end || *mirror != int(i) ||
                  mass.Weights()[mirror - mass.Columns().begin()] != mass.Weights()[k])
                return 5;
            }
            if (!mixed) {
              double expected = faces[i].aperture == 0 ? 1 : faces[i].area * faces[i].distance * faces[i].aperture;
              if (mass.Offsets()[i + 1] - mass.Offsets()[i] != 1 ||
                  std::abs(mass.Diagonal()[i] - expected) > 1e-13 * expected)
                return 6;
            }
          }
          double maximum = 0;
          if (!embedded)
            for (int axis = 0; axis < 3; ++axis) {
              std::vector<double> u(faces.size());
              for (size_t i = 0; i < u.size(); ++i)
                u[i] = faces[i].axis == axis ? 1 : 0;
              auto product = Apply(mass, u);
              for (size_t i = 0; i < u.size(); ++i) {
                auto f = faces[i];
                double low = f.low < 0 ? f.centre[axis] : mesh.Cells()[f.low].centre[axis];
                double high = f.high < 0 ? f.centre[axis] : mesh.Cells()[f.high].centre[axis];
                maximum = std::max(maximum, std::abs(product[i] - f.area * (high - low)));
              }
            }
          if (maximum > 1e-12 * width * width * width)
            return 7;
          std::mt19937 random(123);
          std::uniform_real_distribution<double> distribution(-1, 1);
          for (int trial = 0; trial < 8; ++trial) {
            std::vector<double> u(faces.size());
            for (size_t i = 0; i < u.size(); ++i)
              u[i] = faces[i].aperture == 0 ? 0 : distribution(random);
            auto product = Apply(mass, u);
            double energy = 0;
            for (size_t i = 0; i < u.size(); ++i)
              energy += u[i] * product[i];
            if (!(energy > 0) || !std::isfinite(energy))
              return 8;
          }
          auto weights = mass.Weights();
          auto offsets = mass.Offsets();
          if (mass.Build(Mesh{}) || mass.Weights() != weights || mass.Offsets() != offsets)
            return 9;
          std::cout << "mass split=" << split_axis << " width=" << width << " mixed=" << mixed
                    << " embedded=" << embedded << " linear consistency=" << maximum << " nnz=" << weights.size()
                    << '\n';
        }
  // A body crossing a split cell's other face is unsupported even if the
  // refinement interface itself is open. Reject without losing the old mass.
  Topology tree;
  Mesh mesh;
  FaceMass mass;
  if (!tree.Regrid({2, 1, 1}, 1, 16, [](Block b) { return b.x == 0; }) || !mesh.Build(tree, 4, 1) || !mass.Build(mesh))
    return 10;
  auto saved = mass.Weights();
  if (!mesh.Embed([](std::array<double, 3> p) {
        return std::max({std::abs(p[0] - 1.23), std::abs(p[1] - .375), std::abs(p[2] - .375)}) - .03;
      }) ||
      mass.Build(mesh) || mass.Weights() != saved)
    return 11;
  std::cout << "PASS compatible face mass symmetry, positive energy, linear consistency, cut rows and rollback\n";
}
