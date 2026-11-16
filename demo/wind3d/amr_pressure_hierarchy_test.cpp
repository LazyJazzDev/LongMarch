#include "amr_pressure_hierarchy.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "amr_partition.h"

using namespace wind3d::amr;

std::vector<double> Apply(const PressureLevel &level, const std::vector<double> &x) {
  std::vector<double> y(level.owners.size());
  for (size_t i = 0; i < y.size(); ++i)
    for (int k = level.offsets[i]; k < level.offsets[i + 1]; ++k)
      y[i] += level.weights[k] * x[level.columns[k]];
  return y;
}

bool Close(double a, double b) {
  return std::isfinite(a) && std::isfinite(b) && std::abs(a - b) < 2e-12 * std::max({1., std::abs(a), std::abs(b)});
}

int Run(bool mixed, int geometry, double width, int ranks) {
  Topology tree;
  Mesh mesh;
  FaceMass mass;
  PressurePartition partition;
  PressureHierarchy hierarchy;
  if (!tree.Regrid({2, 2, 2}, 1, 64, [mixed](Block b) { return !mixed || (b.x == 0 && b.y == 0); }) ||
      !mesh.Build(tree, 4, width, {-width, -width, -width}))
    return 1;
  if (geometry && !mesh.Embed([=](std::array<double, 3> p) {
        if (geometry == 2)
          return -1.;
        return std::sqrt((p[0] + .5 * width) * (p[0] + .5 * width) + (p[1] + .5 * width) * (p[1] + .5 * width) +
                         (p[2] + .5 * width) * (p[2] + .5 * width)) -
               .18 * width;
      }))
    return 2;
  if (!mass.Build(mesh) || !partition.Build(mesh, tree, mass, ranks) ||
      !hierarchy.Build(mesh, partition.CellOwners(), ranks))
    return 3;
  const auto &levels = hierarchy.Levels();
  auto old = levels.front().weights;
  if (hierarchy.Build(mesh, {}, ranks) || hierarchy.Build(mesh, partition.CellOwners(), 0) ||
      hierarchy.Build(mesh, partition.CellOwners(), ranks, 0) || levels.front().weights != old)
    return 4;
  if (!levels.back().parent.empty() || (!levels.back().identity && levels.back().owners.size() > 64))
    return 5;
  std::vector<std::vector<std::vector<double>>> local;
  std::vector<std::vector<double>> values;
  for (const auto &level : levels) {
    size_t n = level.owners.size();
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i)
      x[i] = std::sin(.31 * i) + .1;
    values.push_back(x);
    local.emplace_back(ranks, std::vector<double>(n, std::numeric_limits<double>::quiet_NaN()));
    auto &v = local.back();
    std::vector<int> seen(n);
    for (int r = 0; r < ranks; ++r)
      for (int i : level.owned_cells[r]) {
        if (level.owners[i] != r || seen[i]++)
          return 6;
        v[r][i] = x[i];
      }
    if (std::count(seen.begin(), seen.end(), 1) != int(n))
      return 7;
    for (const auto &e : level.exchanges) {
      if (e.source == e.destination || !std::is_sorted(e.cells.begin(), e.cells.end()))
        return 8;
      for (int i : e.cells) {
        if (level.owners[i] != e.source)
          return 9;
        v[e.destination][i] = v[e.source][i];
      }
    }
    auto expected = Apply(level, x);
    for (size_t i = 0; i < n; ++i) {
      double actual = 0;
      for (int k = level.offsets[i]; k < level.offsets[i + 1]; ++k)
        actual += level.weights[k] * v[level.owners[i]][level.columns[k]];
      if (!Close(actual, expected[i]))
        return 10;
    }
    double energy = 0;
    for (size_t i = 0; i < n; ++i)
      energy += x[i] * expected[i];
    if (!(energy > 0))
      return 11;
  }
  // Independently evaluate the fine face-wise pressure stencil.
  const auto &p = values.front();
  std::vector<double> expected(p.size());
  for (const auto &f : mesh.Faces())
    if (f.aperture > 0 && ((f.low >= 0 && f.high >= 0) || (f.axis == 0 && f.high < 0))) {
      double flux = f.area * f.aperture / f.distance * ((f.low >= 0 ? p[f.low] : 0) - (f.high >= 0 ? p[f.high] : 0));
      if (f.low >= 0)
        expected[f.low] += flux;
      if (f.high >= 0)
        expected[f.high] -= flux;
    }
  auto actual = Apply(levels.front(), p);
  for (size_t i = 0; i < p.size(); ++i)
    if (!Close(actual[i], mesh.Cells()[i].active ? expected[i] : p[i]))
      return 12;
  for (size_t d = 0; d + 1 < levels.size(); ++d) {
    const auto &fine = levels[d];
    const auto &coarse = levels[d + 1];
    std::vector<double> prolonged(fine.owners.size()), restricted(coarse.owners.size()), distributed(restricted.size());
    double left = 0, right = 0;
    for (size_t i = 0; i < fine.parent.size(); ++i)
      if (fine.parent[i] >= 0) {
        int j = fine.parent[i];
        prolonged[i] = local[d + 1][fine.owners[i]][j];
        if (!Close(prolonged[i], values[d + 1][j]))
          return 13;
        restricted[j] += values[d][i];
        distributed[j] += local[d][coarse.owners[j]][i];
        left += values[d][i] * prolonged[i];
      }
    for (size_t j = 0; j < restricted.size(); ++j) {
      if (!Close(restricted[j], distributed[j]))
        return 14;
      right += restricted[j] * values[d + 1][j];
    }
    if (!Close(left, right))
      return 15;
    auto fine_product = Apply(fine, prolonged), coarse_product = Apply(coarse, values[d + 1]);
    std::fill(restricted.begin(), restricted.end(), 0);
    for (size_t i = 0; i < fine.parent.size(); ++i)
      if (fine.parent[i] >= 0)
        restricted[fine.parent[i]] += fine_product[i];
    for (size_t j = 0; j < restricted.size(); ++j)
      if (!Close(restricted[j], coarse_product[j]))
        return 16;
  }
  return 0;
}

int main() {
  for (bool mixed : {false, true})
    for (int geometry : {0, 1, 2})
      for (double width : {1., 1.3})
        for (int ranks : {1, 2, 4, 8})
          if (!(mixed && geometry == 2))
            if (int result = Run(mixed, geometry, width, ranks)) {
              std::cerr << "hierarchy failure " << result << " mixed=" << mixed << " geometry=" << geometry
                        << " width=" << width << " ranks=" << ranks << '\n';
              return result;
            }
  std::cout << "PASS fine pressure, Galerkin identity, adjoint transfers, distributed A/R/P imports and rollback\n";
}
