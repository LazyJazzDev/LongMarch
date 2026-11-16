#include "amr_partition.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

int Run(bool mixed, bool embedded, double width) {
  using namespace wind3d::amr;
  Topology tree;
  if (!tree.Regrid({2, 2, 2}, 1, 64, [mixed](Block b) { return !mixed || (b.x == 0 && b.y == 0); }))
    return 1;
  Mesh mesh;
  if (!mesh.Build(tree, 4, width, {-width, -width, -width}))
    return 2;
  if (embedded && !mesh.Embed([width](std::array<double, 3> p) {
        return std::sqrt((p[0] + .5 * width) * (p[0] + .5 * width) + (p[1] + .5 * width) * (p[1] + .5 * width) +
                         (p[2] + .5 * width) * (p[2] + .5 * width)) -
               .18 * width;
      }))
    return 3;
  FaceMass mass;
  if (!mass.Build(mesh))
    return 4;
  size_t nc = mesh.Cells().size(), nf = mesh.Faces().size();
  std::vector<double> u(nf), p(nc);
  for (size_t i = 0; i < nf; ++i)
    u[i] = std::sin(double(i));
  for (size_t i = 0; i < nc; ++i)
    p[i] = std::cos(double(i));
  for (int ranks : {1, 2, 4, 8}) {
    PressurePartition plan;
    if (!plan.Build(mesh, tree, mass, ranks))
      return 5;
    auto owners = plan.CellOwners(), face_owners = plan.FaceOwners();
    if (plan.Build(mesh, tree, mass, 0) || plan.CellOwners() != owners || plan.FaceOwners() != face_owners)
      return 6;
    FaceMass empty;
    if (plan.Build(mesh, tree, empty, ranks) || plan.CellOwners() != owners)
      return 7;
    size_t low = nc, high = 0;
    std::vector<int> seen_cells(nc), seen_faces(nf);
    double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::vector<double>> local_u(ranks, std::vector<double>(nf, nan));
    std::vector<std::vector<double>> local_p(ranks, std::vector<double>(nc, nan));
    for (int r = 0; r < ranks; ++r) {
      low = std::min(low, plan.OwnedCells()[r].size());
      high = std::max(high, plan.OwnedCells()[r].size());
      for (int i : plan.OwnedCells()[r]) {
        if (owners[i] != r || seen_cells[i]++)
          return 8;
        local_p[r][i] = p[i];
      }
      for (int i : plan.OwnedFaces()[r]) {
        if (face_owners[i] != r || seen_faces[i]++)
          return 9;
        local_u[r][i] = u[i];
      }
    }
    if (high - low > 64 || std::count(seen_cells.begin(), seen_cells.end(), 1) != int(nc) ||
        std::count(seen_faces.begin(), seen_faces.end(), 1) != int(nf))
      return 10;
    for (const auto &e : plan.Exchanges()) {
      if (e.source == e.destination || !std::is_sorted(e.cells.begin(), e.cells.end()) ||
          !std::is_sorted(e.faces.begin(), e.faces.end()) ||
          std::adjacent_find(e.cells.begin(), e.cells.end()) != e.cells.end() ||
          std::adjacent_find(e.faces.begin(), e.faces.end()) != e.faces.end())
        return 11;
      for (int i : e.cells) {
        if (owners[i] != e.source)
          return 12;
        local_p[e.destination][i] = local_p[e.source][i];
      }
      for (int i : e.faces) {
        if (face_owners[i] != e.source)
          return 13;
        local_u[e.destination][i] = local_u[e.source][i];
      }
    }
    // NaN-initialized unavailable values make an incomplete import plan fail.
    for (size_t i = 0; i < nc; ++i) {
      double reference = 0, actual = 0;
      for (int k = mesh.Offsets()[i]; k < mesh.Offsets()[i + 1]; ++k) {
        int entry = mesh.Incidence()[k], j = std::abs(entry) - 1;
        double weight = (entry > 0 ? 1 : -1) * mesh.Faces()[j].area * mesh.Faces()[j].aperture;
        reference += weight * u[j];
        actual += weight * local_u[owners[i]][j];
      }
      if (actual != reference)
        return 14;
    }
    for (size_t i = 0; i < nf; ++i) {
      int r = face_owners[i];
      double reference = 0, actual = 0;
      for (int k = mass.Offsets()[i]; k < mass.Offsets()[i + 1]; ++k) {
        int j = mass.Columns()[k];
        reference += mass.Weights()[k] * u[j];
        actual += mass.Weights()[k] * local_u[r][j];
      }
      if (actual != reference)
        return 15;
      auto f = mesh.Faces()[i];
      reference = ((f.low >= 0 ? p[f.low] : 0) - (f.high >= 0 ? p[f.high] : 0)) * f.area * f.aperture;
      actual = ((f.low >= 0 ? local_p[r][f.low] : 0) - (f.high >= 0 ? local_p[r][f.high] : 0)) * f.area * f.aperture;
      if (actual != reference)
        return 16;
    }
  }
  return 0;
}

int main() {
  for (bool mixed : {false, true})
    for (bool embedded : {false, true})
      for (double width : {1., 1.3})
        if (int status = Run(mixed, embedded, width)) {
          std::cerr << "partition failure " << status << '\n';
          return status;
        }
  std::cout << "PASS unique ownership, block balance, directed imports, exact D/D^T/M application and rollback\n";
}
